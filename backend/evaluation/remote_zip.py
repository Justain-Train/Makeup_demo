"""Read individual members of a remote zip with HTTP Range requests (standard library only).

Guards (each covered by an offline test with a fake fetcher):
  * https only; redirects are refused;
  * a ranged GET must answer 206 with a matching Content-Range (a 200 aborts: the server ignored the range);
  * Content-Length of every archive is pinned and checked by HEAD;
  * a byte budget for the whole session;
  * member size caps, checked before and while decompressing;
  * CRC-32 and size are verified on read;
  * nothing is ever extracted to a path: members are returned as bytes.
"""

from __future__ import annotations

import re
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass
from typing import Protocol

USER_AGENT = "makeup-demo-eval/1"
TIMEOUT_S = 60
MAX_TAIL_BYTES = 32 * 1024 * 1024  # central directory limit
MAX_MEMBERS = 200_000
LOCAL_HEADER_FIXED = 30
LOCAL_EXTRA_GUESS = 256


class FetchError(Exception):
    """Any violation of a fetch guard (the message never contains image data)."""


@dataclass(frozen=True)
class Head:
    length: int
    etag: str | None


class RangeFetcher(Protocol):
    def head(self, url: str) -> Head: ...

    def get_range(self, url: str, start: int, end: int) -> bytes:
        """Bytes [start, end] inclusive. Must raise FetchError unless the server honoured the range."""
        ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise FetchError(f"redirect refused (HTTP {code})")


class HttpsRangeFetcher:
    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(_NoRedirect)

    @staticmethod
    def _check_url(url: str) -> None:
        if urllib.parse.urlsplit(url).scheme != "https":
            raise FetchError("only https URLs are allowed")

    def head(self, url: str) -> Head:
        self._check_url(url)
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        try:
            with self._opener.open(req, timeout=TIMEOUT_S) as r:
                length = r.headers.get("Content-Length")
                if r.status != 200 or length is None or not length.isdigit():
                    raise FetchError("unexpected HEAD response")
                return Head(int(length), r.headers.get("ETag"))
        except urllib.error.URLError as exc:
            raise FetchError(f"HEAD failed ({type(exc).__name__})") from None

    def get_range(self, url: str, start: int, end: int) -> bytes:
        self._check_url(url)
        if start < 0 or end < start:
            raise FetchError("bad range")
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        )
        try:
            with self._opener.open(req, timeout=TIMEOUT_S) as r:
                if r.status != 206:
                    # A 200 would stream the whole archive: close without reading and abort.
                    raise FetchError(f"server did not honour the range (HTTP {r.status})")
                m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", r.headers.get("Content-Range", ""))
                if not m or int(m.group(1)) != start or int(m.group(2)) != end:
                    raise FetchError("Content-Range does not match the request")
                data = r.read(end - start + 2)  # one byte more than asked: detects an over-long body
        except urllib.error.HTTPError as exc:
            raise FetchError(f"HTTP {exc.code}") from None
        except urllib.error.URLError as exc:
            raise FetchError(f"request failed ({type(exc).__name__})") from None
        except (TimeoutError, OSError) as exc:
            raise FetchError(f"request failed ({type(exc).__name__})") from None
        if len(data) != end - start + 1:
            raise FetchError("short or long body")
        return data


class BudgetedFetcher:
    """Wraps a fetcher: pinned archive sizes, a total byte budget, byte and time accounting."""

    def __init__(self, inner: RangeFetcher, budget_bytes: int, pins: dict[str, int]) -> None:
        self._inner, self._budget, self._pins = inner, budget_bytes, dict(pins)
        self._lock = threading.Lock()
        self.bytes_received = 0
        self.requests = 0
        self._t0 = time.perf_counter()

    def verified_length(self, url: str) -> int:
        """HEAD, then compare with the pinned Content-Length. No bytes of the body are read."""
        pinned = self._pins.get(url)
        if pinned is None:
            raise FetchError("URL is not pinned")
        head = self._inner.head(url)
        if head.length != pinned:
            raise FetchError(f"Content-Length changed (pinned {pinned}, got {head.length})")
        return head.length

    def pinned_length(self, url: str) -> int:
        if url not in self._pins:
            raise FetchError("URL is not pinned")
        return self._pins[url]

    def get_range(self, url: str, start: int, end: int) -> bytes:
        pinned = self._pins.get(url)
        if pinned is None:
            raise FetchError("URL is not pinned")
        if end >= pinned:
            raise FetchError("range beyond the pinned archive size")
        with self._lock:  # reserve first, so concurrent threads cannot overshoot
            if self.bytes_received + (end - start + 1) > self._budget:
                raise FetchError("byte budget exceeded")
            self.bytes_received += end - start + 1
            self.requests += 1
        return self._inner.get_range(url, start, end)

    @property
    def throughput_mb_s(self) -> float:
        dt = time.perf_counter() - self._t0
        return self.bytes_received / 1e6 / dt if dt > 0 else 0.0


@dataclass(frozen=True)
class MemberInfo:
    name: str
    header_offset: int
    compress_size: int
    file_size: int
    crc: int
    method: int


class _TailFile:
    """Seekable view of a big file of which only the last bytes are known."""

    class NeedMore(Exception):
        def __init__(self, pos: int) -> None:
            self.pos = pos

    def __init__(self, tail: bytes, total: int) -> None:
        self._tail, self._total, self._pos = tail, total, 0
        self._start = total - len(tail)

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, off: int, whence: int = 0) -> int:
        self._pos = {0: off, 1: self._pos + off, 2: self._total + off}[whence]
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self._total - self._pos
        if n == 0:
            return b""
        if self._pos < self._start:
            raise _TailFile.NeedMore(self._pos)
        out = self._tail[self._pos - self._start : self._pos - self._start + n]
        self._pos += len(out)
        return out


def read_central_directory(fetcher: BudgetedFetcher, url: str) -> dict[str, MemberInfo]:
    """Member table of a remote zip, from its central directory only."""
    total = fetcher.verified_length(url)
    tail_len = min(total, 256 * 1024)
    while True:
        tail = fetcher.get_range(url, total - tail_len, total - 1)
        try:
            with zipfile.ZipFile(_TailFile(tail, total)) as zf:  # type: ignore[arg-type]
                infos = zf.infolist()
            break
        except _TailFile.NeedMore as need:
            new_len = min(total, max(tail_len * 2, total - need.pos + 64 * 1024))
            if new_len > MAX_TAIL_BYTES or new_len <= tail_len:
                raise FetchError("central directory too large") from None
            tail_len = new_len
        except zipfile.BadZipFile:
            raise FetchError("not a readable zip central directory") from None
    if len(infos) > MAX_MEMBERS:
        raise FetchError("too many members")
    table: dict[str, MemberInfo] = {}
    for i in infos:
        if i.flag_bits & 0x1:
            raise FetchError("encrypted member")
        table[i.filename] = MemberInfo(i.filename, i.header_offset, i.compress_size, i.file_size, i.CRC, i.compress_type)
    return table


def fetch_member(fetcher: BudgetedFetcher, url: str, info: MemberInfo, max_size: int) -> bytes:
    """Download, decompress and verify (size cap, CRC-32, size) one member. Never touches the file system."""
    if info.file_size > max_size or info.compress_size > max_size:
        raise FetchError("member exceeds the size cap")
    if info.method not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise FetchError("unsupported compression")
    name_len = len(info.name.encode("utf-8"))
    first_len = LOCAL_HEADER_FIXED + name_len + LOCAL_EXTRA_GUESS + info.compress_size
    last = fetcher.pinned_length(url) - 1
    if info.header_offset > last:
        raise FetchError("member offset beyond the archive")
    raw = fetcher.get_range(url, info.header_offset, min(info.header_offset + first_len - 1, last))
    if raw[:4] != b"PK\x03\x04":
        raise FetchError("bad local header")
    fn_len, extra_len = struct.unpack("<HH", raw[26:30])
    data_start = LOCAL_HEADER_FIXED + fn_len + extra_len
    if raw[LOCAL_HEADER_FIXED : LOCAL_HEADER_FIXED + fn_len] != info.name.encode("utf-8"):
        raise FetchError("local header name mismatch")
    need = data_start + info.compress_size
    if need > len(raw):
        raw += fetcher.get_range(url, info.header_offset + len(raw), info.header_offset + need - 1)
    comp = raw[data_start:need]
    if len(comp) != info.compress_size:
        raise FetchError("truncated member")
    if info.method == zipfile.ZIP_STORED:
        data = comp
    else:
        d = zlib.decompressobj(-zlib.MAX_WBITS)
        data = d.decompress(comp, max_size + 1)
        if d.unconsumed_tail or len(data) > max_size:
            raise FetchError("member exceeds the size cap while decompressing")
        data += d.flush()
    if len(data) != info.file_size or len(data) > max_size:
        raise FetchError("member size mismatch")
    if zlib.crc32(data) & 0xFFFFFFFF != info.crc:
        raise FetchError("CRC mismatch")
    return data
