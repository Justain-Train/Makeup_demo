"""Fetcher for a logical URL that answers 302 to a short-lived signed CDN URL (Hugging Face resolve links).

The logical URL is the only URL callers see (and the one BudgetedFetcher pins). The signed URL is kept in a
private attribute and is never logged, repr'd or put into an exception message.
"""

from __future__ import annotations

import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

from evaluation.remote_zip import USER_AGENT, FetchError, Head, HttpsRangeFetcher, RangeFetcher

ALLOWED_SUFFIXES = (".hf.co",)
_REJECTED = "redirect target rejected"
_RETRY_MESSAGES = ("HTTP 403", "HTTP 410")


class _KeepRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None  # surfaces the 3xx as an HTTPError whose headers we can read


def validate_location(location: str, allowed_suffixes: tuple[str, ...] = ALLOWED_SUFFIXES) -> str:
    """Return the location unchanged if it is an acceptable https CDN URL, else raise FetchError."""
    try:
        parts = urllib.parse.urlsplit(location)
        host = parts.hostname
        port = parts.port
    except ValueError:
        raise FetchError(_REJECTED) from None
    if parts.scheme != "https" or not host or "@" in parts.netloc or parts.username or parts.password:
        raise FetchError(_REJECTED)
    if port not in (None, 443):
        raise FetchError(_REJECTED)
    host = host.lower()
    if not any(host == s.lstrip(".") or host.endswith(s) for s in allowed_suffixes):
        raise FetchError(_REJECTED)
    return location


def resolve_signed_url(logical_url: str, allowed_suffixes: tuple[str, ...] = ALLOWED_SUFFIXES) -> str:
    if urllib.parse.urlsplit(logical_url).scheme != "https":
        raise FetchError("only https URLs are allowed")
    opener = urllib.request.build_opener(_KeepRedirect)
    attempts = (
        urllib.request.Request(logical_url, method="HEAD", headers={"User-Agent": USER_AGENT}),
        urllib.request.Request(logical_url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"}),
    )
    for req in attempts:
        try:
            with opener.open(req, timeout=60) as r:
                r.read(1)
            continue  # a direct answer is not a redirect: try the next form, then fail
        except urllib.error.HTTPError as exc:
            if exc.code in (302, 307):
                location = exc.headers.get("Location")
                if not location:
                    raise FetchError(_REJECTED) from None
                return validate_location(location, allowed_suffixes)
            continue
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise FetchError(f"resolve failed ({type(exc).__name__})") from None
    raise FetchError("resolve failed (no redirect)")


class ResolvingFetcher:
    def __init__(
        self,
        logical_url: str,
        allowed_suffixes: tuple[str, ...] = ALLOWED_SUFFIXES,
        resolve_fn: Callable[[str], str] | None = None,
        inner: RangeFetcher | None = None,
    ) -> None:
        self._logical = logical_url
        self._suffixes = allowed_suffixes
        self._resolve_fn = resolve_fn or (lambda u: resolve_signed_url(u, allowed_suffixes))
        self._inner: RangeFetcher = inner or HttpsRangeFetcher()
        self._lock = threading.Lock()
        self._signed: str | None = None
        self._generation = 0
        self.resolutions = 0

    def __repr__(self) -> str:
        return "ResolvingFetcher()"

    def _signed_url(self, stale_generation: int | None = None) -> tuple[str, int]:
        with self._lock:
            if self._signed is None or (stale_generation is not None and stale_generation == self._generation):
                self._signed = validate_location(self._resolve_fn(self._logical), self._suffixes)
                self._generation += 1
                self.resolutions += 1
            return self._signed, self._generation

    def _call(self, url: str, op: Callable[[str], object]) -> object:
        if url != self._logical:
            raise FetchError("unexpected URL")
        signed, gen = self._signed_url()
        try:
            return op(signed)
        except FetchError as exc:
            if str(exc) not in _RETRY_MESSAGES:
                raise
        signed, _ = self._signed_url(stale_generation=gen)
        return op(signed)

    def head(self, url: str) -> Head:
        return self._call(url, self._inner.head)  # type: ignore[return-value]

    def get_range(self, url: str, start: int, end: int) -> bytes:
        return self._call(url, lambda s: self._inner.get_range(s, start, end))  # type: ignore[return-value]
