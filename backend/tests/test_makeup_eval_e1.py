"""Offline tests for evaluation step E-1 (HF resolve, FFHQ-Makeup index/sample/fetch, contact sheet).

No network: an autouse fixture makes any socket connect fail. Remote archives are served by a fake RangeFetcher.
"""

from __future__ import annotations

import argparse
import io
import socket
import urllib.error
import zipfile
import zlib
from email.message import Message
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evaluation import cli
from evaluation import hf_resolve as hr
from evaluation import makeup_dataset as md
from evaluation.dataset import REPO_ROOT, CacheError
from evaluation.makeup_pipeline import PairResult
from evaluation.makeup_sheet import COLUMNS, HEADER, TILE, build_sheet
from evaluation.remote_zip import BudgetedFetcher, FetchError, Head
from app.services.makeup_preview.renderer import RenderResult

LOGICAL = "https://huggingface.co/datasets/cyberagent/FFHQ-Makeup/resolve/main/FFHQ-Makeup.zip"
CANARY = "SECRET123"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("network access attempted in offline test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def jpeg(size: int = 512) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (120, 80, 60)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def make_zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


class FakeRange:
    """Serves one archive; records calls; `fail_once` maps a call index to an exception."""

    def __init__(self, blob: bytes) -> None:
        self.blob = blob
        self.calls = 0

    def head(self, url: str) -> Head:
        return Head(len(self.blob), None)

    def get_range(self, url: str, start: int, end: int) -> bytes:
        self.calls += 1
        assert 0 <= start <= end < len(self.blob), "range outside archive"
        return self.blob[start : end + 1]


def bf_for(fake: FakeRange, budget: int = 10**9) -> BudgetedFetcher:
    return BudgetedFetcher(fake, budget, {md.URL: len(fake.blob)})


def full_identity(i: str, make=jpeg) -> dict[str, bytes]:
    return {f"{i}/{k}.jpg": make() for k in md.KINDS}


# --- criterion 1: HF resolve ------------------------------------------------------------------------

@pytest.mark.parametrize(
    "loc",
    ["https://us.aws.cdn.hf.co/x?sig=1", "https://hf.co/x", "https://cdn-lfs.hf.co/a/b"],
)
def test_validate_location_accepts(loc):
    assert hr.validate_location(loc) == loc


@pytest.mark.parametrize(
    "loc",
    [
        "http://us.aws.cdn.hf.co/x",
        "https://hf.co.evil.com/x",
        "https://evilhf.co/x",
        "https://a@x.hf.co/",
        "https://user:pw@x.hf.co/",
        "https://x.hf.co:8080/x",
        "https:///x",
        "",
    ],
)
def test_validate_location_rejects(loc):
    with pytest.raises(FetchError, match="redirect target rejected"):
        hr.validate_location(loc)


class _Resp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        return b""


class FakeOpener:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    def open(self, req, timeout=None):
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


def http_redirect(code: int, location: str | None) -> urllib.error.HTTPError:
    hdrs = Message()
    if location is not None:
        hdrs["Location"] = location
    return urllib.error.HTTPError(LOGICAL, code, "redirect", hdrs, None)


def patch_opener(monkeypatch, outcomes):
    monkeypatch.setattr(hr.urllib.request, "build_opener", lambda *_h: FakeOpener(outcomes))


def test_resolve_accepts_302_and_307(monkeypatch):
    patch_opener(monkeypatch, [http_redirect(302, "https://us.aws.cdn.hf.co/x?sig=1")])
    assert hr.resolve_signed_url(LOGICAL) == "https://us.aws.cdn.hf.co/x?sig=1"
    patch_opener(monkeypatch, [http_redirect(307, "https://hf.co/y")])
    assert hr.resolve_signed_url(LOGICAL) == "https://hf.co/y"


@pytest.mark.parametrize("code", [301, 303, 308])
def test_resolve_rejects_other_redirect_statuses(monkeypatch, code):
    patch_opener(monkeypatch, [http_redirect(code, "https://us.aws.cdn.hf.co/x"), http_redirect(code, "https://us.aws.cdn.hf.co/x")])
    with pytest.raises(FetchError, match="resolve failed"):
        hr.resolve_signed_url(LOGICAL)


def test_resolve_refuses_evil_location_and_missing_location(monkeypatch):
    patch_opener(monkeypatch, [http_redirect(302, f"https://evil.com/?sig={CANARY}")])
    with pytest.raises(FetchError, match="redirect target rejected") as ei:
        hr.resolve_signed_url(LOGICAL)
    assert CANARY not in str(ei.value)
    patch_opener(monkeypatch, [http_redirect(302, None)])
    with pytest.raises(FetchError, match="redirect target rejected"):
        hr.resolve_signed_url(LOGICAL)


def test_resolve_rejects_non_https_logical_url():
    with pytest.raises(FetchError, match="only https"):
        hr.resolve_signed_url("http://hf.co/x")


def test_resolve_network_error_does_not_leak_canary(monkeypatch):
    err = urllib.error.URLError(f"failed https://us.aws.cdn.hf.co/x?sig={CANARY}")
    patch_opener(monkeypatch, [err])
    with pytest.raises(FetchError, match="resolve failed") as ei:
        hr.resolve_signed_url(LOGICAL)
    assert CANARY not in str(ei.value)


def test_resolving_fetcher_reresolves_once_on_403(caplog, capsys):
    class Inner:
        def __init__(self):
            self.urls, self.script = [], [FetchError("HTTP 403"), b"abcd"]

        def head(self, url):  # pragma: no cover - not used
            raise AssertionError

        def get_range(self, url, start, end):
            self.urls.append(url)
            out = self.script.pop(0)
            if isinstance(out, BaseException):
                raise out
            return out

    n = {"c": 0}

    def resolve(url):
        n["c"] += 1
        return f"https://us.aws.cdn.hf.co/x?sig={n['c']}"

    inner = Inner()
    f = hr.ResolvingFetcher(LOGICAL, resolve_fn=resolve, inner=inner)
    assert f.get_range(LOGICAL, 0, 3) == b"abcd"
    assert f.resolutions == 2
    assert inner.urls == ["https://us.aws.cdn.hf.co/x?sig=1", "https://us.aws.cdn.hf.co/x?sig=2"]
    assert f._signed == "https://us.aws.cdn.hf.co/x?sig=2"  # the re-resolved URL is kept for later calls

    # a second 403 after the single re-resolve is not retried again
    inner2 = Inner()
    inner2.script = [FetchError("HTTP 403"), FetchError("HTTP 403")]
    f2 = hr.ResolvingFetcher(LOGICAL, resolve_fn=resolve, inner=inner2)
    with pytest.raises(FetchError, match="HTTP 403"):
        f2.get_range(LOGICAL, 0, 3)
    assert f2.resolutions == 2

    # other errors are not retried
    inner3 = Inner()
    inner3.script = [FetchError("HTTP 500")]
    f3 = hr.ResolvingFetcher(LOGICAL, resolve_fn=resolve, inner=inner3)
    with pytest.raises(FetchError, match="HTTP 500"):
        f3.get_range(LOGICAL, 0, 3)
    assert f3.resolutions == 1

    with pytest.raises(FetchError, match="unexpected URL"):
        f3.get_range("https://hf.co/other", 0, 1)


def test_signed_query_never_in_errors_or_logs(caplog, capsys):
    import logging

    caplog.set_level(logging.DEBUG)
    signed = f"https://us.aws.cdn.hf.co/x?sig={CANARY}"

    class Inner:
        def head(self, url):
            raise FetchError("HTTP 500")

        def get_range(self, url, start, end):
            raise FetchError("HTTP 500")

    msgs = []
    f_bad = hr.ResolvingFetcher(LOGICAL, resolve_fn=lambda u: f"https://evil.com/?sig={CANARY}", inner=Inner())
    try:
        f_bad.head(LOGICAL)
    except FetchError as exc:
        msgs.append(str(exc))
    f_ok = hr.ResolvingFetcher(LOGICAL, resolve_fn=lambda u: signed, inner=Inner())
    try:
        f_ok.get_range(LOGICAL, 0, 1)
    except FetchError as exc:
        msgs.append(str(exc))
    msgs.append(repr(f_ok))
    out = capsys.readouterr()
    for text in msgs + [out.out, out.err, caplog.text]:
        assert CANARY not in text


# --- criterion 2: member regex and index -------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    ["../x/bare.jpg", "000001/bare.jpeg", "00001/bare.jpg", "000001/makeup_06.jpg", "/000001/bare.jpg", "000001/makeup_00.jpg"],
)
def test_member_regex_rejects(name):
    assert md.MEMBER_RE.fullmatch(name) is None


def test_member_regex_accepts_valid():
    assert md.MEMBER_RE.fullmatch("000001/bare.jpg")
    assert md.MEMBER_RE.fullmatch("123456/makeup_05.jpg")


def test_only_complete_identities_indexed():
    names = set()
    for k in md.KINDS:
        names.add(f"000001/{k}.jpg")
    for k in md.KINDS[:-1]:
        names.add(f"000002/{k}.jpg")  # missing makeup_05
    names |= {f"000003/{k}.jpg" for k in md.KINDS} | {"../x/bare.jpg", "00004/bare.jpg", "000004/makeup_06.jpg"}
    assert md.complete_ids(names) == ["000001", "000003"]


def test_load_or_build_index_reads_once_then_caches(tmp_path):
    members = {}
    members.update(full_identity("000001"))
    members.update({f"000002/{k}.jpg": jpeg() for k in md.KINDS[:-1]})
    fake = FakeRange(make_zip(members))
    cache = md.ensure_cache(tmp_path / "cache")
    ids, table = md.load_or_build_index(cache, bf_for(fake))
    assert ids == ["000001"] and table is not None and "000001/bare.jpg" in table
    fake.calls = 0
    ids2, table2 = md.load_or_build_index(cache, bf_for(fake))
    assert ids2 == ["000001"] and table2 is None and fake.calls == 0


# --- criterion 3: sampling and sample file -------------------------------------------------------------

IDS = [f"{n:06d}" for n in range(1, 301)]


def test_draw_samples_deterministic_and_seed_sensitive():
    a = md.draw_samples(IDS, 7, 40, 15, 8)
    b = md.draw_samples(IDS, 7, 40, 15, 8)
    c = md.draw_samples(IDS, 8, 40, 15, 8)
    assert a == b
    assert a != c


def test_cross_triples_and_sheet_subset():
    s = md.draw_samples(IDS, 11, 40, 60, 8)
    assert len(s.cross) == 60
    for x, y, k in s.cross:
        assert x != y and x in IDS and y in IDS
        assert 1 <= k <= 5
    assert set(s.sheet) <= set(s.pairs)
    assert len(s.sheet) == 8 and len(set(s.pairs)) == 40


def test_draw_samples_rejects_oversized_requests():
    with pytest.raises(ValueError):
        md.draw_samples(IDS[:5], 1, 10, 2, 2)
    with pytest.raises(ValueError):
        md.draw_samples(IDS, 1, 10, 2, 11)


def test_format_parse_round_trip():
    s = md.draw_samples(IDS, 5, 30, 12, 6)
    assert md.parse_samples(md.format_samples(s, 5, 30)) == s


@pytest.mark.parametrize(
    "text",
    [
        "pair\t000001\nsheet\t000002\n",  # sheet not subset of pairs
        "pair\t000001\ncross\t000001\t000001\t2\n",  # j == i
        "pair\t000001\ncross\t000001\t000002\t6\n",  # k out of range
        "pair\t../etc\n",  # invalid id
        "bogus\t000001\n",
    ],
)
def test_parse_rejects_bad_samples(text):
    with pytest.raises(ValueError):
        md.parse_samples(text)


def test_required_files_are_sorted_and_valid():
    s = md.draw_samples(IDS, 2, 5, 4, 2)
    req = md.required_files(s)
    assert req == sorted(set(req))
    assert all(md.require_id(i) and md.require_kind(k) for i, k in req)


# --- criterion 4: fetch -----------------------------------------------------------------------------------

def test_fetch_writes_only_images_under_cache(tmp_path):
    members = {}
    for i in ("000001", "000002"):
        members.update(full_identity(i))
    fake = FakeRange(make_zip(members))
    cache = tmp_path / "cache"
    files = [("000001", k) for k in md.KINDS] + [("000002", k) for k in md.KINDS]
    s = md.fetch_makeup(cache, files, bf=bf_for(fake), table=None, workers=1)
    assert s.downloaded == 12 and s.failed == 0 and s.skipped == 0 and not s.budget_reached
    rel = {p.relative_to(cache).as_posix() for p in cache.rglob("*") if p.is_file()}
    expected = {md.MARKER_NAME, "manifest.json"} | {f"images/{i}_{k}.jpg" for i, k in files}
    assert rel == expected
    assert not any(p.is_dir() and p.name in ("000001", "000002") for p in cache.rglob("*"))
    assert md.load_image(cache, "000001", "bare").shape == (512, 512, 3)


def test_fetch_stops_at_budget_and_resumes(tmp_path):
    members = {}
    for i in ("000001", "000002", "000003"):
        members.update(full_identity(i))
    blob = make_zip(members)
    files = [(i, k) for i in ("000001", "000002", "000003") for k in md.KINDS]
    cache = tmp_path / "cache"
    per_member = len(blob) // len(files) + 400
    fake = FakeRange(blob)
    table = md.read_central_directory(bf_for(fake), md.URL)  # directory read on a generous budget first
    s1 = md.fetch_makeup(cache, files, bf=bf_for(fake, budget=int(per_member * 4)), table=table, workers=1)
    assert s1.budget_reached
    assert 1 <= s1.downloaded < len(files)
    assert s1.bytes_received <= int(per_member * 4)

    fake2 = FakeRange(blob)
    s2 = md.fetch_makeup(cache, files, bf=bf_for(fake2), table=None, workers=1)
    assert s2.skipped == s1.downloaded
    assert s2.downloaded == len(files) - s1.downloaded and s2.failed == 0
    assert not s2.budget_reached

    fake3 = FakeRange(blob)
    s3 = md.fetch_makeup(cache, files, bf=bf_for(fake3), table=None, workers=1)
    assert s3.skipped == len(files) and s3.downloaded == 0 and fake3.calls == 0


def test_fetch_redownloads_when_manifest_hash_mismatches(tmp_path):
    members = full_identity("000001")
    fake = FakeRange(make_zip(members))
    cache = tmp_path / "cache"
    files = [("000001", "bare")]
    md.fetch_makeup(cache, files, bf=bf_for(fake), table=None, workers=1)
    (cache / "images" / "000001_bare.jpg").write_bytes(b"\xff\xd8\xff tampered")
    s = md.fetch_makeup(cache, files, bf=bf_for(FakeRange(make_zip(members))), table=None, workers=1)
    assert s.downloaded == 1 and s.skipped == 0


def test_fetch_rejects_bad_magic_and_wrong_size(tmp_path):
    png = io.BytesIO()
    Image.new("RGB", (512, 512), (1, 2, 3)).save(png, format="PNG")
    members = {
        "000001/bare.jpg": png.getvalue(),  # wrong magic bytes
        "000001/makeup_01.jpg": jpeg(64),  # JPEG but not 512x512
        "000001/makeup_02.jpg": b"",  # empty
        "000001/makeup_03.jpg": jpeg(),
    }
    fake = FakeRange(make_zip(members))
    cache = tmp_path / "cache"
    files = [("000001", k) for k in ("bare", "makeup_01", "makeup_02", "makeup_03")]
    s = md.fetch_makeup(cache, files, bf=bf_for(fake), table=None, workers=1)
    assert s.downloaded == 1 and s.failed == 3
    assert (cache / "images" / "000001_makeup_03.jpg").is_file()
    assert not (cache / "images" / "000001_bare.jpg").exists()


def test_fetch_missing_member_counts_failed(tmp_path):
    fake = FakeRange(make_zip(full_identity("000001")))
    s = md.fetch_makeup(tmp_path / "c", [("000009", "bare")], bf=bf_for(fake), table=None, workers=1)
    assert s.failed == 1 and s.downloaded == 0


def test_fetch_rejects_invalid_ids_before_network(tmp_path):
    fake = FakeRange(make_zip(full_identity("000001")))
    with pytest.raises(ValueError):
        md.fetch_makeup(tmp_path / "c", [("../x", "bare")], bf=bf_for(fake), table=None)
    with pytest.raises(ValueError):
        md.fetch_makeup(tmp_path / "c", [("000001", "makeup_06")], bf=bf_for(fake), table=None)
    assert fake.calls == 0


def test_cache_refuses_repo_and_non_empty_unmarked_dirs(tmp_path):
    with pytest.raises(CacheError):
        md.resolve_cache_dir(REPO_ROOT / "cache_e1")
    with pytest.raises(CacheError):
        md.ensure_cache(REPO_ROOT / "cache_e1")
    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "file.txt").write_text("x")
    with pytest.raises(CacheError):
        md.ensure_cache(stray)
    assert not (REPO_ROOT / "cache_e1").exists()


# --- criterion 5: sheet and CLI refuse repo paths -----------------------------------------------------

class FakeRunner:
    def __init__(self):
        self.calls = 0

    def run(self, selfie, reference):
        self.calls += 1
        if self.calls == 2:
            return PairResult(False, None, None, None, "profile_no_face", 0.0)
        render = RenderResult(image=np.full((512, 512, 3), 200, np.uint8), reports=(), alpha_union=np.zeros((512, 512), np.float32))
        return PairResult(True, None, None, render, None, 0.0)


def _stage_cache(tmp_path: Path, samples: md.MakeupSamples) -> Path:
    cache = md.ensure_cache(tmp_path / "cache")
    for i, k in md.required_files(samples):
        md.image_path(cache, i, k).write_bytes(jpeg())
    return cache


def test_build_sheet_fake_runner_grid_and_skip_row(tmp_path):
    ids = [f"{n:06d}" for n in range(1, 11)]
    s = md.draw_samples(ids, 3, n_pairs=6, n_cross=5, n_sheet=3)
    cache = _stage_cache(tmp_path, s)
    runner = FakeRunner()
    out = build_sheet(cache, s, runner, tmp_path / "out" / "contact_sheet.png")
    assert runner.calls == 3 + 4  # sheet rows + first four cross rows
    im = Image.open(out)
    assert im.format == "PNG"
    assert im.size == (TILE * len(COLUMNS), HEADER + TILE * 7)


def test_build_sheet_refuses_output_inside_repo(tmp_path):
    ids = [f"{n:06d}" for n in range(1, 11)]
    s = md.draw_samples(ids, 3, n_pairs=6, n_cross=5, n_sheet=3)
    cache = _stage_cache(tmp_path, s)
    with pytest.raises(CacheError):
        build_sheet(cache, s, FakeRunner(), REPO_ROOT / "contact_sheet_e1.png")
    assert not (REPO_ROOT / "contact_sheet_e1.png").exists()


def test_cli_makeup_sheet_and_fetch_refuse_repo_paths(tmp_path):
    ids = [f"{n:06d}" for n in range(1, 11)]
    s = md.draw_samples(ids, 3, n_pairs=6, n_cross=5, n_sheet=3)
    sample = tmp_path / "sample.txt"
    sample.write_text(md.format_samples(s, 3, 6))
    cache = tmp_path / "cache"
    with pytest.raises(CacheError):
        cli.cmd_makeup_sheet(argparse.Namespace(cache_dir=str(cache), sample=str(sample), runs_dir=str(REPO_ROOT / "runs_e1")))
    assert not (REPO_ROOT / "runs_e1").exists()
    with pytest.raises(CacheError):
        cli.cmd_makeup_fetch(
            argparse.Namespace(
                cache_dir=str(REPO_ROOT / "cache_e1"), seed=1, n=10, n_cross=5, budget_mb=1, workers=1, sample_out=None
            )
        )
