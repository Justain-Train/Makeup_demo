"""Evaluation data handling: every fetch guard, cache safety, sampling. Offline: fake fetcher, in-memory zips."""

import io
import json
import os
import stat
import urllib.error
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from evaluation import dataset as ds
from evaluation import remote_zip as rz

IDS = [f"{i:08x}-0000-4000-8000-{i:012x}" for i in range(1, 9)]
URL_A, URL_B = "https://example.invalid/a.zip", "https://example.invalid/b.zip"


def make_zip(members: dict[str, bytes], compression=zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


class FakeFetcher:
    """Serves in-memory archives like a range-capable server; records requests."""

    def __init__(self, archives: dict[str, bytes], honour_range: bool = True):
        self.archives, self.honour_range, self.calls = archives, honour_range, []

    def head(self, url):
        return rz.Head(len(self.archives[url]), None)

    def get_range(self, url, start, end):
        self.calls.append((url, start, end))
        if not self.honour_range:
            raise rz.FetchError("server did not honour the range (HTTP 200)")
        data = self.archives[url]
        return data[start : end + 1]


def pins(fetcher: FakeFetcher) -> dict[str, int]:
    return {u: len(b) for u, b in fetcher.archives.items()}


# --- remote zip guards -----------------------------------------------------------------------------------

def test_central_directory_and_member_roundtrip():
    payload = os.urandom(5000) + b"A" * 20000
    f = FakeFetcher({URL_A: make_zip({"images/x.jpg": payload, "other.txt": b"hi"})})
    bf = rz.BudgetedFetcher(f, 10**7, pins(f))
    table = rz.read_central_directory(bf, URL_A)
    assert set(table) == {"images/x.jpg", "other.txt"}
    assert rz.fetch_member(bf, URL_A, table["images/x.jpg"], 10**6) == payload
    assert rz.fetch_member(bf, URL_A, table["other.txt"], 10**6) == b"hi"


def test_stored_members_work_too():
    f = FakeFetcher({URL_A: make_zip({"a.bin": b"stored" * 100}, zipfile.ZIP_STORED)})
    bf = rz.BudgetedFetcher(f, 10**7, pins(f))
    info = rz.read_central_directory(bf, URL_A)["a.bin"]
    assert rz.fetch_member(bf, URL_A, info, 10**6) == b"stored" * 100


def test_only_requested_members_are_downloaded():
    members = {f"m{i}.bin": os.urandom(50_000) for i in range(10)}
    f = FakeFetcher({URL_A: make_zip(members)})
    bf = rz.BudgetedFetcher(f, 10**8, pins(f))
    info = rz.read_central_directory(bf, URL_A)["m3.bin"]
    before = bf.bytes_received
    rz.fetch_member(bf, URL_A, info, 10**6)
    assert bf.bytes_received - before < 50_500 + 400  # one member plus header slack, not the archive


def test_crc_mismatch_is_detected():
    data = bytearray(make_zip({"a.bin": b"hello world" * 50}, zipfile.ZIP_STORED))
    pos = data.index(b"hello world")
    data[pos] ^= 0xFF  # corrupt the payload, leave the CRC
    f = FakeFetcher({URL_A: bytes(data)})
    bf = rz.BudgetedFetcher(f, 10**7, pins(f))
    info = rz.read_central_directory(bf, URL_A)["a.bin"]
    with pytest.raises(rz.FetchError, match="CRC"):
        rz.fetch_member(bf, URL_A, info, 10**6)


def test_member_size_cap_before_and_while_decompressing():
    f = FakeFetcher({URL_A: make_zip({"big.bin": b"\0" * 1_000_000})})  # compresses to ~1 KB
    bf = rz.BudgetedFetcher(f, 10**7, pins(f))
    info = rz.read_central_directory(bf, URL_A)["big.bin"]
    with pytest.raises(rz.FetchError, match="size cap"):
        rz.fetch_member(bf, URL_A, info, 10_000)  # declared size over the cap: refused before downloading
    assert bf.bytes_received < 5000  # nothing but the central directory was downloaded
    # a lying central directory (declared size small, real size big) is caught while decompressing
    lying = rz.MemberInfo(info.name, info.header_offset, info.compress_size, 100, info.crc, info.method)
    with pytest.raises(rz.FetchError):
        rz.fetch_member(bf, URL_A, lying, 10_000)


def test_local_header_name_mismatch_is_refused():
    f = FakeFetcher({URL_A: make_zip({"a.bin": b"x" * 100, "b.bin": b"y" * 100})})
    bf = rz.BudgetedFetcher(f, 10**7, pins(f))
    table = rz.read_central_directory(bf, URL_A)
    forged = rz.MemberInfo("a.bin", table["b.bin"].header_offset, table["b.bin"].compress_size, table["b.bin"].file_size, table["b.bin"].crc, table["b.bin"].method)
    with pytest.raises(rz.FetchError, match="name"):
        rz.fetch_member(bf, URL_A, forged, 10**6)


def test_encrypted_and_unsupported_members_are_refused():
    raw = bytearray(make_zip({"a.bin": b"x" * 10}))
    # set the encryption flag bit in the central directory entry
    cd = raw.index(b"PK\x01\x02")
    raw[cd + 8] |= 1
    f = FakeFetcher({URL_A: bytes(raw)})
    with pytest.raises(rz.FetchError, match="encrypted"):
        rz.read_central_directory(rz.BudgetedFetcher(f, 10**7, pins(f)), URL_A)
    info = rz.MemberInfo("a.bin", 0, 10, 10, 0, zipfile.ZIP_BZIP2)
    f2 = FakeFetcher({URL_A: make_zip({"a.bin": b"x"})})
    with pytest.raises(rz.FetchError, match="compression"):
        rz.fetch_member(rz.BudgetedFetcher(f2, 10**7, pins(f2)), URL_A, info, 10**6)


def test_byte_budget_is_enforced_and_counted():
    f = FakeFetcher({URL_A: make_zip({"a.bin": os.urandom(20000)})})
    bf = rz.BudgetedFetcher(f, 500, pins(f))
    with pytest.raises(rz.FetchError, match="budget"):
        rz.read_central_directory(bf, URL_A)
    assert bf.bytes_received <= 500


def test_pinned_length_mismatch_and_unpinned_urls():
    f = FakeFetcher({URL_A: make_zip({"a.bin": b"x"})})
    bf = rz.BudgetedFetcher(f, 10**7, {URL_A: len(f.archives[URL_A]) + 1})
    with pytest.raises(rz.FetchError, match="Content-Length changed"):
        rz.read_central_directory(bf, URL_A)
    with pytest.raises(rz.FetchError, match="not pinned"):
        rz.BudgetedFetcher(f, 10**7, {}).verified_length(URL_A)
    with pytest.raises(rz.FetchError, match="not pinned"):
        rz.BudgetedFetcher(f, 10**7, {}).get_range(URL_A, 0, 1)
    with pytest.raises(rz.FetchError, match="beyond"):
        rz.BudgetedFetcher(f, 10**7, pins(f)).get_range(URL_A, 0, len(f.archives[URL_A]))


def test_not_a_zip_is_refused():
    f = FakeFetcher({URL_A: b"this is not a zip file" * 100})
    with pytest.raises(rz.FetchError):
        rz.read_central_directory(rz.BudgetedFetcher(f, 10**7, pins(f)), URL_A)


# --- the real HTTP fetcher with a fake opener (206 / Content-Range / https / redirects) -------------------------

class _Resp:
    def __init__(self, status, headers, body=b""):
        self.status, self.headers, self._body = status, headers, body

    def read(self, n=-1):
        return self._body if n < 0 else self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fetcher_with(resp):
    fx = rz.HttpsRangeFetcher()
    fx._opener = SimpleNamespace(open=lambda req, timeout=None: resp)
    return fx


def test_http_fetcher_accepts_only_a_matching_206():
    ok = _Resp(206, {"Content-Range": "bytes 10-14/100"}, b"12345")
    assert _fetcher_with(ok).get_range("https://h/x.zip", 10, 14) == b"12345"
    with pytest.raises(rz.FetchError, match="did not honour"):
        _fetcher_with(_Resp(200, {}, b"x" * 100)).get_range("https://h/x.zip", 10, 14)  # server ignored Range
    with pytest.raises(rz.FetchError, match="Content-Range"):
        _fetcher_with(_Resp(206, {"Content-Range": "bytes 0-4/100"}, b"12345")).get_range("https://h/x.zip", 10, 14)
    with pytest.raises(rz.FetchError, match="Content-Range"):
        _fetcher_with(_Resp(206, {}, b"12345")).get_range("https://h/x.zip", 10, 14)
    with pytest.raises(rz.FetchError, match="short or long"):
        _fetcher_with(_Resp(206, {"Content-Range": "bytes 10-14/100"}, b"1234567")).get_range("https://h/x.zip", 10, 14)


def test_http_fetcher_refuses_plain_http_and_redirects():
    fx = rz.HttpsRangeFetcher()
    with pytest.raises(rz.FetchError, match="https"):
        fx.get_range("http://h/x.zip", 0, 1)
    with pytest.raises(rz.FetchError, match="https"):
        fx.head("ftp://h/x.zip")
    handler = rz._NoRedirect()
    with pytest.raises(rz.FetchError, match="redirect"):
        handler.redirect_request(None, None, 302, "Found", {}, "https://elsewhere/x")
    assert any(isinstance(h, rz._NoRedirect) for h in fx._opener.handlers)


def test_http_errors_become_fetch_errors_without_details():
    def boom(req, timeout=None):
        raise urllib.error.URLError("secret detail")

    fx = rz.HttpsRangeFetcher()
    fx._opener = SimpleNamespace(open=boom)
    with pytest.raises(rz.FetchError) as e:
        fx.get_range("https://h/x", 0, 1)
    assert "secret detail" not in str(e.value)


# --- ids, paths, cache ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "../../etc/passwd", IDS[0] + "/../x", "ABCDEF01-0000-4000-8000-00000000000A", IDS[0] + ".jpg", IDS[0][:-1], "a" * 36, IDS[0] + "\n"])
def test_uuid_guard_rejects_everything_else(bad, tmp_path):
    assert not ds.is_valid_id(bad)
    with pytest.raises(ValueError):
        ds.image_path(tmp_path, bad)
    with pytest.raises(ValueError):
        ds.annotation_path(tmp_path, bad)


def test_member_names_come_only_from_valid_ids(tmp_path):
    assert ds.image_path(tmp_path, IDS[0]).parent == tmp_path / "images"
    assert ds.IMAGE_MEMBER.format(id=IDS[0]) == f"images/test/{IDS[0]}.jpg"
    assert ds.ANNOTATION_MEMBER.format(id=IDS[0]) == f"annotations/test/{IDS[0]}.png"


def test_cache_inside_the_repo_is_refused():
    for p in (ds.REPO_ROOT, ds.REPO_ROOT / "docs" / "cache", ds.REPO_ROOT / "backend" / "evaluation" / "c", ds.REPO_ROOT.parent):
        with pytest.raises(ds.CacheError):
            ds.refuse_inside_repo(p)
    with pytest.raises(ds.CacheError):
        ds.resolve_cache_dir(str(ds.REPO_ROOT / "x"))


def test_symlink_into_the_repo_is_refused(tmp_path):
    link = tmp_path / "link"
    link.symlink_to(ds.REPO_ROOT / "docs")
    with pytest.raises(ds.CacheError):
        ds.refuse_inside_repo(link / "cache")


def test_env_var_and_default_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(ds.CACHE_ENV, str(tmp_path / "envcache"))
    assert ds.resolve_cache_dir(None) == (tmp_path / "envcache").resolve()
    monkeypatch.delenv(ds.CACHE_ENV)
    assert ds.resolve_cache_dir(None) == Path(os.path.expanduser("~/.cache/makeup_demo/easyportrait")).resolve()


def test_cache_permissions_marker_and_clean(tmp_path):
    cache = ds.ensure_cache(tmp_path / "c")
    assert stat.S_IMODE(cache.stat().st_mode) == 0o700
    assert stat.S_IMODE((cache / "images").stat().st_mode) == 0o700
    marker = cache / ds.MARKER_NAME
    assert marker.is_file() and stat.S_IMODE(marker.stat().st_mode) == 0o600
    ds.write_private(cache / "images" / "f.jpg", b"x")
    assert stat.S_IMODE((cache / "images" / "f.jpg").stat().st_mode) == 0o600
    assert ds.clean_cache(cache) >= 2 and not cache.exists()


def test_clean_refuses_directories_without_the_marker(tmp_path):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "precious.txt").write_text("keep")
    with pytest.raises(ds.CacheError):
        ds.clean_cache(foreign)
    with pytest.raises(ds.CacheError):
        ds.ensure_cache(foreign)  # non-empty and unmarked: not adopted
    assert (foreign / "precious.txt").read_text() == "keep"


# --- meta, sampling -----------------------------------------------------------------------------------------------

def _meta_text(n=400, with_bad=True):
    rng = np.random.default_rng(3)
    lines = ["image_name,user_id,height,width,set,brightness"]
    for i in range(n):
        iid = f"{i + 1000:08x}-1111-4222-8333-{i + 1000:012x}"
        h, w = rng.choice([(1920.0, 1440.0), (1440.0, 1920.0), (1920.0, 1080.0), (800.0, 600.0)])
        lines.append(f"{iid},SECRET-USER-{i},{h},{w},{'test' if i % 3 else 'train'},{rng.uniform(20, 230):.4f}")
    if with_bad:
        lines.append("not-a-uuid,SECRET-USER-x,1920.0,1440.0,test,100.0")
        lines.append(f"{IDS[0]},SECRET-USER-y,abc,1440.0,test,100.0")
    return "\n".join(lines)


def test_meta_parsing_never_keeps_user_id_and_drops_bad_rows():
    rows, dropped = ds.parse_meta(_meta_text())
    assert dropped == 2 and all(ds.is_valid_id(r.image_id) for r in rows)
    out = ds.sanitized_meta_csv(rows)
    assert "SECRET" not in out and "user_id" not in out
    assert ds.load_meta_csv(out) == sorted(rows, key=lambda r: r.image_id)
    with pytest.raises(ValueError):
        ds.parse_meta("a,b\n1,2")


def test_allocation_is_exact_with_floor_and_capacity():
    sizes = {(0, 0): 1000, (0, 1): 3, (1, 0): 500, (1, 1): 0 + 1, (2, 2): 200}
    a = ds.allocate(sizes, 100)
    assert sum(a.values()) == 100 and all(a[k] <= sizes[k] for k in sizes)
    assert a[(0, 1)] == 3 and a[(1, 1)] == 1  # tiny cells fully kept (floor)
    assert a[(0, 0)] > a[(1, 0)] > a[(2, 2)]  # roughly proportional
    with pytest.raises(ValueError):
        ds.allocate({(0, 0): 5}, 6)


def test_sample_is_deterministic_exact_and_covers_every_stratum():
    rows, _ = ds.parse_meta(_meta_text(with_bad=False))
    a = ds.draw_sample(rows, 60, seed=20261110)
    b = ds.draw_sample(list(reversed(rows)), 60, seed=20261110)
    c = ds.draw_sample(rows, 60, seed=1)
    assert a == b and a != c and len(a) == 60 == len({e.image_id for e in a})
    edges = np.quantile([r.brightness for r in sorted(rows, key=lambda r: r.image_id)], [0.25, 0.5, 0.75])
    populated = {(int(np.searchsorted(edges, r.brightness, side="right")), ds.resolution_group(r.megapixels)) for r in rows}
    assert populated == {(e.brightness_quartile, e.resolution_group) for e in a}


def test_dev_test_split_rule_and_sample_file_roundtrip():
    for i in IDS:
        import hashlib

        expect = "dev" if int(hashlib.sha256(i.encode()).hexdigest(), 16) % 10 < 4 else "test"
        assert ds.split_of(i) == expect
    rows, _ = ds.parse_meta(_meta_text(with_bad=False))
    entries = ds.draw_sample(rows, 100)
    text = ds.format_sample(entries, ds.DEFAULT_SEED, 100)
    assert ds.parse_sample(text) == entries
    assert "SECRET" not in text
    # tampered split / duplicate / bad id
    first = entries[0]
    wrong = "test" if first.split == "dev" else "dev"
    with pytest.raises(ValueError):
        ds.parse_sample(f"{first.image_id}\t{wrong}\t0\t0\n")
    with pytest.raises(ValueError):
        ds.parse_sample(text + f"{first.image_id}\t{first.split}\t0\t0\n")
    with pytest.raises(ValueError):
        ds.parse_sample("../../x\tdev\t0\t0\n")


# --- fetch_samples end to end with a fake server -----------------------------------------------------------------

def _png(w=8, h=6, value=0):
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w), value, dtype=np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _fake_dataset(ids):
    imgs = {f"images/test/{i}.jpg": os.urandom(3000) for i in ids}
    anns = {f"annotations/test/{i}.png": _png() for i in ids}
    return FakeFetcher({ds.URL_TEST: make_zip(imgs, zipfile.ZIP_STORED), ds.URL_ANNOTATIONS: make_zip(anns)})


def test_fetch_is_resumable_verified_and_downloads_only_the_sample(tmp_path, monkeypatch):
    fake = _fake_dataset(IDS)
    monkeypatch.setitem(ds.PINNED_LENGTHS, ds.URL_TEST, len(fake.archives[ds.URL_TEST]))
    monkeypatch.setitem(ds.PINNED_LENGTHS, ds.URL_ANNOTATIONS, len(fake.archives[ds.URL_ANNOTATIONS]))
    entries = [ds.SampleEntry(i, ds.split_of(i), 0, 0) for i in IDS[:3]]
    s = ds.fetch_samples(entries, tmp_path / "c", fake, budget_bytes=10**7, threads=2)
    assert (s.n_downloaded, s.n_skipped, s.n_failed) == (3, 0, 0) and s.bytes_received > 0
    cache = tmp_path / "c"
    assert sorted(p.name for p in (cache / "images").iterdir()) == sorted(f"{e.image_id}.jpg" for e in entries)
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in (cache / "images").iterdir())
    manifest = json.loads((cache / "manifest.json").read_text())
    assert set(manifest["files"]) == {e.image_id for e in entries}
    # second run: nothing to do, nothing downloaded
    s2 = ds.fetch_samples(entries, cache, fake, budget_bytes=10**7)
    assert (s2.n_downloaded, s2.n_skipped, s2.bytes_received) == (0, 3, 0)
    # a corrupted cached file is detected by its SHA-256 and re-fetched
    (cache / "images" / f"{entries[0].image_id}.jpg").write_bytes(b"corrupt")
    s3 = ds.fetch_samples(entries, cache, fake, budget_bytes=10**7)
    assert (s3.n_downloaded, s3.n_skipped) == (1, 2)


def test_fetch_aborts_when_the_server_ignores_range_or_budget_is_exceeded(tmp_path, monkeypatch):
    fake = _fake_dataset(IDS)
    monkeypatch.setitem(ds.PINNED_LENGTHS, ds.URL_TEST, len(fake.archives[ds.URL_TEST]))
    monkeypatch.setitem(ds.PINNED_LENGTHS, ds.URL_ANNOTATIONS, len(fake.archives[ds.URL_ANNOTATIONS]))
    entries = [ds.SampleEntry(i, ds.split_of(i), 0, 0) for i in IDS[:2]]
    with pytest.raises(rz.FetchError):
        ds.fetch_samples(entries, tmp_path / "a", FakeFetcher(fake.archives, honour_range=False), budget_bytes=10**7)
    with pytest.raises(rz.FetchError, match="budget"):
        ds.fetch_samples(entries, tmp_path / "b", fake, budget_bytes=2000)


def test_fetch_reports_missing_members_without_aborting(tmp_path, monkeypatch):
    fake = _fake_dataset(IDS[:1])
    monkeypatch.setitem(ds.PINNED_LENGTHS, ds.URL_TEST, len(fake.archives[ds.URL_TEST]))
    monkeypatch.setitem(ds.PINNED_LENGTHS, ds.URL_ANNOTATIONS, len(fake.archives[ds.URL_ANNOTATIONS]))
    entries = [ds.SampleEntry(i, ds.split_of(i), 0, 0) for i in IDS[:2]]
    s = ds.fetch_samples(entries, tmp_path / "c", fake, budget_bytes=10**7)
    assert (s.n_downloaded, s.n_failed) == (1, 1)


def test_fetch_refuses_a_cache_inside_the_repo():
    with pytest.raises(ds.CacheError):
        ds.fetch_samples([], ds.REPO_ROOT / "docs" / "evaluation" / "cache", FakeFetcher({}))


# --- annotation decoding ---------------------------------------------------------------------------------------------

def test_annotation_decoder_enforces_png_mode_shape_and_values():
    good = _png(8, 6, 3)
    assert ds.decode_annotation(good, (6, 8)).shape == (6, 8)
    with pytest.raises(ds.InvalidSample) as e:
        ds.decode_annotation(good, (8, 6))  # transposed: never rotated to fit
    assert e.value.reason == "annotation_shape"
    with pytest.raises(ds.InvalidSample) as e:
        ds.decode_annotation(_png(8, 6, 9), (6, 8))
    assert e.value.reason == "annotation_values"
    buf = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buf, format="PNG")
    with pytest.raises(ds.InvalidSample) as e:
        ds.decode_annotation(buf.getvalue(), (6, 8))
    assert e.value.reason == "annotation_mode"
    buf = io.BytesIO()
    Image.new("L", (8, 6)).save(buf, format="JPEG")
    with pytest.raises(ds.InvalidSample) as e:
        ds.decode_annotation(buf.getvalue(), (6, 8))
    assert e.value.reason == "annotation_undecodable"  # a JPEG is not accepted as an annotation
    with pytest.raises(ds.InvalidSample):
        ds.decode_annotation(b"garbage", (6, 8))


def test_cli_refuses_in_repo_paths_and_clean_needs_yes(tmp_path):
    from evaluation import cli

    assert cli.main(["clean", "--cache-dir", str(ds.REPO_ROOT / "docs"), "--yes"]) == 1
    assert cli.main(["fetch", "--cache-dir", str(ds.REPO_ROOT / "docs" / "c"), "--sample", str(tmp_path / "none.txt")]) == 1
    cache = ds.ensure_cache(tmp_path / "c")
    assert cli.main(["clean", "--cache-dir", str(cache)]) == 0 and cache.exists()  # dry run without --yes
    assert cli.main(["clean", "--cache-dir", str(cache), "--yes"]) == 0 and not cache.exists()
