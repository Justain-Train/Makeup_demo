"""EasyPortrait v2 test split: cache, sampling, fetching and per-sample loading.

Privacy and safety rules implemented here:
  * the cache lives outside the repository (refused otherwise), directory mode 0700, files 0600;
  * a marker file identifies a cache this tool created; `clean` refuses to touch any other directory;
  * member names are built only from ids that match a strict UUID regex; nothing is ever extracted by name;
  * the meta table's `user_id` column is never stored, returned or logged;
  * no image-derived data reaches the logs (ids, byte counts and timings only).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.labels import EP_NUM_CLASSES
from evaluation.remote_zip import (
    BudgetedFetcher,
    FetchError,
    HttpsRangeFetcher,
    MemberInfo,
    RangeFetcher,
    fetch_member,
    read_central_directory,
)

logger = logging.getLogger("evaluation")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = Path("~/.cache/makeup_demo/easyportrait")
CACHE_ENV = "EVAL_CACHE_DIR"
MARKER_NAME = ".makeup_demo_easyportrait_cache"

BASE_URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/v2/"
URL_TEST = BASE_URL + "test.zip"
URL_ANNOTATIONS = BASE_URL + "annotations.zip"
URL_META = BASE_URL + "meta.zip"
# Content-Length values observed by HEAD on 2026-11-10 (README: test 12.8 GB, annotations 657.1 MB, meta 1.9 MB).
PINNED_LENGTHS = {URL_TEST: 12_794_142_394, URL_ANNOTATIONS: 657_142_345, URL_META: 1_941_735}
ETAGS_OBSERVED = {
    URL_TEST: "d4d15f2735bbc640a0a57f52fd53ceb2-2441",
    URL_ANNOTATIONS: "30f2daf7a76c98bf390d764ccb121272-126",
    URL_META: "270702dda4f7ae81414475f3a87fbbab",
}
IMAGE_MEMBER = "images/test/{id}.jpg"
ANNOTATION_MEMBER = "annotations/test/{id}.png"
META_MEMBER = "meta.tsv"  # comma-separated despite the extension

MAX_IMAGE_BYTES = 25 * 1024 * 1024  # largest observed test image: 9.6 MB
MAX_ANNOTATION_BYTES = 2 * 1024 * 1024  # largest observed annotation: 64 KB
MAX_META_BYTES = 16 * 1024 * 1024
DEFAULT_BUDGET_BYTES = 3 * 1024**3
DEFAULT_SEED = 20261110
DEFAULT_N = 600

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
BRIGHTNESS_QUARTILES = 4
RESOLUTION_EDGES_MP = (1.5, 2.5, 3.0)
RESOLUTION_GROUPS = ("lt1.5MP", "1.5-2.5MP", "2.5-3.0MP", "ge3.0MP")
DEV_FRACTION_MOD = 10
DEV_BELOW = 4  # sha256(id) mod 10 < 4 -> dev


class CacheError(Exception):
    pass


def is_valid_id(value: str) -> bool:
    return UUID_RE.fullmatch(value) is not None


def require_id(value: str) -> str:
    if not is_valid_id(value):
        raise ValueError("not a valid sample id")
    return value


# --- cache directory ---------------------------------------------------------------------------

def refuse_inside_repo(path: Path, what: str = "directory") -> Path:
    """Resolved absolute path, refused if it is (inside) the repository."""
    resolved = Path(os.path.expanduser(str(path))).resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents or resolved in REPO_ROOT.parents:
        raise CacheError(f"refusing to use a {what} inside or above the repository")
    return resolved


def resolve_cache_dir(arg: str | os.PathLike[str] | None = None) -> Path:
    raw = arg or os.environ.get(CACHE_ENV) or DEFAULT_CACHE_DIR
    return refuse_inside_repo(Path(raw), "cache directory")


def _chmod_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def ensure_cache(cache: Path) -> Path:
    """Create the cache (0700) with its marker, or verify an existing one. Refuses a non-empty foreign directory."""
    cache = refuse_inside_repo(cache, "cache directory")
    marker = cache / MARKER_NAME
    if cache.exists() and not marker.exists() and any(cache.iterdir()):
        raise CacheError("directory exists, is not empty and has no cache marker")
    _chmod_private_dir(cache)
    for sub in ("images", "annotations"):
        _chmod_private_dir(cache / sub)
    if not marker.exists():
        write_private(marker, b"makeup_demo easyportrait evaluation cache; safe to remove with `clean`\n")
    return cache


def write_private(path: Path, data: bytes) -> None:
    """Atomic write with mode 0600."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def clean_cache(cache: Path) -> int:
    """Delete a cache created by this tool. Returns the number of files removed."""
    cache = refuse_inside_repo(cache, "cache directory")
    if not (cache / MARKER_NAME).is_file():
        raise CacheError("no cache marker: refusing to delete")
    n = 0
    for sub in ("images", "annotations"):
        d = cache / sub
        if d.is_dir():
            n += sum(1 for p in d.iterdir() if p.is_file())
            shutil.rmtree(d)
    for name in ("manifest.json", "meta_test.csv", MARKER_NAME):
        p = cache / name
        if p.exists():
            p.unlink()
            n += 1
    try:
        cache.rmdir()
    except OSError:
        pass  # something else is in there: leave the directory, we only removed what we created
    return n


# --- meta table and sampling ---------------------------------------------------------------------

@dataclass(frozen=True)
class MetaRow:
    image_id: str
    height: int
    width: int
    brightness: float

    @property
    def megapixels(self) -> float:
        return self.height * self.width / 1e6


def parse_meta(text: str, split: str = "test") -> tuple[list[MetaRow], int]:
    """Rows of the given split, plus the number of rows dropped as malformed.

    Columns are read by NAME and `user_id` is deliberately never touched.
    """
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header is None:
        raise ValueError("empty meta table")
    need = {"image_name", "height", "width", "set", "brightness"}
    if not need <= set(header):
        raise ValueError("meta table lacks required columns")
    ci = {name: header.index(name) for name in need}
    rows, dropped = [], 0
    for rec in reader:
        try:
            if rec[ci["set"]] != split:
                continue
            image_id = rec[ci["image_name"]]
            h, w, b = float(rec[ci["height"]]), float(rec[ci["width"]]), float(rec[ci["brightness"]])
            if not is_valid_id(image_id) or not (0 < h < 100_000 and 0 < w < 100_000 and np.isfinite(b)):
                raise ValueError
            rows.append(MetaRow(image_id, int(round(h)), int(round(w)), b))
        except (ValueError, IndexError):
            dropped += 1
    return rows, dropped


def sanitized_meta_csv(rows: list[MetaRow]) -> str:
    out = io.StringIO()
    wr = csv.writer(out, lineterminator="\n")
    wr.writerow(["image_name", "height", "width", "brightness"])
    for r in sorted(rows, key=lambda r: r.image_id):
        wr.writerow([r.image_id, r.height, r.width, repr(r.brightness)])
    return out.getvalue()


def load_meta_csv(text: str) -> list[MetaRow]:
    reader = csv.DictReader(io.StringIO(text))
    return [MetaRow(r["image_name"], int(r["height"]), int(r["width"]), float(r["brightness"])) for r in reader]


def resolution_group(megapixels: float) -> int:
    return int(np.searchsorted(RESOLUTION_EDGES_MP, megapixels, side="right"))


def split_of(image_id: str) -> str:
    """Deterministic dev/test assignment: dev if sha256(id) mod 10 < 4."""
    return "dev" if int(hashlib.sha256(image_id.encode("ascii")).hexdigest(), 16) % DEV_FRACTION_MOD < DEV_BELOW else "test"


@dataclass(frozen=True)
class SampleEntry:
    image_id: str
    split: str
    brightness_quartile: int
    resolution_group: int


def allocate(cell_sizes: dict[tuple[int, int], int], n: int, floor: int = 5) -> dict[tuple[int, int], int]:
    """Proportional allocation with a per-cell floor of min(size, floor), summing to exactly n.

    The floors are given first; the rest is shared proportionally to each cell's spare capacity by the
    largest-remainder method, never exceeding a cell's size.
    """
    total = sum(cell_sizes.values())
    if n > total:
        raise ValueError("requested sample is larger than the population")
    base = {k: min(c, floor) for k, c in cell_sizes.items()}
    rest = n - sum(base.values())
    if rest < 0:
        raise ValueError("n is smaller than the sum of the floors")
    spare = {k: cell_sizes[k] - base[k] for k in cell_sizes}
    spare_total = sum(spare.values())
    alloc = dict(base)
    if rest and spare_total:
        exact = {k: rest * spare[k] / spare_total for k in spare}
        for k in spare:
            alloc[k] += min(int(exact[k]), spare[k])
        left = n - sum(alloc.values())
        for k in sorted(spare, key=lambda k: (-(exact[k] - int(exact[k])), k)):
            if left == 0:
                break
            if alloc[k] < cell_sizes[k]:
                alloc[k] += 1
                left -= 1
        while left > 0:  # only if rounding hit a cap: top up wherever there is room
            for k in sorted(cell_sizes):
                if left and alloc[k] < cell_sizes[k]:
                    alloc[k] += 1
                    left -= 1
    assert sum(alloc.values()) == n
    return alloc


def draw_sample(rows: list[MetaRow], n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> list[SampleEntry]:
    """Stratified (brightness quartile x resolution group) sample of exactly n ids, deterministic for a seed."""
    if not rows:
        raise ValueError("no rows to sample from")
    ordered = sorted(rows, key=lambda r: r.image_id)
    edges = np.quantile([r.brightness for r in ordered], [0.25, 0.5, 0.75])
    cells: dict[tuple[int, int], list[MetaRow]] = {}
    for r in ordered:
        q = int(np.searchsorted(edges, r.brightness, side="right"))
        cells.setdefault((q, resolution_group(r.megapixels)), []).append(r)
    alloc = allocate({k: len(v) for k, v in cells.items()}, n)
    rng = np.random.default_rng(seed)
    chosen: list[SampleEntry] = []
    for key in sorted(cells):
        members = cells[key]
        picks = rng.choice(len(members), size=alloc[key], replace=False)
        chosen.extend(SampleEntry(members[i].image_id, split_of(members[i].image_id), key[0], key[1]) for i in sorted(picks))
    return sorted(chosen, key=lambda e: e.image_id)


def format_sample(entries: list[SampleEntry], seed: int, n: int) -> str:
    lines = [
        "# EasyPortrait v2 test split, stratified sample (brightness quartile x resolution group).",
        f"# seed={seed} n={n}. Opaque dataset ids only; no images, annotations or per-image results.",
        "# Source: https://github.com/hukenovs/easyportrait (Kapitanov, Kvanchiani, Kirillova, arXiv:2304.13509).",
        "# split: dev if sha256(id) mod 10 < 4 else test (tune on dev only).",
        "# columns: id<TAB>split<TAB>brightness_quartile(0-3)<TAB>resolution_group(" + ",".join(RESOLUTION_GROUPS) + " = 0-3)",
    ]
    lines += [f"{e.image_id}\t{e.split}\t{e.brightness_quartile}\t{e.resolution_group}" for e in entries]
    return "\n".join(lines) + "\n"


def parse_sample(text: str) -> list[SampleEntry]:
    out: list[SampleEntry] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            raise ValueError("malformed sample line")
        image_id, split, q, g = parts
        require_id(image_id)
        if split not in ("dev", "test") or image_id in seen:
            raise ValueError("bad split or duplicate id in sample file")
        if split != split_of(image_id):
            raise ValueError("split does not match sha256 rule")
        seen.add(image_id)
        out.append(SampleEntry(image_id, split, int(q), int(g)))
    return out


# --- fetching ------------------------------------------------------------------------------------

def fetch_meta_rows(fetcher: BudgetedFetcher) -> tuple[list[MetaRow], int, str]:
    """Download meta.zip (1.9 MB), return (test rows, malformed-row count, sha256 of the zip)."""
    total = fetcher.verified_length(URL_META)
    blob = fetcher.get_range(URL_META, 0, total - 1)
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            info = zf.getinfo(META_MEMBER)
            if info.file_size > MAX_META_BYTES:
                raise FetchError("meta table exceeds the size cap")
            text = zf.read(info).decode("utf-8")  # zipfile verifies the CRC
    except (KeyError, zipfile.BadZipFile, UnicodeDecodeError):
        raise FetchError("meta.zip is not the expected archive") from None
    rows, dropped = parse_meta(text)
    return rows, dropped, hashlib.sha256(blob).hexdigest()


def load_or_fetch_meta(cache: Path, fetcher: RangeFetcher | None = None) -> list[MetaRow]:
    path = cache / "meta_test.csv"
    if path.is_file():
        return load_meta_csv(path.read_text())
    budgeted = BudgetedFetcher(fetcher or HttpsRangeFetcher(), 16 * 1024 * 1024, PINNED_LENGTHS)
    rows, dropped, digest = fetch_meta_rows(budgeted)
    write_private(path, sanitized_meta_csv(rows).encode())
    logger.info("meta: %d test rows (%d malformed dropped), meta.zip sha256 %s", len(rows), dropped, digest)
    return rows


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_path(cache: Path, image_id: str) -> Path:
    return cache / "images" / f"{require_id(image_id)}.jpg"


def annotation_path(cache: Path, image_id: str) -> Path:
    return cache / "annotations" / f"{require_id(image_id)}.png"


def _load_manifest(cache: Path) -> dict[str, dict[str, object]]:
    p = cache / "manifest.json"
    return json.loads(p.read_text()).get("files", {}) if p.is_file() else {}


def _save_manifest(cache: Path, files: dict[str, dict[str, object]]) -> None:
    write_private(cache / "manifest.json", json.dumps({"pinned_lengths": PINNED_LENGTHS, "files": files}, indent=1, sort_keys=True).encode())


@dataclass
class FetchSummary:
    n_requested: int
    n_downloaded: int
    n_skipped: int
    n_failed: int
    bytes_received: int
    seconds: float
    throughput_mb_s: float
    requests: int


def fetch_samples(
    entries: list[SampleEntry],
    cache: Path,
    fetcher: RangeFetcher | None = None,
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
    threads: int = 4,
    progress_every: int = 50,
) -> FetchSummary:
    """Download exactly the sampled images and annotations. Resumable: verified files are skipped."""
    cache = ensure_cache(cache)
    for e in entries:
        require_id(e.image_id)
    bf = BudgetedFetcher(fetcher or HttpsRangeFetcher(), budget_bytes, PINNED_LENGTHS)
    t0 = time.perf_counter()
    manifest = _load_manifest(cache)

    def have(image_id: str) -> bool:
        m = manifest.get(image_id)
        ip, ap = image_path(cache, image_id), annotation_path(cache, image_id)
        return bool(m) and ip.is_file() and ap.is_file() and _sha256(ip) == m["image_sha256"] and _sha256(ap) == m["annotation_sha256"]

    todo = [e.image_id for e in entries if not have(e.image_id)]
    skipped = len(entries) - len(todo)
    downloaded = failed = 0
    if todo:
        img_table = read_central_directory(bf, URL_TEST)
        ann_table = read_central_directory(bf, URL_ANNOTATIONS)
        logger.info("central directories read: %d image members, %d annotation members, %.1f MB", len(img_table), len(ann_table), bf.bytes_received / 1e6)

        def one(image_id: str) -> tuple[str, dict[str, object] | None]:
            ii: MemberInfo | None = img_table.get(IMAGE_MEMBER.format(id=image_id))
            ai: MemberInfo | None = ann_table.get(ANNOTATION_MEMBER.format(id=image_id))
            if ii is None or ai is None:
                return image_id, None
            img = fetch_member(bf, URL_TEST, ii, MAX_IMAGE_BYTES)
            ann = fetch_member(bf, URL_ANNOTATIONS, ai, MAX_ANNOTATION_BYTES)
            write_private(image_path(cache, image_id), img)
            write_private(annotation_path(cache, image_id), ann)
            return image_id, {
                "image_sha256": hashlib.sha256(img).hexdigest(),
                "annotation_sha256": hashlib.sha256(ann).hexdigest(),
                "image_bytes": len(img),
                "annotation_bytes": len(ann),
            }

        with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
            futures = [pool.submit(one, i) for i in todo]
            for k, fut in enumerate(futures, 1):
                try:
                    image_id, meta = fut.result()
                except FetchError as exc:
                    for f in futures:
                        f.cancel()
                    _save_manifest(cache, manifest)
                    raise FetchError(f"fetch aborted after {downloaded} samples: {exc}") from None
                if meta is None:
                    failed += 1
                    logger.warning("sample %s: member not found in the archives", image_id)
                else:
                    manifest[image_id] = meta
                    downloaded += 1
                if k % progress_every == 0 or k == len(futures):
                    _save_manifest(cache, manifest)
                    logger.info(
                        "fetched %d/%d  %.1f MB  %.2f MB/s", k, len(futures), bf.bytes_received / 1e6, bf.throughput_mb_s
                    )
        _save_manifest(cache, manifest)
    secs = time.perf_counter() - t0
    return FetchSummary(len(entries), downloaded, skipped, failed, bf.bytes_received, secs, bf.throughput_mb_s, bf.requests)


# --- loading one sample --------------------------------------------------------------------------

class InvalidSample(Exception):
    """The annotation or image of a sample is unusable. `reason` is a fixed category, never image data."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def decode_annotation(data: bytes, expected_hw: tuple[int, int]) -> np.ndarray:
    """PNG, mode L or P, values <= 8, shape equal to the decoded image. Never rotated or resized to fit."""
    try:
        with Image.open(io.BytesIO(data), formats=["PNG"]) as im:
            if im.mode not in ("L", "P"):
                raise InvalidSample("annotation_mode")
            if (im.height, im.width) != tuple(expected_hw):
                raise InvalidSample("annotation_shape")
            im.load()
            arr = np.asarray(im, dtype=np.uint8)
    except InvalidSample:
        raise
    except Exception as exc:  # noqa: BLE001 - Pillow raises many types on corrupt data
        raise InvalidSample("annotation_undecodable") from exc
    if arr.size == 0 or int(arr.max()) >= EP_NUM_CLASSES:
        raise InvalidSample("annotation_values")
    return arr
