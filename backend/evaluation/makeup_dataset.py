"""FFHQ-Makeup (CC BY-NC-SA 4.0): index, deterministic sampling, budgeted fetching and loading.

Same rules as dataset.py: cache outside the repo (0700/0600), marker file, strict member-name regex, nothing
extracted by member name, no image data or URLs in logs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services.image_loading import load_rgb_image
from evaluation.dataset import CacheError, refuse_inside_repo, write_private
from evaluation.hf_resolve import ResolvingFetcher
from evaluation.remote_zip import (
    BudgetedFetcher,
    FetchError,
    MemberInfo,
    RangeFetcher,
    fetch_member,
    read_central_directory,
)

logger = logging.getLogger("evaluation")

URL = "https://huggingface.co/datasets/cyberagent/FFHQ-Makeup/resolve/main/FFHQ-Makeup.zip"
PINNED_LENGTHS = {URL: 3_811_722_185}
DEFAULT_CACHE_DIR = Path("~/.cache/makeup_demo/ffhq_makeup")
MARKER_NAME = ".makeup_demo_ffhq_makeup_cache"
CACHE_ENV = "EVAL_MAKEUP_CACHE_DIR"
MEMBER_RE = re.compile(r"^(\d{6})/(bare|makeup_0[1-5])\.jpg$")
ID_RE = re.compile(r"\d{6}")
KINDS = ("bare", "makeup_01", "makeup_02", "makeup_03", "makeup_04", "makeup_05")
MAX_MEMBER_BYTES = 2 * 1024 * 1024
IMAGE_SIZE = 512
DEFAULT_SEED = 20261113
BUDGET_MSG = "byte budget exceeded"


def require_id(value: str) -> str:
    if not ID_RE.fullmatch(value):
        raise ValueError("not a valid identity id")
    return value


def require_kind(value: str) -> str:
    if value not in KINDS:
        raise ValueError("not a valid image kind")
    return value


def makeup_kind(k: int) -> str:
    if not 1 <= k <= 5:
        raise ValueError("makeup index out of range")
    return f"makeup_0{k}"


# --- cache ---------------------------------------------------------------------------------------

def resolve_cache_dir(arg: str | os.PathLike[str] | None = None) -> Path:
    raw = arg or os.environ.get(CACHE_ENV) or DEFAULT_CACHE_DIR
    return refuse_inside_repo(Path(raw), "cache directory")


def ensure_cache(cache: Path) -> Path:
    cache = refuse_inside_repo(cache, "cache directory")
    marker = cache / MARKER_NAME
    if cache.exists() and not marker.exists() and any(cache.iterdir()):
        raise CacheError("directory exists, is not empty and has no cache marker")
    for d in (cache, cache / "images"):
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
    if not marker.exists():
        write_private(marker, b"makeup_demo FFHQ-Makeup evaluation cache (CC BY-NC-SA 4.0)\n")
    return cache


def image_path(cache: Path, image_id: str, kind: str) -> Path:
    return cache / "images" / f"{require_id(image_id)}_{require_kind(kind)}.jpg"


# --- index ---------------------------------------------------------------------------------------

def complete_ids(names: list[str] | set[str]) -> list[str]:
    """Sorted ids that have all six files, judged from member names that match the strict regex."""
    seen: dict[str, set[str]] = {}
    for name in names:
        m = MEMBER_RE.fullmatch(name)
        if m:
            seen.setdefault(m.group(1), set()).add(m.group(2))
    full = set(KINDS)
    return sorted(i for i, kinds in seen.items() if kinds == full)


def load_or_build_index(cache: Path, bf: BudgetedFetcher) -> tuple[list[str], dict[str, MemberInfo] | None]:
    """(ids, central directory if it was read now). The cached index holds ids only."""
    path = cache / "index.json"
    if path.is_file():
        ids = json.loads(path.read_text())["ids"]
        return [require_id(i) for i in ids], None
    table = read_central_directory(bf, URL)
    ids = complete_ids(list(table))
    write_private(path, json.dumps({"ids": ids}).encode())
    logger.info("indexed %d complete identities (%d members)", len(ids), len(table))
    return ids, table


# --- sampling ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MakeupSamples:
    pairs: list[str]
    cross: list[tuple[str, str, int]]
    sheet: list[str]


def draw_samples(ids: list[str], seed: int, n_pairs: int = 200, n_cross: int = 60, n_sheet: int = 24) -> MakeupSamples:
    pool = sorted(set(ids))
    if n_pairs > len(pool) or n_sheet > n_pairs or len(pool) < 2:
        raise ValueError("sample sizes exceed the population")
    rng = np.random.default_rng(seed)
    pairs = [pool[i] for i in sorted(rng.choice(len(pool), size=n_pairs, replace=False))]
    cross: list[tuple[str, str, int]] = []
    for _ in range(n_cross):
        i = int(rng.integers(len(pool)))
        j = int(rng.integers(len(pool) - 1))
        if j >= i:
            j += 1
        cross.append((pool[i], pool[j], int(rng.integers(1, 6))))
    sheet = [pairs[i] for i in sorted(rng.choice(n_pairs, size=n_sheet, replace=False))]
    return MakeupSamples(pairs, cross, sheet)


def format_samples(s: MakeupSamples, seed: int, n: int) -> str:
    lines = [
        "# FFHQ-Makeup (cyberagent, CC BY-NC-SA 4.0) sample. Opaque identity ids only; no images or results.",
        f"# seed={seed} n={n}. Rows: pair<TAB>id | cross<TAB>bare_id<TAB>makeup_id<TAB>k | sheet<TAB>id",
    ]
    lines += [f"pair\t{i}" for i in s.pairs]
    lines += [f"cross\t{a}\t{b}\t{k}" for a, b, k in s.cross]
    lines += [f"sheet\t{i}" for i in s.sheet]
    return "\n".join(lines) + "\n"


def parse_samples(text: str) -> MakeupSamples:
    pairs: list[str] = []
    cross: list[tuple[str, str, int]] = []
    sheet: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split("\t")
        if p[0] == "pair" and len(p) == 2:
            pairs.append(require_id(p[1]))
        elif p[0] == "sheet" and len(p) == 2:
            sheet.append(require_id(p[1]))
        elif p[0] == "cross" and len(p) == 4:
            a, b, k = require_id(p[1]), require_id(p[2]), int(p[3])
            if a == b or not 1 <= k <= 5:
                raise ValueError("bad cross sample")
            cross.append((a, b, k))
        else:
            raise ValueError("malformed sample line")
    if not set(sheet) <= set(pairs):
        raise ValueError("sheet ids must be a subset of the pairs")
    return MakeupSamples(pairs, cross, sheet)


def required_files(s: MakeupSamples) -> list[tuple[str, str]]:
    """Every (id, kind) the pairs and cross samples need, sorted and de-duplicated."""
    need: set[tuple[str, str]] = set()
    for i in s.pairs:
        need.update((i, k) for k in KINDS)
    for a, b, k in s.cross:
        need.add((a, "bare"))
        need.add((b, makeup_kind(k)))
    return sorted(need)


# --- fetching ------------------------------------------------------------------------------------

@dataclass
class MakeupFetchSummary:
    downloaded: int
    skipped: int
    failed: int
    bytes_received: int
    budget_reached: bool


def validate_jpeg_512(data: bytes) -> None:
    if data[:3] != b"\xff\xd8\xff":
        raise ValueError("bad magic bytes")
    rgb = load_rgb_image(data)
    if rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError("unexpected image size")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_makeup(
    cache: Path,
    files: list[tuple[str, str]],
    budget_mb: float = 300,
    workers: int = 4,
    fetcher: RangeFetcher | None = None,
    bf: BudgetedFetcher | None = None,
    table: dict[str, MemberInfo] | None = None,
) -> MakeupFetchSummary:
    """`files` is a list of (id, kind). Resumable, budgeted, writes only under the cache."""
    cache = ensure_cache(cache)
    for i, k in files:
        require_id(i)
        require_kind(k)
    if bf is None:
        bf = BudgetedFetcher(fetcher or ResolvingFetcher(URL), int(budget_mb * 1024 * 1024), PINNED_LENGTHS)
    mpath = cache / "manifest.json"
    manifest: dict[str, str] = json.loads(mpath.read_text()).get("files", {}) if mpath.is_file() else {}

    def have(i: str, k: str) -> bool:
        p = image_path(cache, i, k)
        return p.is_file() and manifest.get(f"{i}_{k}") == _sha256(p)

    todo = [(i, k) for i, k in files if not have(i, k)]
    skipped = len(files) - len(todo)
    downloaded = failed = 0
    stop = threading.Event()
    if todo:
        if table is None:
            table = read_central_directory(bf, URL)

        def one(i: str, k: str) -> str | None:
            if stop.is_set():
                return "stopped"
            info = table.get(f"{i}/{k}.jpg")
            if info is None:
                return "failed"
            try:
                data = fetch_member(bf, URL, info, MAX_MEMBER_BYTES)
            except FetchError as exc:
                if str(exc) == BUDGET_MSG:
                    stop.set()
                    return "stopped"
                return "failed"
            try:
                validate_jpeg_512(data)
            except Exception:  # noqa: BLE001 - outcome, not a crash
                return "failed"
            write_private(image_path(cache, i, k), data)
            manifest[f"{i}_{k}"] = hashlib.sha256(data).hexdigest()
            return None

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for outcome in pool.map(lambda t: one(*t), todo):
                if outcome is None:
                    downloaded += 1
                elif outcome == "failed":
                    failed += 1
        write_private(mpath, json.dumps({"files": manifest}, indent=1, sort_keys=True).encode())
    return MakeupFetchSummary(downloaded, skipped, failed, bf.bytes_received, stop.is_set())


def load_image(cache: Path, image_id: str, kind: str) -> np.ndarray:
    return load_rgb_image(image_path(cache, image_id, kind).read_bytes())
