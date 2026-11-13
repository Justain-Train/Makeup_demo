"""Contact sheet: bare | reference | preview | truth rows (offline, private output outside the repo)."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw

from evaluation import makeup_dataset as md
from evaluation.dataset import refuse_inside_repo, write_private
from evaluation.makeup_pipeline import PairResult

TILE = 256
HEADER = 18
COLUMNS = ("bare", "reference", "preview", "truth")
N_CROSS_ROWS = 4
GREY = (128, 128, 128)


class Runner(Protocol):
    def run(self, selfie_rgb: np.ndarray, reference_rgb: np.ndarray) -> PairResult: ...


def _tile(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb).resize((TILE, TILE), Image.LANCZOS)


def _grey(label: str) -> Image.Image:
    im = Image.new("RGB", (TILE, TILE), GREY)
    ImageDraw.Draw(im).text((8, TILE // 2 - 6), label[:40], fill=(255, 255, 255))
    return im


def _preview_tile(res: PairResult) -> Image.Image:
    if res.render is None:
        return _grey(res.skip_reason or "skipped")
    return _tile(res.render.image)


def build_sheet(cache: Path, samples: md.MakeupSamples, runner: Runner, out_path: Path) -> Path:
    out = refuse_inside_repo(Path(out_path), "output file")
    rows: list[list[Image.Image]] = []
    for idx, pid in enumerate(samples.sheet):
        k = md.makeup_kind((idx % 5) + 1)
        bare = md.load_image(cache, pid, "bare")
        ref = md.load_image(cache, pid, k)
        res = runner.run(bare, ref)
        rows.append([_tile(bare), _tile(ref), _preview_tile(res), _tile(ref)])
    for a, b, k in samples.cross[:N_CROSS_ROWS]:
        bare = md.load_image(cache, a, "bare")
        ref = md.load_image(cache, b, md.makeup_kind(k))
        res = runner.run(bare, ref)
        rows.append([_tile(bare), _tile(ref), _preview_tile(res), Image.new("RGB", (TILE, TILE), (255, 255, 255))])

    sheet = Image.new("RGB", (TILE * len(COLUMNS), HEADER + TILE * max(1, len(rows))), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for c, name in enumerate(COLUMNS):
        draw.text((c * TILE + 6, 3), name, fill=(0, 0, 0))
    for r, tiles in enumerate(rows):
        for c, t in enumerate(tiles):
            sheet.paste(t, (c * TILE, HEADER + r * TILE))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    out.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(out.parent, 0o700)
    write_private(out, buf.getvalue())
    return out
