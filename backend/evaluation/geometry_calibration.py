"""Phase 7.5 step-1 calibration: distribution of the eye / lip / brow classifier measurements.

Offline only, same privacy pattern as face_shape_calibration.py: reads the private cache and the
committed sample ids, runs MediaPipe landmarks, and writes **aggregate statistics only** (no ids, no
per-image values) to docs/evaluation/. The measurement that each classifier bands on is the first
key of its `measurements`; every key is summarised so thresholds can be set from the real data.

Usage (from backend/):  .venv/bin/python -m evaluation.geometry_calibration [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np

from app.services.application_zones.frame import FaceFrame
from app.services.facial_features import brow, eye, lip
from app.services.image_loading import load_rgb_image
from evaluation import dataset as ds
from evaluation.cli import DEFAULT_SAMPLE, REPO_ROOT
from evaluation.runner import build_services

PCTS = (5, 10, 20, 25, 33, 50, 67, 75, 80, 90, 95)
CLASSIFIERS = {n: getattr(m, n) for m in (eye, lip, brow) for n in sorted(dir(m)) if n.startswith("classify_")}


def summarize(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    s = {"n": float(a.size), "mean": float(a.mean()), "sd": float(a.std(ddof=1))}
    s.update({f"p{q}": float(np.percentile(a, q)) for q in PCTS})
    return {k: round(v, 4) for k, v in s.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--sample", default=str(DEFAULT_SAMPLE))
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "evaluation" / "geometry_calibration_2026-11-12.json"))
    args = ap.parse_args(argv)

    cache = ds.resolve_cache_dir(args.cache_dir)
    entries = ds.parse_sample(Path(args.sample).read_text())
    landmarks, _parser = build_services()
    values: dict[str, dict[str, list[float]]] = {n: collections.defaultdict(list) for n in CLASSIFIERS}
    labels: dict[str, collections.Counter[str]] = {n: collections.Counter() for n in CLASSIFIERS}
    counts = {"sampled": len(entries), "missing": 0, "undecodable": 0, "no_single_face": 0, "measured": 0}
    try:
        for e in entries:
            path = ds.image_path(cache, e.image_id)
            if not path.exists():
                counts["missing"] += 1
                continue
            try:
                rgb = load_rgb_image(path.read_bytes())
            except Exception:  # noqa: BLE001 - outcome, not a crash
                counts["undecodable"] += 1
                continue
            res = landmarks.detect_landmarks(rgb)
            if res.status != "ok" or not res.landmarks:
                counts["no_single_face"] += 1
                continue
            h, w = rgb.shape[:2]
            pts = np.array([[q.x * w, q.y * h] for q in sorted(res.landmarks, key=lambda q: q.index)])
            frame = FaceFrame.from_landmarks(pts)
            counts["measured"] += 1
            for name, fn in CLASSIFIERS.items():
                c = fn(pts, frame)
                labels[name][c.label] += 1
                for k, v in c.measurements.items():
                    if math.isfinite(v):
                        values[name][k].append(v)
    finally:
        landmarks.close()

    report = {
        "purpose": "Phase 7.5 step-1 threshold calibration (MediaPipe landmark space). Aggregates only.",
        "dataset": "EasyPortrait v2 test split, committed stratified sample (seed 20261110, n=600)",
        "counts": counts,
        "label_counts": {n: dict(c) for n, c in labels.items()},
        "measurements": {n: {k: summarize(v) for k, v in vs.items() if len(v) > 2} for n, vs in values.items()},
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
