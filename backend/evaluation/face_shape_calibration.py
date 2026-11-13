"""Phase 7.5 face-shape calibration: measurement distribution over the cached EasyPortrait sample.

Offline only. Reads the private cache (never inside the repo) and the committed sample id list,
runs MediaPipe landmarks, and writes **aggregate statistics only** (no ids, no per-image values,
no image-derived data) to docs/evaluation/. Nothing is logged except counts.

Usage (from backend/):  .venv/bin/python -m evaluation.face_shape_calibration [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from app.services.application_zones.frame import FaceFrame
from app.services.facial_features._common import project
from app.services.image_loading import load_rgb_image
from evaluation import dataset as ds
from evaluation.cli import DEFAULT_SAMPLE, REPO_ROOT
from evaluation.runner import build_services

# Face-oval chain, subject's right side, top to chin: 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152.
PAIRS = {
    "cheek_234": (234, 454),
    "forehead_54": (54, 284),
    "jaw_58": (58, 288),
    "jaw_172": (172, 397),
    "jaw_136": (136, 365),
    "jaw_150": (150, 379),
    "chin_176": (176, 400),
    "chin_149": (149, 378),
}
GONION_CANDIDATES = {"58": (58, 288), "172": (172, 397), "136": (136, 365)}
CHEEK, CHIN, TOP, NOSE = (234, 454), 152, 10, 1
PCTS = (5, 10, 25, 50, 75, 90, 95)


def _angle(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    v1, v2 = a - vertex, b - vertex
    c = float(v1 @ v2) / (float(np.linalg.norm(v1) * np.linalg.norm(v2)) or math.nan)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def measure(pts: np.ndarray) -> dict[str, float]:
    frame = FaceFrame.from_landmarks(pts)
    p = {i: project(pts, frame, i) for i in range(pts.shape[0])}
    width = {k: abs(p[b][0] - p[a][0]) for k, (a, b) in PAIRS.items()}
    cheek = width["cheek_234"]
    out = {f"{k}_to_cheek": w / cheek for k, w in width.items() if k != "cheek_234"}
    out["length_to_width"] = abs(p[CHIN][1] - p[TOP][1]) / cheek
    out["taper_176_over_172"] = width["chin_176"] / width["jaw_172"]
    for k, (r, l) in GONION_CANDIDATES.items():
        # Gonial-angle proxy: angle at the candidate between the cheek edge above and the chin.
        out[f"jaw_angle_{k}_deg"] = (_angle(p[CHEEK[0]], p[r], p[CHIN]) + _angle(p[CHEEK[1]], p[l], p[CHIN])) / 2
    nose_u = p[NOSE][0]
    d_r, d_l = abs(p[CHEEK[0]][0] - nose_u), abs(p[CHEEK[1]][0] - nose_u)
    out["yaw_asymmetry"] = abs(d_r - d_l) / (d_r + d_l)
    return out


def summarize(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    s = {"mean": float(a.mean()), "sd": float(a.std(ddof=1))}
    s.update({f"p{q}": float(np.percentile(a, q)) for q in PCTS})
    return {k: round(v, 4) for k, v in s.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--sample", default=str(DEFAULT_SAMPLE))
    ap.add_argument("--max-yaw", type=float, default=0.10, help="frontal subset threshold on yaw_asymmetry")
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "evaluation" / "face_shape_calibration_2026-11-12.json"))
    args = ap.parse_args(argv)

    cache = ds.resolve_cache_dir(args.cache_dir)
    entries = ds.parse_sample(Path(args.sample).read_text())
    landmarks, _parser = build_services()
    rows: list[dict[str, float]] = []
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
            m = measure(pts)
            if all(math.isfinite(v) for v in m.values()):
                rows.append(m)
                counts["measured"] += 1
    finally:
        landmarks.close()

    frontal = [r for r in rows if r["yaw_asymmetry"] <= args.max_yaw]
    keys = sorted(rows[0]) if rows else []
    report = {
        "purpose": "Phase 7.5 face-shape threshold calibration (MediaPipe landmark space). Aggregates only.",
        "dataset": "EasyPortrait v2 test split, committed stratified sample (seed 20261110, n=600)",
        "counts": {**counts, "frontal": len(frontal), "frontal_max_yaw_asymmetry": args.max_yaw},
        "all": {k: summarize([r[k] for r in rows]) for k in keys},
        "frontal": {k: summarize([r[k] for r in frontal]) for k in keys} if frontal else {},
        "correlation_frontal": {
            f"{a}~{b}": round(float(np.corrcoef([r[a] for r in frontal], [r[b] for r in frontal])[0, 1]), 3)
            for a, b in [
                ("length_to_width", "jaw_172_to_cheek"),
                ("forehead_54_to_cheek", "jaw_172_to_cheek"),
                ("jaw_172_to_cheek", "chin_176_to_cheek"),
                ("jaw_172_to_cheek", "taper_176_over_172"),
                ("length_to_width", "forehead_54_to_cheek"),
                ("length_to_width", "jaw_angle_172_deg"),
            ]
        } if len(frontal) > 2 else {},
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
