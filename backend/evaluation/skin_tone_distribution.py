"""Phase 7.5 skin-tone distribution: Lab/ITA/hue statistics over the cached EasyPortrait sample.

Offline only. Reads the private cache (never inside the repo) and the committed sample id list,
and writes **aggregate statistics only** (no ids, no per-image values) to docs/evaluation/.

Usage (from backend/):  .venv/bin/python -m evaluation.skin_tone_distribution [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.services.facial_features.skin_tone import classify_skin_tone
from app.services.facial_profile.builder import build_profile
from app.services.facial_profile.providers import (
    DefaultParsingProvider,
    DefaultZoneProvider,
    MediaPipeLandmarkProvider,
)
from app.services.image_loading import load_rgb_image
from evaluation import dataset as ds
from evaluation.cli import DEFAULT_SAMPLE, REPO_ROOT
from evaluation.runner import build_services

PCTS = (5, 10, 25, 50, 75, 90, 95)
KEYS = ("l_star", "a_star", "b_star", "ita_deg", "hue_deg", "chroma", "cast_chroma")


def summarize(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {}
    s = {"n": float(a.size), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0}
    s.update({f"p{q}": float(np.percentile(a, q)) for q in PCTS})
    return {k: round(v, 4) for k, v in s.items()}


MIN_BAND_N = 5


def _suppressed(n: int) -> dict[str, int | bool]:
    return {"n": n, "suppressed": True} if n < MIN_BAND_N else {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--sample", default=str(DEFAULT_SAMPLE))
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "evaluation" / "skin_tone_distribution_2026-11-12.json"))
    args = ap.parse_args(argv)

    cache = ds.resolve_cache_dir(args.cache_dir)
    entries = ds.parse_sample(Path(args.sample).read_text())
    landmarks, parser = build_services()
    lm_provider = MediaPipeLandmarkProvider(landmarks)
    parsing_provider = DefaultParsingProvider(parser)
    zone_provider = DefaultZoneProvider()

    counts = {"sampled": len(entries), "missing": 0, "undecodable": 0, "not_ok": 0, "classified": 0}
    depth_counts: Counter[str] = Counter()
    under_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    rows: list[tuple[str, str, dict[str, float]]] = []
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
            try:
                profile = build_profile(rgb, lm_provider, parsing_provider, zone_provider, include_masks=True)
            except Exception:  # noqa: BLE001
                counts["not_ok"] += 1
                continue
            comps = profile.components
            if profile.status != "ok" or comps is None or comps.zones.data is None:
                counts["not_ok"] += 1
                continue
            glasses = any(w.code == "glasses_detected" for w in profile.quality.warnings)
            res = classify_skin_tone(rgb, comps.zones.data.zones, glasses)
            counts["classified"] += 1
            depth_counts[res.depth.label] += 1
            under_counts[res.undertone.label] += 1
            quality_counts[res.lighting_quality] += 1
            reason_counts.update(res.lighting_reasons)
            m = {k: v for k, v in res.depth.measurements.items() if k in KEYS and math.isfinite(v)}
            if not m:
                m = {k: v for k, v in res.undertone.measurements.items() if k in KEYS and math.isfinite(v)}
            rows.append((res.depth.label, res.undertone.label, m))
    finally:
        landmarks.close()
        close = getattr(parser, "close", None)
        if callable(close):
            close()

    by_band: dict[str, list[dict[str, float]]] = defaultdict(list)
    unk_by_band: Counter[str] = Counter()
    under_by_band: dict[str, Counter[str]] = defaultdict(Counter)
    for label, under, m in rows:
        by_band[label].append(m)
        under_by_band[label][under] += 1
        if under == "unknown":
            unk_by_band[label] += 1
    n = len(rows)
    unknown_rate = {
        "depth": round(depth_counts["unknown"] / n, 4),
        "undertone": round(under_counts["unknown"] / n, 4),
    } if n else {}

    def dist(ms: list[dict[str, float]]) -> dict[str, dict[str, float]]:
        return {k: summarize([m[k] for m in ms if k in m]) for k in KEYS}

    report = {
        "purpose": "Phase 7.5 skin-tone threshold calibration inputs. Aggregates only.",
        "dataset": "EasyPortrait v2 test split, committed stratified sample (seed 20261110, n=600)",
        "counts": counts,
        "depth_label_counts": dict(depth_counts),
        "undertone_label_counts": dict(under_counts),
        "unknown_rate_overall": unknown_rate,
        "undertone_unknown_rate_per_depth_band": {
            label: _suppressed(len(ms)) or round(unk_by_band[label] / len(ms), 4)
            for label, ms in sorted(by_band.items())
        },
        "undertone_label_counts_per_depth_band": {
            label: _suppressed(len(ms)) or dict(under_by_band[label]) for label, ms in sorted(by_band.items())
        },
        "lighting_reason_counts": dict(reason_counts),
        "exposure_reason_counts": {r: reason_counts[r] for r in ("too_dark", "overexposed")},
        "lighting_quality_counts": dict(quality_counts),
        "distribution_overall": dist([m for _, _, m in rows]),
        "distribution_per_depth_band": {
            label: _suppressed(len(ms)) or dist(ms) for label, ms in sorted(by_band.items())
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
