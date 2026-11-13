"""Phase 8 rule-firing evaluation: how often each feature rule applies over the cached EasyPortrait sample.

Offline only. Reads the private cache (never inside the repo) and the committed sample id list, and writes
**aggregate counts only** (no ids, no per-image values) to docs/evaluation/. Nothing is logged except counts.

Usage (from backend/):  .venv/bin/python -m evaluation.rule_firing [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.schemas.facial_profile import ImageSize
from app.schemas.makeup_blueprint import BlueprintComponent, MakeupBlueprint
from app.schemas.makeup_component import MakeupComponent
from app.services.facial_profile.builder import build_profile
from app.services.facial_profile.providers import (
    DefaultFeatureProvider,
    DefaultParsingProvider,
    DefaultZoneProvider,
    MediaPipeLandmarkProvider,
)
from app.services.image_loading import load_rgb_image
from app.services.makeup_adaptation.adapter import adapt
from app.services.makeup_adaptation.rules import RULES
from evaluation import dataset as ds
from evaluation.cli import DEFAULT_SAMPLE, REPO_ROOT
from evaluation.runner import build_services

FULL = (
    ("eyebrows", "brow_pencil"),
    ("eye", "eyeliner"),
    ("eye", "eyeshadow"),
    ("blush", "powder_blush"),
    ("bronzer_contour", "contour_powder"),
    ("bronzer_contour", "bronzer"),
    ("lip", "lip_liner"),
    ("lip", "lipstick"),
)
CV_ONLY = (
    ("lip", "lipstick"),
    ("blush", "powder_blush"),
    ("eye", "eyeshadow"),
    ("highlighter", "powder_highlighter"),
    ("bronzer_contour", "contour_powder"),
)
OUTCOMES = (
    "applied", "skipped_uncertain", "skipped_not_triggered",
    "skipped_zone_unavailable", "skipped_not_applicable", "skipped_conflict",
)


def _blueprint(spec: tuple[tuple[str, str], ...]) -> MakeupBlueprint:
    comps = [
        BlueprintComponent(
            component=MakeupComponent(category=cat, product_type=pt),
            sources={"category": "cv", "product_type": "cv"},
        )
        for cat, pt in spec
    ]
    return MakeupBlueprint(status="ok", image=ImageSize(width=200, height=200), components=comps)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--sample", default=str(DEFAULT_SAMPLE))
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "evaluation" / "rule_firing_2026-11-12.json"))
    args = ap.parse_args(argv)

    cache = ds.resolve_cache_dir(args.cache_dir)
    entries = ds.parse_sample(Path(args.sample).read_text())
    landmarks, parser = build_services()
    scenarios = {"full": _blueprint(FULL), "cv_only": _blueprint(CV_ONLY)}
    counts = {"sampled": len(entries), "missing": 0, "undecodable": 0, "no_profile": 0, "evaluated": 0}
    agg: dict[str, dict[str, dict[str, int]]] = {
        s: {r.rule_id: {o: 0 for o in OUTCOMES} for r in RULES} for s in scenarios
    }
    faces_applied = {s: 0 for s in scenarios}
    try:
        lp = MediaPipeLandmarkProvider(landmarks)
        pp = DefaultParsingProvider(parser)
        zp = DefaultZoneProvider()
        fp = DefaultFeatureProvider()
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
            profile = build_profile(rgb, lp, pp, zp, True, fp)
            if profile.status != "ok":
                counts["no_profile"] += 1
                continue
            counts["evaluated"] += 1
            for name, bp in scenarios.items():
                plan = adapt(profile, bp)
                any_applied = False
                for comp in plan.components:
                    for ev in comp.rule_evaluations:
                        agg[name][ev.rule_id][ev.outcome] += 1
                        any_applied = any_applied or ev.outcome == "applied"
                if any_applied:
                    faces_applied[name] += 1
    finally:
        landmarks.close()
        close = getattr(parser, "close", None)
        if callable(close):
            close()

    n = counts["evaluated"]
    report = {
        "purpose": "Phase 8 rule-firing rates (component-evaluations per rule per scenario). Aggregates only.",
        "dataset": "EasyPortrait v2 test split, committed stratified sample (seed 20261110, n=600)",
        "counts": counts,
        "scenarios": {
            s: {
                "faces_with_applied_rule": faces_applied[s],
                "faces_with_applied_rule_pct": round(100.0 * faces_applied[s] / n, 2) if n else 0.0,
                "rules": {
                    rid: {
                        "counts": c,
                        "pct": {o: round(100.0 * v / max(1, sum(c.values())), 2) for o, v in c.items()},
                    }
                    for rid, c in agg[s].items()
                },
            }
            for s in scenarios
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
