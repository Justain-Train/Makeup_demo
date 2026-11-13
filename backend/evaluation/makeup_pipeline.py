"""In-process selfie + reference pipeline for evaluation (no HTTP, no rate limiter, VLM off)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from app.schemas.adapted_makeup_plan import AdaptedMakeupPlan
from app.schemas.facial_profile import FacialProfile
from app.schemas.makeup_blueprint import MakeupBlueprint
from app.services.facial_profile.builder import build_profile
from app.services.facial_profile.providers import (
    DefaultFeatureProvider,
    DefaultParsingProvider,
    DefaultZoneProvider,
    MediaPipeLandmarkProvider,
)
from app.services.makeup_adaptation.adapter import adapt
from app.services.makeup_blueprint.cv_analysis import CvReferenceAnalyzer
from app.services.makeup_blueprint.service import ReferenceDeps, analyze_reference
from app.services.makeup_blueprint.vlm import StubVlmProvider
from app.services.makeup_preview.renderer import RenderResult, render_layers
from evaluation.runner import build_services


@dataclass(frozen=True)
class PairResult:
    profile_ok: bool
    blueprint: MakeupBlueprint | None
    plan: AdaptedMakeupPlan | None
    render: RenderResult | None
    skip_reason: str | None
    ms: float


class PairRunner:
    def __init__(self) -> None:
        self._landmarks, self._parser = build_services()
        self._lp = MediaPipeLandmarkProvider(self._landmarks)
        self._pp = DefaultParsingProvider(self._parser)
        self._zp = DefaultZoneProvider()
        self._fp = DefaultFeatureProvider()
        self._deps = ReferenceDeps(self._lp, self._pp, self._zp, CvReferenceAnalyzer(), StubVlmProvider())

    def run(self, selfie_rgb: np.ndarray, reference_rgb: np.ndarray) -> PairResult:
        t0 = time.perf_counter()
        profile_ok = False
        blueprint: MakeupBlueprint | None = None
        plan: AdaptedMakeupPlan | None = None
        reason: str | None = None
        render: RenderResult | None = None

        def ms() -> float:
            return round((time.perf_counter() - t0) * 1000.0, 2)

        try:
            profile = build_profile(selfie_rgb, self._lp, self._pp, self._zp, True, self._fp)
            profile_ok = profile.status == "ok"
            if not profile_ok:
                return PairResult(False, None, None, None, f"profile_{profile.status}", ms())
            blueprint = analyze_reference(reference_rgb, deps=self._deps)
            if blueprint.status != "ok":
                return PairResult(True, blueprint, None, None, f"reference_{blueprint.status}", ms())
            plan = adapt(profile, blueprint)
            render = render_layers(selfie_rgb, plan)
        except Exception as exc:  # noqa: BLE001 - outcome, not a crash
            reason = type(exc).__name__
            render = None
        return PairResult(profile_ok, blueprint, plan, render, reason, ms())

    def profile(self, rgb: np.ndarray) -> FacialProfile:
        """Reference-style profile (masks included, no features), as analyze_reference builds it."""
        return build_profile(rgb, self._lp, self._pp, self._zp, True)

    def close(self) -> None:
        self._landmarks.close()
        close = getattr(self._parser, "close", None)
        if callable(close):
            close()
