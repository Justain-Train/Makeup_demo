"""Reference analysis orchestration: image -> FacialProfile -> CV components -> VLM semantics.

Returns a MakeupBlueprint that never contains masks, labels, landmarks or image data.
A failing VLM degrades to a CV-only result with a vlm_failed warning; it never fails the request.

overall_confidence:
  base = mean of component.confidence over components that have one; if there are none, the CV
  analysis confidence; if that is None too, overall_confidence is None.
  If any warning code in {vlm_failed, vlm_refused, glasses_present} is present, base *= 0.9.
  The result is clamped to [0, 1] and rounded to 4 decimals.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import Settings
from app.schemas.makeup_blueprint import (
    BlueprintComponent,
    BlueprintWarning,
    MakeupBlueprint,
    ReferenceProviderInfo,
)
from app.services.facial_profile.builder import build_profile
from app.services.facial_profile.providers import (
    DefaultZoneProvider,
    LandmarkProvider,
    ParsingProvider,
    ZoneProvider,
)
from app.services.makeup_blueprint.cv_analysis import (
    CvAnalysisError,
    CvReferenceAnalyzer,
    ReferenceAnalyzer,
)
from app.services.makeup_blueprint.vlm import (
    StubVlmProvider,
    VlmAnalysis,
    VlmProvider,
    attach_annotations,
    build_vlm_provider,
)
from app.services.makeup_mapping import CatalogDriftError
from app.services.segmentation.face_parser import FaceParserBusyError

MAX_WARNINGS = 32
VLM_PENALTY_FACTOR = 0.9
_PENALTY_CODES = frozenset({"vlm_failed", "vlm_refused", "glasses_present"})
_MSG_VLM_FAILED = "Semantic analysis failed."


class ReferenceAnalysisError(Exception):
    """Fixed message only; never carries image-derived or provider text."""

    def __init__(self, stage: str) -> None:
        super().__init__("Reference analysis failed.")
        self.stage = stage


class ParserUnavailableError(ReferenceAnalysisError):
    def __init__(self) -> None:
        super().__init__("profile")


@dataclass(frozen=True)
class ReferenceDeps:
    landmark_provider: LandmarkProvider
    parsing_provider: ParsingProvider | None
    zone_provider: ZoneProvider
    cv_analyzer: ReferenceAnalyzer
    vlm_provider: VlmProvider


def build_reference_deps(
    settings: Settings,
    landmark_provider: LandmarkProvider,
    parsing_provider: ParsingProvider | None,
    zone_provider: ZoneProvider | None = None,
    vlm_client: Any = None,
) -> ReferenceDeps:
    return ReferenceDeps(
        landmark_provider=landmark_provider,
        parsing_provider=parsing_provider,
        zone_provider=zone_provider if zone_provider is not None else DefaultZoneProvider(),
        cv_analyzer=CvReferenceAnalyzer(),
        vlm_provider=build_vlm_provider(settings, vlm_client),
    )


def _vlm_info(provider: VlmProvider) -> ReferenceProviderInfo:
    try:
        return provider.info()
    except Exception:
        return StubVlmProvider().info()


def _run_vlm(provider: VlmProvider, image: np.ndarray, cv: list[BlueprintComponent]) -> VlmAnalysis:
    try:
        return provider.analyze(image, cv)
    except Exception:
        return VlmAnalysis(
            info=_vlm_info(provider),
            warnings=[BlueprintWarning(code="vlm_failed", message=_MSG_VLM_FAILED)],
        )


def _dedupe(*groups: list[BlueprintWarning]) -> list[BlueprintWarning]:
    seen: set[str] = set()
    out: list[BlueprintWarning] = []
    for group in groups:
        for w in group:
            if w.code in seen:
                continue
            seen.add(w.code)
            out.append(w)
    return out[:MAX_WARNINGS]


def _overall_confidence(
    components: list[BlueprintComponent],
    cv_confidence: float | None,
    warnings: list[BlueprintWarning],
) -> float | None:
    values = [c.component.confidence for c in components if c.component.confidence is not None]
    base = sum(values) / len(values) if values else cv_confidence
    if base is None:
        return None
    if any(w.code in _PENALTY_CODES for w in warnings):
        base *= VLM_PENALTY_FACTOR
    return round(min(1.0, max(0.0, base)), 4)


def _ms(start: float, end: float) -> float:
    return round((end - start) * 1000.0, 2)


def analyze_reference(
    image: np.ndarray,
    *,
    deps: ReferenceDeps,
    clock: Callable[[], float] = time.perf_counter,
) -> MakeupBlueprint:
    if deps.parsing_provider is None:
        raise ParserUnavailableError()

    t0 = clock()
    try:
        profile = build_profile(
            image, deps.landmark_provider, deps.parsing_provider, deps.zone_provider, True
        )
    except FaceParserBusyError:
        raise
    except Exception:
        raise ReferenceAnalysisError("profile") from None
    t1 = clock()

    if profile.status != "ok":
        try:
            return MakeupBlueprint.model_validate(
                {
                    "status": profile.status,
                    "message": profile.message,
                    "image": profile.image.model_dump(),
                    "timings_ms": {"profile": _ms(t0, t1), "total": _ms(t0, t1)},
                }
            )
        except Exception:
            raise ReferenceAnalysisError("profile") from None

    try:
        cv = deps.cv_analyzer.analyze(image, profile)
    except CatalogDriftError:
        raise ReferenceAnalysisError("catalog") from None
    except CvAnalysisError:
        raise ReferenceAnalysisError("cv") from None
    except Exception:
        raise ReferenceAnalysisError("cv") from None
    t2 = clock()

    vlm = _run_vlm(deps.vlm_provider, image, cv.components)
    t3 = clock()

    try:
        components = attach_annotations(cv.components, vlm)
        warnings = _dedupe(cv.warnings, vlm.warnings)
        return MakeupBlueprint.model_validate(
            {
                "status": "ok",
                "message": profile.message,
                "image": profile.image.model_dump(),
                "components": components,
                "style": vlm.style,
                "quality": {
                    "warnings": warnings,
                    "overall_confidence": _overall_confidence(components, cv.confidence, warnings),
                },
                "providers": [deps.cv_analyzer.info(), _vlm_info(deps.vlm_provider)],
                "timings_ms": {
                    "profile": _ms(t0, t1),
                    "cv": _ms(t1, t2),
                    "vlm": _ms(t2, t3),
                    "total": _ms(t0, t3),
                },
            }
        )
    except Exception:
        raise ReferenceAnalysisError("cv") from None
