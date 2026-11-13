"""Selfie + reference images -> AdaptedMakeupPlan. Sequential; no I/O beyond the injected providers."""

import numpy as np

from app.schemas.adapted_makeup_plan import AdaptedMakeupPlan
from app.services.facial_profile.builder import build_profile
from app.services.facial_profile.providers import FeatureProvider, LandmarkProvider, ParsingProvider, ZoneProvider
from app.services.makeup_blueprint.service import ParserUnavailableError, ReferenceDeps, analyze_reference

from .adapter import adapt


def _drop_masks(plan: AdaptedMakeupPlan) -> AdaptedMakeupPlan:
    data = plan.model_dump()
    for comp in data["components"]:
        comp["mask"] = None
    return AdaptedMakeupPlan.model_validate(data)


def adapt_from_images(
    selfie: np.ndarray,
    reference: np.ndarray,
    profile_deps: tuple[LandmarkProvider, ParsingProvider | None, ZoneProvider, FeatureProvider],
    reference_deps: ReferenceDeps,
    include_masks: bool = True,
) -> AdaptedMakeupPlan:
    """Raises ParserUnavailableError, FaceParserBusyError, ReferenceAnalysisError, CatalogDriftError."""
    landmark_provider, parsing_provider, zone_provider, feature_provider = profile_deps
    if parsing_provider is None:
        raise ParserUnavailableError()
    profile = build_profile(
        selfie,
        landmark_provider,
        parsing_provider,
        zone_provider,
        include_masks=True,
        feature_provider=feature_provider,
    )
    blueprint = analyze_reference(reference, deps=reference_deps)
    plan = adapt(profile, blueprint)
    return plan if include_masks else _drop_masks(plan)
