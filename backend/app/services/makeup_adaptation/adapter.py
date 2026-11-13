"""Pure blueprint-to-profile adaptation (step 1: zone selection and user masks only).

No I/O and no logging. CatalogDriftError propagates as a server fault.
"""

import time
from collections.abc import Callable
from typing import Any

from app.schemas.adapted_makeup_plan import (
    AdaptedComponent,
    AdaptedMakeupPlan,
    PlanProviderSummary,
    PlanWarning,
    ReferenceSummary,
)
from app.schemas.face_segmentation import Zone
from app.schemas.facial_profile import FaceFrameData, FacialProfile, FeaturesData
from app.schemas.makeup_blueprint import BlueprintComponent, MakeupBlueprint
from app.services.application_zones import catalog
from app.services.makeup_mapping import ZoneOutsideLayerError, map_zones
from app.services.makeup_mapping.mapper import ZoneMapping

from .directions import attach_angles, usable_frame
from .masks import union_zone_masks
from .rules import evaluate
from .versions import PLAN_SCHEMA_VERSION

_NOTE_MESSAGES: dict[str, str] = {
    "reference_zones_outside_layer": "Reference zones are outside the product layer; the layer default was used.",
    "no_zones_for_placement": "No application zones match this product and placement.",
    "zone_unavailable_on_user": "Some zones are not available on the user's face.",
    "zone_mask_invalid": "Some zone masks on the user's face were invalid and skipped.",
    "no_stage": "This product has no application stage.",
    "direction_angles_unavailable": "Direction angles could not be computed for the user's face.",
}
_PROFILE_MESSAGES: dict[str, str] = {
    "no_face_detected": "No face was detected in the user's photo.",
    "multiple_faces_detected": "Multiple faces were detected in the user's photo.",
}


def _user_zones(profile: FacialProfile) -> dict[str, Zone] | None:
    comps = profile.components
    if comps is None or comps.zones.data is None:
        return None
    return {z.id: z for z in comps.zones.data.zones if z.status != "unavailable"}


def _features(profile: FacialProfile) -> FeaturesData | None:
    comps = profile.components
    if comps is None or comps.features is None:
        return None
    return comps.features.data


def _providers(profile: FacialProfile, blueprint: MakeupBlueprint | None) -> list[PlanProviderSummary]:
    out: list[PlanProviderSummary] = []
    comps = profile.components
    if comps is not None:
        for c in (comps.landmarks, comps.frame, comps.parsing, comps.zones, comps.features):
            if c is not None:
                p = c.meta.provider
                out.append(PlanProviderSummary(id=p.id, kind=p.kind, algorithm_version=p.algorithm_version))
    if blueprint is not None:
        for rp in blueprint.providers:
            out.append(
                PlanProviderSummary(id=rp.id, kind=rp.kind, algorithm_version=rp.algorithm_version, model=rp.model)
            )
    return out


def _map(bc: BlueprintComponent) -> tuple[ZoneMapping, bool, bool]:
    """(mapping, used_reference_zones, outside_layer)."""
    c = bc.component
    try:
        return map_zones(c.product_type, c.placement, c.target_zones), bool(c.target_zones), False
    except ZoneOutsideLayerError:
        return map_zones(c.product_type, c.placement, ()), False, True


def _adapt_component(
    bc: BlueprintComponent,
    original_index: int,
    zones: dict[str, Zone] | None,
    image: Any,
    features: FeaturesData | None = None,
    frame: FaceFrameData | None = None,
) -> tuple[tuple[bool, int, int], dict[str, Any]]:
    c = bc.component
    mapping, used_reference, outside = _map(bc)
    notes: list[str] = []
    if outside:
        notes.append("reference_zones_outside_layer")
    if mapping.stage_key is None:
        notes.append("no_stage")
    if not mapping.zone_ids:
        notes.append("no_zones_for_placement")

    kept: list[str] = []
    dropped = False
    available = zones or {}
    for zid in mapping.zone_ids:
        if zid in available:
            kept.append(zid)
        else:
            dropped = True
    if dropped:
        notes.append("zone_unavailable_on_user")

    evaluations, adjustments, new_zones = evaluate(features, c, kept, available.keys())
    adjustments = attach_angles(adjustments, frame)
    if frame is None and any(a.effect == "direction" for a in adjustments):
        notes.append("direction_angles_unavailable")
    reason: str | None = mapping.reason
    if new_zones is not None:
        kept = list(new_zones)
        reason = next(a.reason for a in adjustments if a.effect == "zones")

    union = union_zone_masks([available[z] for z in kept], image) if kept else None
    if union is not None and union.invalid_ids:
        notes.append("zone_mask_invalid")

    if not kept:
        source = "none"
    elif new_zones is not None or mapping.source == "placement":
        source = "placement"
    elif used_reference or mapping.source == "explicit":
        source = "reference"
    else:
        source = "layer_default"

    stage_number = next((s.number for s in catalog.STAGES if s.key == mapping.stage_key), None)
    has_mask = union is not None and union.png_b64 is not None
    data: dict[str, Any] = {
        "component": c,
        "stage_key": mapping.stage_key,
        "stage_number": stage_number,
        "layer_key": mapping.layer_key,
        "zones": kept,
        "zone_selection": {"source": source, "reason": reason},
        "sources": dict(bc.sources),
        "confidence": c.confidence,
        "adjustments": adjustments,
        "rule_evaluations": evaluations,
        "notes": notes,
    }
    if has_mask and union is not None and union.bbox is not None and union.centroid is not None:
        x, y, w, h = union.bbox
        data["mask"] = {"png_base64": union.png_b64}
        data["bbox"] = {"x": x, "y": y, "width": w, "height": h}
        data["centroid"] = union.centroid
        data["pixel_count"] = union.pixel_count
    key = (stage_number is None, stage_number or 0, original_index)
    return key, data


def adapt(
    profile: FacialProfile,
    blueprint: MakeupBlueprint,
    clock: Callable[[], float] = time.perf_counter,
) -> AdaptedMakeupPlan:
    start = clock()

    def timings() -> dict[str, float]:
        return {"adapt": max(0.0, (clock() - start) * 1000.0)}

    if profile.status != "ok":
        return AdaptedMakeupPlan(
            status=profile.status,
            message=_PROFILE_MESSAGES[profile.status],
            image=profile.image,
            timings_ms=timings(),
        )

    if blueprint.status != "ok":
        return AdaptedMakeupPlan(
            status="reference_unavailable",
            message="The reference image could not be analysed.",
            image=profile.image,
            providers=_providers(profile, None),
            timings_ms=timings(),
        )

    style = blueprint.style
    reference = ReferenceSummary(
        image=blueprint.image,
        blueprint_schema_version=blueprint.schema_version,
        style_labels=[] if style is None else list(style.labels),
        overall_confidence=blueprint.quality.overall_confidence,
        component_count=len(blueprint.components),
    )
    providers = _providers(profile, blueprint)

    if not blueprint.components:
        return AdaptedMakeupPlan(
            status="no_makeup_detected",
            message="No makeup was detected in the reference image.",
            image=profile.image,
            reference=reference,
            providers=providers,
            timings_ms=timings(),
        )

    zones = _user_zones(profile)
    features = _features(profile)
    frame = usable_frame(profile)
    built = [
        _adapt_component(bc, i, zones, profile.image, features, frame)
        for i, bc in enumerate(blueprint.components)
    ]
    built.sort(key=lambda item: item[0])
    components = [AdaptedComponent.model_validate({**data, "index": i}) for i, (_, data) in enumerate(built)]

    warnings: list[PlanWarning] = []
    if zones is None:
        warnings.append(
            PlanWarning(code="zones_unavailable", message="Zones are unavailable for the user's photo.")
        )
    if features is None:
        warnings.append(
            PlanWarning(code="features_unavailable", message="Facial features are unavailable for the user's photo.")
        )
    seen: set[str] = set()
    for comp in components:
        for note in comp.notes:
            if note not in seen:
                seen.add(note)
                warnings.append(PlanWarning(code=note, message=_NOTE_MESSAGES[note]))

    return AdaptedMakeupPlan(
        status="ok",
        image=profile.image,
        reference=reference,
        components=components,
        warnings=warnings,
        providers=providers,
        timings_ms=timings(),
    )
