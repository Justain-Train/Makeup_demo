"""Assembles a FacialProfile from swappable providers. Assembly only: no classification."""

import time

import numpy as np

from app.schemas.face_segmentation import (
    AnatomicalLayer,
    ParseWindow as ParseWindowSchema,
    Point,
    SegmentationWarning,
    Stage,
    Zone,
)
from app.schemas.facial_profile import (
    Component,
    ComponentMeta,
    ComponentSet,
    FaceFrameData,
    FacialProfile,
    FeaturesData,
    LandmarksData,
    ParsingData,
    ProfileQuality,
    ProviderInfo,
    ZonesData,
)
from app.services.application_zones.catalog import STAGES
from app.services.application_zones.frame import MIDLINE, FaceFrame
from app.services.face_regions import (
    EDGE_TOLERANCE_PX,
    HEAD_TILT_DEGREES,
    LOW_RESOLUTION_IPD_PX,
    _encode_zone,
    _region_infos,
    _warnings,
    alignment_check,
    iris_checks,
)
from app.services.facial_analysis.geometry import compute_basic_measurements, landmark_pixels
from app.services.facial_profile.providers import (
    FeatureProvider,
    LandmarkProvider,
    ParsingProvider,
    ZoneProvider,
    frame_provider_info,
)
from app.services.facial_profile.versions import COMPONENT_VERSIONS
from app.services.segmentation import masks as mk
from app.services.segmentation.face_parser import FaceParserBusyError
from app.services.segmentation.regions import NOT_PARSED
from app.services.segmentation.sides import assign_sides

_CORE_REGIONS = ("skin", "eye_l", "eye_r", "brow_l", "brow_r", "lip_upper", "lip_lower")


def _meta(
    slot: str, info: ProviderInfo, status: str, confidence: float | None = None, reason: str | None = None
) -> ComponentMeta:
    return ComponentMeta(
        component_version=COMPONENT_VERSIONS[slot],
        status=status,  # type: ignore[arg-type]
        confidence=confidence,
        provider=info,
        reason=reason,
    )


def _unavailable(slot: str, info: ProviderInfo, reason: str) -> Component:
    return Component(meta=_meta(slot, info, "unavailable", reason=reason), data=None)


def build_profile(
    image: np.ndarray,
    landmark_provider: LandmarkProvider,
    parsing_provider: ParsingProvider,
    zone_provider: ZoneProvider,
    include_masks: bool,
    feature_provider: FeatureProvider | None = None,
) -> FacialProfile:
    """Raises FaceParserBusyError and landmark-provider errors; every other later-stage failure
    becomes an 'unavailable' component carrying only the exception type name."""
    t_start = time.perf_counter()
    height, width = image.shape[:2]
    result = landmark_provider.detect(image)
    t_landmarks = time.perf_counter()
    timings: dict[str, float] = {"landmarks": (t_landmarks - t_start) * 1000}
    if result.status != "ok":
        return FacialProfile(
            status=result.status,
            message=result.message,
            image=result.image,
            timings_ms={"landmarks": round(timings["landmarks"], 2), "total": round(timings["landmarks"], 2)},
        )

    lm = landmark_pixels(result)
    pts = np.array([lm[i] for i in range(len(lm))], dtype=np.float64)
    bbox = (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))

    lm_info, parse_info, zone_info, frame_info = (
        landmark_provider.info(),
        parsing_provider.info(),
        zone_provider.info(),
        frame_provider_info(),
    )

    # Frame (landmarks only)
    t0 = time.perf_counter()
    frame: FaceFrame | None = None
    frame_error: str | None = None
    try:
        frame = FaceFrame.from_landmarks(pts)
    except ValueError as exc:
        frame_error = type(exc).__name__
    timings["frame"] = (time.perf_counter() - t0) * 1000

    # Parsing
    parse = None
    parse_error: str | None = None
    try:
        parse = parsing_provider.parse(image, bbox)
        parse.labels = assign_sides(parse.labels, pts, parse.origin)
    except FaceParserBusyError:
        raise
    except Exception as exc:  # noqa: BLE001 - degrade the component, never leak details
        parse = None
        parse_error = type(exc).__name__
    if parse is not None:
        timings.update(parse.timings_ms)

    # Zones
    t0 = time.perf_counter()
    derived = None
    alignment = None
    zones_error: str | None = None
    if parse is not None:
        alignment = alignment_check(pts, parse.labels, parse.origin)
    if parse is None:
        zones_error = "depends on parsing"
    elif frame is None:
        zones_error = "depends on frame"
    else:
        try:
            derived = zone_provider.derive(frame, parse.labels, parse.confidence, parse.origin)
            for side, ok in zip("rl", iris_checks(pts, parse.labels, parse.origin)):
                if not ok:
                    for z in derived:
                        if z.spec.group == "eye" and z.spec.id.endswith(f"_{side}") and z.status == "ok":
                            z.status = "low_confidence"
                            z.reason = "The parsed eye does not contain this side's iris landmark (parser/landmark mismatch)."
        except FaceParserBusyError:
            raise
        except Exception as exc:  # noqa: BLE001
            derived = None
            zones_error = type(exc).__name__
    timings["zones"] = (time.perf_counter() - t0) * 1000

    # Landmarks component
    outside = bool(
        pts[:, 0].min() < -EDGE_TOLERANCE_PX
        or pts[:, 1].min() < -EDGE_TOLERANCE_PX
        or pts[:, 0].max() > width + EDGE_TOLERANCE_PX
        or pts[:, 1].max() > height + EDGE_TOLERANCE_PX
    )
    low_res = frame is not None and frame.ipd < LOW_RESOLUTION_IPD_PX
    tilted = frame is not None and abs(frame.roll_degrees) > HEAD_TILT_DEGREES

    lm_reason = None
    if outside:
        lm_reason = "Part of the face is outside the image."
    elif low_res:
        lm_reason = "The face is small."
    landmarks_component = Component[LandmarksData](
        meta=_meta("landmarks", lm_info, "degraded" if lm_reason else "ok", reason=lm_reason),
        data=LandmarksData(
            thresholds=result.thresholds,
            landmarks=result.landmarks,
            measurements=compute_basic_measurements(result),
        ),
    )

    # Frame component
    if frame is None:
        frame_component = _unavailable("frame", frame_info, frame_error or "ValueError")
    else:
        frame_reason = "The face is small." if low_res else ("Head is tilted." if tilted else None)
        frame_component = Component[FaceFrameData](
            meta=_meta("frame", frame_info, "degraded" if frame_reason else "ok", reason=frame_reason),
            data=FaceFrameData(
                ipd_px=float(frame.ipd),
                roll_deg=float(frame.roll_degrees),
                u=(float(frame.u[0]), float(frame.u[1])),
                v=(float(frame.v[0]), float(frame.v[1])),
                midline=[(round(float(pts[i, 0]), 1), round(float(pts[i, 1]), 1)) for i in MIDLINE],
            ),
        )

    # Parsing component
    t0 = time.perf_counter()
    infos = []
    if parse is None:
        parsing_component = _unavailable("parsing", parse_info, parse_error or "Unknown")
    else:
        infos = _region_infos(parse)
        present = [r for r in infos if r.present and r.mean_probability is not None]
        total = sum(r.pixel_count for r in present)
        confidence = (
            min(1.0, max(0.0, sum(r.pixel_count * r.mean_probability for r in present) / total))  # type: ignore[operator]
            if total
            else None
        )
        present_names = {r.name for r in infos if r.present}
        reasons = []
        if alignment is not None and not alignment.passed:
            reasons.append("Parsed regions do not line up with landmarks.")
        if not all(n in present_names for n in _CORE_REGIONS):
            reasons.append("A core region was not found.")
        label_map = None
        if include_masks:
            label_map = AnatomicalLayer(
                origin=Point(x=parse.origin[0], y=parse.origin[1]),
                width=int(parse.labels.shape[1]),
                height=int(parse.labels.shape[0]),
                not_parsed_value=NOT_PARSED,
                label_map_png_base64=mk.encode_png_base64(parse.labels),
                regions=infos,
            )
        parsing_component = Component[ParsingData](
            meta=_meta(
                "parsing", parse_info, "degraded" if reasons else "ok", confidence, " ".join(reasons) or None
            ),
            data=ParsingData(
                parse_window=ParseWindowSchema(x=parse.window.x, y=parse.window.y, size=parse.window.size),
                includes_masks=include_masks,
                regions=infos,
                label_map=label_map,
            ),
        )

    # Zones component
    full_zone_models: list[Zone] = []
    if derived is None:
        zones_component = _unavailable("zones", zone_info, zones_error or "Unknown")
    else:
        zone_models: list[Zone] = [_encode_zone(z, parse.origin) for z in derived]  # type: ignore[union-attr]
        full_zone_models = zone_models
        if not include_masks:
            zone_models = [z.model_copy(update={"mask": None}) for z in zone_models]
        confs = [z.confidence for z in zone_models if z.confidence is not None]
        zconf = min(1.0, max(0.0, sum(confs) / len(confs))) if confs else None
        degraded = alignment is not None and not alignment.passed
        zones_component = Component[ZonesData](
            meta=_meta(
                "zones",
                zone_info,
                "degraded" if degraded else "ok",
                zconf,
                "Parsed regions do not line up with landmarks." if degraded else None,
            ),
            data=ZonesData(
                zones=zone_models,
                stages=[
                    Stage(number=st.number, key=st.key, name=st.name, layers={k: list(v) for k, v in st.layers.items()})
                    for st in STAGES
                ],
            ),
        )
    timings["encode"] = (time.perf_counter() - t0) * 1000

    # Quality
    quality = ProfileQuality()
    if parse is not None and alignment is not None:
        warnings = []
        retake: list = []
        if frame is None:
            if outside:
                warnings = [SegmentationWarning(
                    code="face_partially_outside_image", message="Part of the face is outside the image.")]
        else:
            warnings = _warnings(pts, frame, parse, derived or [], (width, height), result.face_count, alignment)
            retake = ["glasses"] if any(w.code == "glasses_detected" for w in warnings) else []
        quality = ProfileQuality(alignment=alignment, warnings=warnings, retake_suggested=retake)

    # Features (optional)
    features_component: Component[FeaturesData] | None = None
    if feature_provider is not None:
        t0 = time.perf_counter()
        feature_info = feature_provider.info()
        if frame is None:
            features_component = _unavailable("features", feature_info, "depends on frame")
        else:
            glasses = any(w.code == "glasses_detected" for w in quality.warnings)
            try:
                fdata = feature_provider.classify(
                    image, pts, frame, full_zone_models if derived is not None else None, glasses
                )
            except Exception as exc:  # noqa: BLE001 - never leak details
                features_component = _unavailable("features", feature_info, type(exc).__name__)
            else:
                skin = fdata.skin_tone
                freason: str | None = None
                if skin is None:
                    freason = "Skin tone unavailable."
                elif skin.lighting_quality == "poor":
                    freason = "Lighting is poor."
                elif landmarks_component.meta.status == "degraded" or frame_component.meta.status == "degraded":
                    freason = "The face is small."
                confs: list[float] = []
                for c in (
                    fdata.eye_spacing,
                    fdata.eye_tilt,
                    fdata.eye_shape,
                    fdata.lip_fullness,
                    fdata.lip_balance,
                    fdata.lip_width,
                    fdata.brow_arch,
                    fdata.brow_thickness,
                    fdata.face_shape,
                    skin.depth if skin is not None else None,
                    skin.undertone if skin is not None else None,
                ):
                    if c is not None and c.label != "unknown":
                        confs.append(c.confidence)
                fconf = min(1.0, max(0.0, sum(confs) / len(confs))) if confs else None
                features_component = Component[FeaturesData](
                    meta=_meta("features", feature_info, "degraded" if freason else "ok", fconf, freason),
                    data=fdata,
                )
        timings["features"] = (time.perf_counter() - t0) * 1000

    timings["total"] = (time.perf_counter() - t_start) * 1000
    return FacialProfile(
        status="ok",
        message=result.message,
        image=result.image,
        components=ComponentSet(
            landmarks=landmarks_component,
            frame=frame_component,
            parsing=parsing_component,
            zones=zones_component,
            features=features_component,
        ),
        quality=quality,
        timings_ms={k: round(v, 2) for k, v in timings.items()},
    )
