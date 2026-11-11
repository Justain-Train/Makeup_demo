"""Orchestrates one segmentation request: landmarks -> parser -> zones -> response.

Keeps the HTTP layer thin. Uses canonical region names only; the parser module
is the only place that knows model class ids.
"""

import time

import numpy as np

from app.schemas.face_measurements import FacialLandmarkResponse
from app.schemas.face_segmentation import (
    AlignmentCheck,
    AnatomicalLayer,
    FaceSegmentationResponse,
    ModelInfo,
    ParseWindow as ParseWindowSchema,
    Point,
    RegionInfo,
    SegmentationWarning,
    Stage,
    Zone,
    ZoneBBox,
    ZoneMask,
)
from app.services.application_zones.catalog import STAGES
from app.services.application_zones.derive import DerivedZone, derive_zones
from app.services.application_zones.frame import FaceFrame
from app.services.facial_analysis.geometry import compute_basic_measurements, landmark_pixels
from app.services.facial_analysis.landmarks import FaceLandmarkService
from app.services.segmentation import masks as mk
from app.services.segmentation.face_parser import FaceParser, ParseResult
from app.services.segmentation.sides import assign_sides
from app.services.segmentation.regions import NOT_PARSED, REGION_SIDE, Region, region_name

MIN_ALIGNMENT_FRACTION = 0.6
HEAD_TILT_DEGREES = 25.0
LOW_RESOLUTION_IPD_PX = 40.0
GLASSES_MIN_AREA_FRACTION = 0.01
HAIR_FOREHEAD_FRACTION = 0.15
EDGE_TOLERANCE_PX = 2.0


def _warn(code: str, message: str) -> SegmentationWarning:
    return SegmentationWarning(code=code, message=message)  # type: ignore[arg-type]


def iris_checks(pts: np.ndarray, labels: np.ndarray, origin: tuple[int, int]) -> tuple[bool, bool]:
    """(right iris 468 in the right eye region, left iris 473 in the left eye region)."""
    h, w = labels.shape[:2]

    def near(p: np.ndarray, region: Region) -> bool:
        x, y = int(round(p[0])) - origin[0], int(round(p[1])) - origin[1]
        if not (0 <= x < w and 0 <= y < h):
            return False
        return bool((labels[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2] == int(region)).any())

    return near(pts[468], Region.EYE_R), near(pts[473], Region.EYE_L)


def alignment_check(pts: np.ndarray, labels: np.ndarray, origin: tuple[int, int]) -> AlignmentCheck:
    """Do the landmarks and the parser agree about where the face parts are?

    Guards against a left/right swap or a misplaced crop. Each check looks at a
    3x3 pixel neighbourhood so one-pixel rounding does not fail it.
    """
    h, w = labels.shape[:2]

    def region_at(p: np.ndarray) -> set[int]:
        x, y = int(round(p[0])) - origin[0], int(round(p[1])) - origin[1]
        if not (0 <= x < w and 0 <= y < h):
            return set()
        return {int(v) for v in np.unique(labels[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2])}

    def hit(p: np.ndarray, allowed: set[Region]) -> bool:
        return bool(region_at(p) & {int(r) for r in allowed})

    right_iris_ok, left_iris_ok = iris_checks(pts, labels, origin)
    checks = [
        right_iris_ok,
        left_iris_ok,
        hit(pts[1], {Region.NOSE, Region.SKIN}),  # nose tip
        hit(pts[13], {Region.MOUTH_INTERIOR, Region.LIP_UPPER, Region.LIP_LOWER}),  # mouth centre
        hit((pts[105] + pts[52]) / 2, {Region.BROW_R}),  # right brow middle
        hit((pts[334] + pts[282]) / 2, {Region.BROW_L}),  # left brow middle
    ]
    fraction = sum(checks) / len(checks)
    # Both irises inside an eye region is a HARD requirement: a one-iris failure
    # (e.g. a mask covering both eyes) must not be hidden by the other checks.
    passed = right_iris_ok and left_iris_ok and fraction >= MIN_ALIGNMENT_FRACTION
    return AlignmentCheck(passed=passed, fraction=round(fraction, 3))


def _region_infos(parse: ParseResult) -> list[RegionInfo]:
    labels = parse.labels.ravel()
    counts = np.bincount(labels, minlength=256)
    prob_sums = np.bincount(labels, weights=parse.confidence.ravel(), minlength=256)
    total = labels.size
    infos = []
    for region in Region:
        n = int(counts[int(region)])
        infos.append(
            RegionInfo(
                id=int(region),
                name=region_name(region),
                side=REGION_SIDE.get(region),
                pixel_count=n,
                area_fraction=n / total if total else 0.0,
                mean_probability=float(prob_sums[int(region)] / n) if n else None,
                present=n > 0,
            )
        )
    return infos


def _encode_zone(z: DerivedZone, origin: tuple[int, int]) -> Zone:
    mask = bbox = polygon = None
    if z.mask is not None and z.bbox is not None and z.status != "unavailable":
        x, y, w, h = z.bbox
        crop = z.mask[y - origin[1] : y - origin[1] + h, x - origin[0] : x - origin[0] + w]
        mask = ZoneMask(png_base64=mk.encode_png_base64(crop))
        bbox = ZoneBBox(x=x, y=y, width=w, height=h)
    if z.polygon is not None:
        polygon = [(round(float(px), 1), round(float(py), 1)) for px, py in z.polygon]
    return Zone(
        id=z.spec.id,
        name=z.spec.name,
        group=z.spec.group,
        side=z.spec.side,
        sources=list(z.spec.sources),
        status=z.status,
        reason=z.reason,
        confidence=None if z.confidence is None else round(z.confidence, 4),
        coverage=None if z.coverage is None else round(z.coverage, 4),
        clipped_to=list(z.spec.clipped_to),
        bbox=bbox,
        pixel_count=z.pixel_count if mask is not None else 0,
        mask=mask,
        polygon=polygon,
    )


def _warnings(
    pts: np.ndarray,
    frame: FaceFrame,
    parse: ParseResult,
    zones: list[DerivedZone],
    image_size: tuple[int, int],
    face_count: int,
    alignment: AlignmentCheck,
) -> list[SegmentationWarning]:
    w, h = image_size
    out: list[SegmentationWarning] = []
    if face_count > 1:
        out.append(_warn("background_face_ignored",
                         "A smaller second face was found and ignored; the largest face was analyzed."))
    if (pts[:, 0].min() < -EDGE_TOLERANCE_PX or pts[:, 1].min() < -EDGE_TOLERANCE_PX
            or pts[:, 0].max() > w + EDGE_TOLERANCE_PX or pts[:, 1].max() > h + EDGE_TOLERANCE_PX):
        out.append(_warn("face_partially_outside_image", "Part of the face is outside the image."))
    if frame.ipd < LOW_RESOLUTION_IPD_PX:
        out.append(_warn("low_resolution_face",
                         f"The face is small ({frame.ipd:.0f}px between the eyes); masks will be coarse."))
    if abs(frame.roll_degrees) > HEAD_TILT_DEGREES:
        out.append(_warn("head_tilt", f"Head is tilted {frame.roll_degrees:.0f} degrees; results may be less accurate."))
    glasses = int(np.count_nonzero(parse.labels == int(Region.GLASSES)))
    if glasses >= GLASSES_MIN_AREA_FRACTION * parse.labels.size:
        out.append(_warn("glasses_detected", "Glasses detected; eye and under-eye zones may be unreliable."))
    if not alignment.passed:
        out.append(_warn("parser_landmark_mismatch",
                         "The parsed regions do not line up with the facial landmarks."))
    for z in zones:
        if z.coverage is not None and z.coverage < 0.5 and z.status == "low_confidence":
            out.append(_warn("low_overlap_zone", f"Zone '{z.spec.id}' overlaps little usable skin ({z.coverage:.0%})."))
        if z.spec.id == "forehead" and z.extra.get("hair_fraction", 0.0) >= HAIR_FOREHEAD_FRACTION:
            out.append(_warn("hair_occludes_forehead",
                             f"Hair covers about {z.extra['hair_fraction']:.0%} of the expected forehead."))
    return out


def segment_face(
    image: np.ndarray,
    landmark_service: FaceLandmarkService,
    parser: FaceParser,
) -> FaceSegmentationResponse:
    """Run landmarks + parsing + zone derivation on one decoded RGB image.

    Raises FaceParserError subclasses (busy / inference failure).
    """
    t_start = time.perf_counter()
    height, width = image.shape[:2]
    result = landmark_service.detect_landmarks(image)
    t_landmarks = time.perf_counter()
    landmarks = FacialLandmarkResponse(**result.model_dump(), measurements=compute_basic_measurements(result))
    model = ModelInfo(
        id=parser.info.id,
        architecture=parser.info.architecture,
        weights_sha256=parser.info.weights_sha256,
        input_size=parser.info.input_size,
    )
    base = dict(
        status=result.status,
        message=result.message,
        image=result.image,
        model=model,
        landmarks=landmarks,
    )
    if result.status != "ok":
        # No face / multiple faces: valid request, nothing to segment.
        return FaceSegmentationResponse(
            **base, timings_ms={"landmarks": (t_landmarks - t_start) * 1000, "total": (t_landmarks - t_start) * 1000}
        )

    lm = landmark_pixels(result)
    pts = np.array([lm[i] for i in range(len(lm))], dtype=np.float64)
    bbox = (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))

    parse = parser.parse(image, bbox)
    # Give the side-agnostic paired regions a side from the landmark geometry.
    parse.labels = assign_sides(parse.labels, pts, parse.origin)

    t0 = time.perf_counter()
    frame = FaceFrame.from_landmarks(pts)
    zones = derive_zones(frame, parse.labels, parse.confidence, parse.origin)
    alignment = alignment_check(pts, parse.labels, parse.origin)
    for side, ok in zip("rl", iris_checks(pts, parse.labels, parse.origin)):
        if not ok:  # that side's eye mask does not contain its iris: do not report it as ok
            for z in zones:
                if z.spec.group == "eye" and z.spec.id.endswith(f"_{side}") and z.status == "ok":
                    z.status = "low_confidence"
                    z.reason = "The parsed eye does not contain this side's iris landmark (parser/landmark mismatch)."
    warnings = _warnings(pts, frame, parse, zones, (width, height), result.face_count, alignment)
    retake = ["glasses"] if any(w.code == "glasses_detected" for w in warnings) else []
    t1 = time.perf_counter()

    anatomical = AnatomicalLayer(
        origin=Point(x=parse.origin[0], y=parse.origin[1]),
        width=int(parse.labels.shape[1]),
        height=int(parse.labels.shape[0]),
        not_parsed_value=NOT_PARSED,
        label_map_png_base64=mk.encode_png_base64(parse.labels),
        regions=_region_infos(parse),
    )
    zone_models = [_encode_zone(z, parse.origin) for z in zones]
    t2 = time.perf_counter()

    timings = {"landmarks": (t_landmarks - t_start) * 1000, **parse.timings_ms}
    timings.update({"zones": (t1 - t0) * 1000, "encode": (t2 - t1) * 1000, "total": (t2 - t_start) * 1000})
    return FaceSegmentationResponse(
        **base,
        parse_window=ParseWindowSchema(x=parse.window.x, y=parse.window.y, size=parse.window.size),
        anatomical=anatomical,
        zones=zone_models,
        stages=[Stage(number=st.number, key=st.key, name=st.name, layers={k: list(v) for k, v in st.layers.items()}) for st in STAGES],
        alignment_check=alignment,
        warnings=warnings,
        retake_suggested=retake,
        timings_ms={k: round(v, 2) for k, v in timings.items()},
    )
