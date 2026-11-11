"""Layer B: derive makeup-application zones.

Inputs are the anatomical label map (canonical region ids, Layer A) and the
MediaPipe landmarks. Outputs are masks in the label map's pixel grid.

Zone geometry is expressed in `FaceFrame` units (multiples of the
interpupillary distance), so it scales with face size and follows head roll.
Landmark polygons are always clipped to the anatomical mask they belong to
(skin for face zones), so a zone can never cover eyes, lips or hair that the
parser found. This module uses canonical region names only.

The ipd offsets (0.35, 0.15, 0.3, 0.12, 0.18, 0.22) are approximations from the
approved plan, tuned by eye on one photo. They are not measured accuracy.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from app.services.application_zones.catalog import ZONE_SPEC_BY_ID, ZoneSpec
from app.services.application_zones.frame import FaceFrame
from app.services.segmentation import masks as mk
from app.services.segmentation.regions import Region

ZoneStatus = Literal["ok", "low_confidence", "unavailable"]

MIN_ZONE_PIXELS = 16  # smaller masks are not usable for application
MIN_COVERAGE = 0.5  # clipped area / in-image geometric area
# Agreement thresholds between a landmark polygon and the parser mask (IoU).
MAX_OTHER_SIDE_FRACTION = 0.20
MAX_WIDTH_RATIO = 1.5
MIN_IOU = {"eye": 0.30, "brow": 0.20, "lip": 0.40, "nose": 0.30}

# Contour index sets for the subject's RIGHT side (checked visually and against
# mediapipe's FaceLandmarksConnections in the tests).
EYE_UPPER = [33, 246, 161, 160, 159, 158, 157, 173, 133]
EYE_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]
BROW_UPPER = [70, 63, 105, 66, 107]
BROW_LOWER = [46, 53, 52, 65, 55]
LIP_OUTER_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIP_OUTER_LOWER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LIP_INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIP_INNER_LOWER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
FOREHEAD_ARC = [162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389]
CHIN_ARC = [136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365]
JAW_PATH = [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361]
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
        377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
# Outline order (bridge top, subject-left wing, tip base, right wing); the old criss-cross order made a self-intersecting polygon. Fixed in Batch 5; changes nose, cheek_l and mouth_surround outputs.
NOSE_HULL = [168, 358, 327, 2, 98, 129]
CUPID_BOW_PATH = [40, 39, 37, 0, 267, 269, 270]  # contiguous upper-lip peaks, subject's right to left
UNDER_LIP_PATH = [84, 17, 314]  # lower outer lip contour
LIP_CORNER_INDEX = 61  # subject's right mouth corner; the left one is its mirror (291)

# Batch 1 zone geometry, in ipd units (approximations, not measured).
CHIN_CENTER_HALF_WIDTH = 0.12
CHIN_CENTER_TOP_DV = 0.25  # below landmark 17
UNDER_LIP_NEAR_DV = 0.10
UNDER_LIP_FAR_DV = 0.25
MOUTH_SURROUND_DILATION = 0.20
CUPID_BOW_STROKE = 0.04
LIP_CORNER_RADIUS = 0.12
LIP_CORNER_SEGMENTS = 12
LIP_CORNER_TOLERANCE = 0.05  # boundary tolerance, same as the Phase 3 evaluation (0.05 * IOD)
MIN_LIP_CORNER_COVERAGE = 0.6
LIP_CORNER_DILATION_PX = 1
MIN_STROKE_PX = 2.0

# Batch 2 zone geometry, in ipd units. Untuned approximations, not measured.
# These zones have no parser positive mask; they are landmark geometry clipped to
# parser skin and validated only by the coverage rule (MIN_COVERAGE).
FOREHEAD_CENTER_HALF_WIDTH = 0.25
FOREHEAD_CENTER_BOTTOM_DV = -0.15  # relative to landmark 9 (between the brows)
FOREHEAD_PERIMETER_BAND = 0.12
TEMPLE_BROW_DU = -0.05  # lateral of the brow tail (70)
TEMPLE_BROW_DV = -0.05
TEMPLE_EYE_DU = -0.15  # lateral of the outer eye corner (33)
TEMPLE_EYE_DV = 0.05
JAW_OUTER_PATH_R = [132, 58, 172, 136, 150, 149, 176, 148, 152]  # subject's right half
JAW_OUTER_BAND = 0.07

# Batch 3a zone geometry, in ipd units. Untuned approximations, not measured.
EYE_CORNER_RADIUS = 0.10
EYE_CORNER_TOLERANCE = 0.05  # boundary tolerance, same as the Phase 3 evaluation (0.05 * IOD)
MIN_EYE_CORNER_COVERAGE = 0.6
EYE_CORNER_SEGMENTS = 12
LASH_STROKE = 0.03
GLASSES_ZONE_FRACTION = 0.20

# Batch 3b zone geometry, in ipd units or fractions of the brow length. Untuned approximations.
PARTS_ORDER = ("head", "arch", "tail")
BROW_HEAD_END = 0.35
BROW_TAIL_START = 0.70
BROW_PART_TOLERANCE = 0.05
MIN_BROW_PART_COVERAGE = 0.6
EYELID_HEIGHT = 0.18
LOWER_EYELID_HEIGHT = 0.07
BROW_BONE_HEIGHT = 0.12
BROW_BONE_LM = [46, 53, 52, 65]  # lower brow contour, subject's right

# Batch 4 cheek zone geometry: (landmark, du, dv) in ipd units for the subject's right side
# (+du is medial, +dv is down; the left side mirrors). Untuned approximations, not measured.
CHEEKBONE_SPEC = [(33, 0.12, 0.32), (234, 0.05, 0.0), (234, 0.05, 0.20), (33, 0.12, 0.50)]
CONTOUR_HOLLOW_SPEC = [(33, 0.12, 0.56), (234, 0.05, 0.24), (93, 0.05, 0.0), (33, 0.12, 0.84)]
CHEEK_APPLE_SPEC = [(33, 0.22, 0.46), (33, -0.10, 0.52), (33, -0.12, 0.78), (33, 0.20, 0.85)]
CHEEK_UPPER_SPEC = [(33, 0.22, 0.30), (33, -0.12, 0.32), (33, -0.12, 0.52), (33, 0.22, 0.48)]
CHEEK_OUTER_SPEC = [(33, -0.10, 0.50), (234, 0.05, 0.0), (93, 0.05, 0.0), (33, -0.10, 0.82)]
CHEEK_EYE_DILATION_PX = 1

# Batch 5 nose zone geometry: (landmark, du, dv) in ipd units for the subject's right side
# (+du is medial, +dv is down; the left side mirrors). Untuned approximations, not measured.
NOSE_BRIDGE_SPEC = [(168, -0.07, 0.05), (168, 0.07, 0.05), (5, 0.07, -0.03), (5, -0.07, -0.03)]
NOSE_TIP_SPEC = [(5, -0.09, 0.02), (5, 0.09, 0.02), (2, 0.09, -0.02), (2, -0.09, -0.02)]
NOSE_SIDE_SPEC = [(168, -0.07, 0.10), (133, 0.02, 0.25), (129, 0.02, -0.05), (98, 0.0, -0.02), (5, -0.09, 0.0)]
NOSTRIL_CENTRE = (98, 0.05, -0.04)
NOSTRIL_RADIUS_U = 0.05
NOSTRIL_RADIUS_V = 0.035


@dataclass
class DerivedZone:
    spec: ZoneSpec
    status: ZoneStatus
    reason: str | None
    confidence: float | None
    mask: np.ndarray | None  # bool, label-map grid (region-local); None if unavailable
    polygon: np.ndarray | None  # (N, 2) image pixels, landmark zones only
    coverage: float | None = None
    pixel_count: int = 0
    bbox: tuple[int, int, int, int] | None = None  # image pixels (x, y, w, h)
    extra: dict[str, float] = field(default_factory=dict)


class ZoneDeriver:
    """Builds all application zones (initial 18 plus Phase 4 batches 1 to 5) for one face."""

    def __init__(
        self,
        frame: FaceFrame,
        labels: np.ndarray,
        confidence: np.ndarray,
        origin: tuple[int, int],
    ) -> None:
        self.f = frame
        self.labels = labels
        self.conf = confidence
        self.origin = np.array(origin, dtype=np.float64)
        self.shape = labels.shape[:2]
        self._zones: dict[str, DerivedZone] = {}
        self._face_skin: np.ndarray | None = None

    # -- helpers ------------------------------------------------------------
    def region(self, *regions: Region) -> np.ndarray:
        return np.isin(self.labels, [int(r) for r in regions])

    def raster(self, polygon: np.ndarray) -> np.ndarray:
        return mk.rasterize_polygon(polygon - self.origin, self.shape)

    def _mean_conf(self, mask: np.ndarray) -> float | None:
        return float(self.conf[mask].mean()) if mask.any() else None

    def _bbox_image(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        b = mk.bbox_of(mask)
        if b is None:
            return None
        return b[0] + int(self.origin[0]), b[1] + int(self.origin[1]), b[2], b[3]

    def _unavailable(self, spec: ZoneSpec, reason: str, polygon: np.ndarray | None = None) -> DerivedZone:
        return DerivedZone(spec, "unavailable", reason, None, None, polygon)

    def _finish(
        self,
        spec: ZoneSpec,
        mask: np.ndarray,
        *,
        polygon: np.ndarray | None = None,
        geometric: np.ndarray | None = None,
        confidence: float | None = None,
        geometric_only: bool = False,
        agreement: tuple[float, float, str] | None = None,
    ) -> DerivedZone:
        """Apply the validation rules: minimum size, coverage, agreement.

        geometric: the unclipped in-image zone mask (for coverage).
        agreement: (iou, min_iou, label) between a landmark polygon and a parser mask.
        geometric_only: purely landmark-derived zone, so no confidence is reported.
        """
        n = int(np.count_nonzero(mask))
        if n < MIN_ZONE_PIXELS:
            why = "Zone is outside the parsed area or smaller than 16 pixels."
            if geometric is not None and np.count_nonzero(geometric) >= MIN_ZONE_PIXELS:
                why = "No skin available here (covered by glasses, hair or another region)."
            if agreement is not None:
                why = "Parser found no usable pixels for this region."
            return self._unavailable(spec, why, polygon)
        status: ZoneStatus = "ok"
        reason: str | None = None
        coverage = None
        if geometric is not None:
            g = int(np.count_nonzero(geometric))
            coverage = n / g if g else 0.0
            if coverage < MIN_COVERAGE:
                status, reason = "low_confidence", (
                    f"Only {coverage:.0%} of the expected area is available skin "
                    "(occlusion by hair, glasses or the image edge)."
                )
        if agreement is not None:
            iou, min_iou, label = agreement
            confidence = iou
            if iou < min_iou:
                status = "low_confidence"
                reason = f"Parser mask and landmark {label} disagree (IoU {iou:.2f})."
        elif confidence is None and not geometric_only:
            confidence = self._mean_conf(mask)
        return DerivedZone(
            spec, status, reason, confidence, mask, polygon,
            coverage=coverage, pixel_count=n, bbox=self._bbox_image(mask),
        )

    # -- zones --------------------------------------------------------------
    def derive(self) -> list[DerivedZone]:
        # Landmark eye/lip polygons are used to carve the skin mask.
        eye_poly = {s: self._eye_polygon(s) for s in ("right", "left")}
        lip_outer = self._lip_outer_polygon()
        carve = self.raster(lip_outer)
        for s in ("right", "left"):
            carve |= self.raster(eye_poly[s])
        carve = mk.dilate(carve, 1)

        skin = self.region(Region.SKIN, Region.NOSE) & ~carve
        self._face_skin = mk.largest_component(skin)
        self._add(self._finish(ZONE_SPEC_BY_ID["face_skin"], self._face_skin))
        # Face zones are clipped to the parser's `skin` class (so they exclude
        # the nose class, which has its own zone) inside face_skin (so they also
        # exclude landmark eye/lip areas the parser may have mislabelled).
        clip = self._face_skin & self.region(Region.SKIN)

        self._add(self._forehead(clip))
        self._add(self._forehead_center(clip))
        self._add(self._forehead_perimeter(clip))
        self._add(self._chin(clip))
        self._add(self._chin_center(clip))
        self._add(self._under_lower_lip(clip))
        self._add(self._mouth_surround(clip, lip_outer))
        self._add(self._jawline(clip))
        for s, k in (("right", "r"), ("left", "l")):
            self._add(self._cheek(clip, s, k))
            self._add(self._under_eye(clip, s, k, eye_poly[s]))
            self._add(self._temple(clip, s, k))
            self._add(self._jawline_outer(clip, s, k))
            other = "left" if s == "right" else "right"
            self._add(self._model_zone(f"eye_{k}", Region.EYE_R if s == "right" else Region.EYE_L,
                                       eye_poly[s], "eye", eye_poly[other]))
            self._add(self._model_zone(f"brow_{k}", Region.BROW_R if s == "right" else Region.BROW_L,
                                       self._brow_polygon(s), "brow", self._brow_polygon(other)))
        for s, k in (("right", "r"), ("left", "l")):
            self._eye_zones(clip, s, k, eye_poly[s])
            self._brow_eyelid_zones(clip, s, k, eye_poly[s])
            self._cheek_zones(clip, s, k, eye_poly[s], lip_outer)
        self._add(self._nose())
        self._nose_zones()
        self._lips()
        order = list(ZONE_SPEC_BY_ID)
        return sorted(self._zones.values(), key=lambda z: order.index(z.spec.id))

    def _add(self, z: DerivedZone) -> None:
        self._zones[z.spec.id] = z

    def _eye_polygon(self, side: str) -> np.ndarray:
        f = self.f
        idx = f.indices(EYE_UPPER + EYE_LOWER[::-1][1:-1], side)
        return f.pts[idx]

    def _brow_polygon(self, side: str) -> np.ndarray:
        f = self.f
        return f.pts[f.indices(BROW_UPPER + BROW_LOWER[::-1], side)]

    def _lip_outer_polygon(self) -> np.ndarray:
        return self.f.pts[LIP_OUTER_UPPER + LIP_OUTER_LOWER[::-1][1:-1]]

    def _forehead_parts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        f = self.f
        brow_mid = (f.pts[105] + f.pts[334]) / 2.0
        d = abs(float(np.dot(f.pts[10] - brow_mid, f.v)))  # brow -> forehead-top distance
        arc = f.pts[FOREHEAD_ARC] - 0.35 * d * f.v
        # Close along both brows' upper contours (shifted up), left side first
        # because the arc ends at the subject's left temple (389).
        left_brow = f.polygon([(i, 0, -0.15) for i in [300, 293, 334, 296, 336]], "right")
        right_brow = f.polygon([(i, 0, -0.15) for i in [107, 66, 105, 63, 70]], "right")
        return arc, left_brow, right_brow, d

    def _forehead_polygon(self) -> np.ndarray:
        arc, left_brow, right_brow, _ = self._forehead_parts()
        return np.vstack([arc, left_brow, right_brow])

    def _set_hair_fraction(self, zone: DerivedZone, geometric: np.ndarray) -> None:
        g = int(np.count_nonzero(geometric))
        if g:
            zone.extra["hair_fraction"] = float(np.count_nonzero(geometric & self.region(Region.HAIR)) / g)

    def _forehead_center(self, clip: np.ndarray) -> DerivedZone:
        """Central forehead column. No parser positive mask: validated by coverage only."""
        f = self.f
        forehead_geom = self.raster(self._forehead_polygon())
        _, _, _, d = self._forehead_parts()
        w, top, bottom = FOREHEAD_CENTER_HALF_WIDTH, -0.35 * d / f.ipd, FOREHEAD_CENTER_BOTTOM_DV
        quad = f.polygon([(10, -w, top), (10, w, top), (9, w, bottom), (9, -w, bottom)])
        geometric = self.raster(quad) & forehead_geom
        zone = self._finish(ZONE_SPEC_BY_ID["forehead_center"], geometric & clip, polygon=quad, geometric=geometric)
        self._set_hair_fraction(zone, geometric)
        return zone

    def _forehead_perimeter(self, clip: np.ndarray) -> DerivedZone:
        """Rim of the forehead polygon along the hairline arc. No parser positive mask."""
        f = self.f
        arc, left_brow, right_brow, _ = self._forehead_parts()
        forehead_poly = np.vstack([arc, left_brow, right_brow])
        inner = np.vstack([arc + FOREHEAD_PERIMETER_BAND * f.ipd * f.v, left_brow, right_brow])
        geometric = self.raster(forehead_poly) & ~self.raster(inner)
        zone = self._finish(ZONE_SPEC_BY_ID["forehead_perimeter"], geometric & clip,
                            polygon=forehead_poly, geometric=geometric)
        self._set_hair_fraction(zone, geometric)
        return zone

    def _temple(self, clip: np.ndarray, side: str, k: str) -> DerivedZone:
        """Temple, lateral to the outer eye corner. No parser positive mask."""
        poly = self.f.polygon([
            (70, TEMPLE_BROW_DU, TEMPLE_BROW_DV), (162, 0, 0), (127, 0, 0), (33, TEMPLE_EYE_DU, TEMPLE_EYE_DV),
        ], side)
        geometric = self.raster(poly)
        return self._finish(ZONE_SPEC_BY_ID[f"temple_{k}"], geometric & clip, polygon=poly, geometric=geometric)

    def _jawline_outer(self, clip: np.ndarray, side: str, k: str) -> DerivedZone:
        """Thin outermost strip of the jaw path. No parser positive mask."""
        f = self.f
        outer = f.pts[f.indices(JAW_OUTER_PATH_R, side)]
        inward = f.pts[OVAL].mean(axis=0) - outer
        inward /= np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-6)
        inner = outer + JAW_OUTER_BAND * f.ipd * inward
        poly = np.vstack([outer, inner[::-1]])
        geometric = self.raster(poly)
        return self._finish(ZONE_SPEC_BY_ID[f"jawline_outer_{k}"], geometric & clip,
                            polygon=poly, geometric=geometric)

    def _forehead(self, clip: np.ndarray) -> DerivedZone:
        poly = self._forehead_polygon()
        geometric = self.raster(poly)
        zone = self._finish(ZONE_SPEC_BY_ID["forehead"], geometric & clip, polygon=poly, geometric=geometric)
        g = int(np.count_nonzero(geometric))
        if g:  # share of the expected forehead that the parser calls hair (bangs, hairline)
            zone.extra["hair_fraction"] = float(np.count_nonzero(geometric & self.region(Region.HAIR)) / g)
        return zone

    def _cheek(self, clip: np.ndarray, side: str, k: str) -> DerivedZone:
        f = self.f
        spec = [(133, 0, 0.3), (33, 0, 0.3), (234, 0, 0), (93, 0, 0), (58, 0, 0), (61, 0, 0), (129, 0, 0)]
        poly = f.polygon(spec, side)
        geometric = self.raster(poly)
        return self._finish(ZONE_SPEC_BY_ID[f"cheek_{k}"], geometric & clip, polygon=poly, geometric=geometric)

    def _chin(self, clip: np.ndarray) -> DerivedZone:
        f = self.f
        poly = np.vstack([f.pts[CHIN_ARC], f.pts[[291]], f.polygon([(17, 0, 0.12)]), f.pts[[61]]])
        geometric = self.raster(poly)
        return self._finish(ZONE_SPEC_BY_ID["chin"], geometric & clip, polygon=poly, geometric=geometric)

    def _chin_center(self, clip: np.ndarray) -> DerivedZone:
        f = self.f
        w, top = CHIN_CENTER_HALF_WIDTH, CHIN_CENTER_TOP_DV
        poly = f.polygon([(17, -w, top), (17, w, top), (152, w, 0.0), (152, -w, 0.0)])
        geometric = self.raster(poly)
        return self._finish(ZONE_SPEC_BY_ID["chin_center"], geometric & clip, polygon=poly, geometric=geometric)

    def _under_lower_lip(self, clip: np.ndarray) -> DerivedZone:
        f = self.f
        near = f.polygon([(i, 0, UNDER_LIP_NEAR_DV) for i in UNDER_LIP_PATH])
        far = f.polygon([(i, 0, UNDER_LIP_FAR_DV) for i in UNDER_LIP_PATH])
        poly = np.vstack([near, far[::-1]])
        geometric = self.raster(poly)
        return self._finish(ZONE_SPEC_BY_ID["under_lower_lip"], geometric & clip, polygon=poly, geometric=geometric)

    def _mouth_surround(self, clip: np.ndarray, lip_outer: np.ndarray) -> DerivedZone:
        outer = self.raster(lip_outer)
        geometric = mk.dilate(outer, MOUTH_SURROUND_DILATION * self.f.ipd) & ~outer
        return self._finish(ZONE_SPEC_BY_ID["mouth_surround"], geometric & clip,
                            polygon=lip_outer, geometric=geometric)

    def _lip_corner(self, side: str, k: str, lip_full: np.ndarray, model: np.ndarray) -> DerivedZone:
        f = self.f
        centre = f.point(LIP_CORNER_INDEX, side=side)
        ang = np.linspace(0.0, 2.0 * np.pi, LIP_CORNER_SEGMENTS, endpoint=False)
        poly = centre + LIP_CORNER_RADIUS * f.ipd * (np.cos(ang)[:, None] * f.u + np.sin(ang)[:, None] * f.v)
        disc = self.raster(poly)
        mask = disc & mk.dilate(lip_full, LIP_CORNER_DILATION_PX)
        near_model = mk.dilate(model, LIP_CORNER_TOLERANCE * f.ipd)
        n = int(np.count_nonzero(mask))
        coverage = float(np.count_nonzero(mask & near_model)) / n if n else 0.0
        z = self._finish(ZONE_SPEC_BY_ID[f"lip_corner_{k}"], mask, polygon=poly,
                         agreement=(coverage, MIN_LIP_CORNER_COVERAGE, "parser lip within tolerance"))
        if z.status == "low_confidence":
            z.reason = (f"Only {coverage:.0%} of the corner lies within {LIP_CORNER_TOLERANCE} ipd "
                        "of the parser lip mask (coverage within a tolerance, not IoU).")
        if z.status == "unavailable":
            z.reason = "Lip polygon is outside the parsed area."
        return z

    def _jawline(self, clip: np.ndarray) -> DerivedZone:
        f = self.f
        outer = f.pts[JAW_PATH]
        centre = f.pts[OVAL].mean(axis=0)
        inward = centre - outer
        inward /= np.linalg.norm(inward, axis=1, keepdims=True)
        inner = outer + 0.18 * f.ipd * inward
        poly = np.vstack([outer, inner[::-1]])
        geometric = self.raster(poly)
        return self._finish(ZONE_SPEC_BY_ID["jawline"], geometric & clip, polygon=poly, geometric=geometric)

    def _under_eye_geometric(self, side: str, eye_poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = self.f
        lid = f.polygon([(i, 0, 0) for i in EYE_LOWER], side)
        below = f.polygon([(i, 0, 0.22) for i in EYE_LOWER], side)
        poly = np.vstack([lid, below[::-1]])
        geometric = self.raster(poly) & ~self.raster(eye_poly)
        return poly, geometric

    def _window(self, mask: np.ndarray) -> tuple[slice, slice, np.ndarray, np.ndarray] | None:
        """Bounding window of `mask` as (row slice, col slice, row index grid, col index grid)."""
        box = mk.bbox_of(mask)
        if box is None:
            return None
        x, y, w, h = box
        ys, xs = slice(y, y + h), slice(x, x + w)
        yy, xx = np.indices((h, w))
        return ys, xs, yy + y, xx + x

    def _disc_polygon(self, centre: np.ndarray, radius_ipd: float) -> np.ndarray:
        f = self.f
        ang = np.linspace(0.0, 2.0 * np.pi, EYE_CORNER_SEGMENTS, endpoint=False)
        return centre + radius_ipd * f.ipd * (np.cos(ang)[:, None] * f.u + np.sin(ang)[:, None] * f.v)

    def _eye_area_guard(self, zone: DerivedZone, k: str, geometric: np.ndarray,
                        parent: str | None = None) -> DerivedZone:
        """Demote only: never turns a weaker status into a stronger one.

        parent: id of the zone whose status gates this one (default: the eye on this side).
        """
        if zone.status != "ok":
            return zone
        parent_id = parent or f"eye_{k}"
        g = int(np.count_nonzero(geometric))
        if g and np.count_nonzero(geometric & self.region(Region.GLASSES)) / g >= GLASSES_ZONE_FRACTION:
            zone.status = "low_confidence"
            zone.reason = "Glasses cover part of this eye zone; nothing is inferred behind the lenses."
        elif self._zones[parent_id].status != "ok":
            what = "brow" if parent_id.startswith("brow") else "nose" if parent_id.startswith("nose") else "eye"
            zone.status = "low_confidence"
            zone.reason = f"The {what} is not parsed or disagrees with the landmarks."
        return zone

    def _brow_eyelid_zones(self, clip: np.ndarray, side: str, k: str, eye_poly: np.ndarray) -> None:
        """Batch 3b for one side: brow head/arch/tail, upper eyelid, brow bone.

        Brow parts split the landmark brow polygon along its head-to-tail axis and are
        checked by coverage within a tolerance of the parser brow mask (not intersected
        with it). Eyelid and brow bone are landmark geometry clipped to skin, validated by
        the coverage rule only (no parser positive mask).
        """
        f = self.f
        brow_poly = self._brow_polygon(side)
        brow_m = self.raster(brow_poly)

        i_head, i_tail = f.indices([107, 70], side)
        head_pt, tail_pt = f.pts[i_head], f.pts[i_tail]
        axis = (tail_pt - head_pt) / max(float(np.linalg.norm(tail_pt - head_pt)), 1e-6)
        sels = {n: np.zeros(self.shape, dtype=bool) for n in PARTS_ORDER}
        win = self._window(brow_m)
        if win is not None:
            ys, xs, yy, xx = win
            proj = (xx + self.origin[0] - head_pt[0]) * axis[0] + (yy + self.origin[1] - head_pt[1]) * axis[1]
            lo, hi = float(proj[brow_m[ys, xs]].min()), float(proj[brow_m[ys, xs]].max())
            s = (proj - lo) / max(hi - lo, 1e-6)
            sels["head"][ys, xs] = s < BROW_HEAD_END
            sels["arch"][ys, xs] = (s >= BROW_HEAD_END) & (s < BROW_TAIL_START)
            sels["tail"][ys, xs] = s >= BROW_TAIL_START
        near_model = mk.dilate(self.region(Region.BROW_R if side == "right" else Region.BROW_L),
                               BROW_PART_TOLERANCE * f.ipd)
        for name, sel in sels.items():
            mask = brow_m & sel
            n = int(np.count_nonzero(mask))
            coverage = float(np.count_nonzero(mask & near_model)) / n if n else 0.0
            z = self._finish(ZONE_SPEC_BY_ID[f"brow_{name}_{k}"], mask, polygon=brow_poly,
                             agreement=(coverage, MIN_BROW_PART_COVERAGE, "parser brow within tolerance"))
            if z.status == "low_confidence":
                z.reason = (f"Only {coverage:.0%} of the brow part lies within {BROW_PART_TOLERANCE} ipd "
                            "of the parser brow mask (coverage within a tolerance, not IoU).")
            self._add(self._eye_area_guard(z, k, mask, parent=f"brow_{k}"))

        upper = f.pts[f.indices(EYE_UPPER, side)]
        lid_poly = np.vstack([upper, (upper - EYELID_HEIGHT * f.ipd * f.v)[::-1]])
        lid_geom = self.raster(lid_poly) & ~mk.dilate(self.raster(eye_poly), 1) & ~brow_m
        z = self._finish(ZONE_SPEC_BY_ID[f"eyelid_upper_{k}"], lid_geom & clip, polygon=lid_poly, geometric=lid_geom)
        self._add(self._eye_area_guard(z, k, lid_geom, parent=f"eye_{k}"))

        lm = f.pts[f.indices(BROW_BONE_LM, side)]
        bone_poly = np.vstack([lm, (lm + BROW_BONE_HEIGHT * f.ipd * f.v)[::-1]])
        bone_geom = self.raster(bone_poly) & ~brow_m & ~self.raster(eye_poly) & ~lid_geom
        z = self._finish(ZONE_SPEC_BY_ID[f"brow_bone_{k}"], bone_geom & clip, polygon=bone_poly, geometric=bone_geom)
        self._add(self._eye_area_guard(z, k, bone_geom, parent=f"brow_{k}"))

    def _cheek_zones(self, clip: np.ndarray, side: str, k: str, eye_poly: np.ndarray,
                     lip_outer: np.ndarray) -> None:
        """Batch 4 for one side: cheek apple, upper cheek, outer cheek, cheekbone, contour hollow.

        Landmark geometry clipped to skin, validated by the coverage rule only (no parser
        positive mask). Eye, under-eye, nose, lips and brows are subtracted; the cheekbone
        and hollow are made disjoint.
        """
        f = self.f
        eye_r, brow_r = (Region.EYE_R, Region.BROW_R) if side == "right" else (Region.EYE_L, Region.BROW_L)
        _, under_eye = self._under_eye_geometric(side, eye_poly)
        excluded = (
            mk.dilate(self.raster(eye_poly), CHEEK_EYE_DILATION_PX) | self.region(eye_r)
            | under_eye | self.raster(f.pts[NOSE_HULL]) | self.region(Region.NOSE)
            | self.raster(lip_outer) | self.raster(self._brow_polygon(side)) | self.region(brow_r)
        )
        polys = {
            name: f.polygon(spec, side) for name, spec in (
                ("cheekbone", CHEEKBONE_SPEC), ("contour_hollow", CONTOUR_HOLLOW_SPEC),
                ("cheek_apple", CHEEK_APPLE_SPEC), ("cheek_upper", CHEEK_UPPER_SPEC),
                ("cheek_outer", CHEEK_OUTER_SPEC),
            )
        }
        geoms = {name: self.raster(poly) & ~excluded for name, poly in polys.items()}
        overlap = geoms["cheekbone"] & geoms["contour_hollow"]
        geoms["cheekbone"] = geoms["cheekbone"] & ~overlap
        geoms["contour_hollow"] = geoms["contour_hollow"] & ~overlap
        for name, geometric in geoms.items():
            z = self._finish(ZONE_SPEC_BY_ID[f"{name}_{k}"], geometric & clip, polygon=polys[name],
                             geometric=geometric)
            if name in ("cheek_upper", "cheekbone"):
                z = self._eye_area_guard(z, k, geometric)
            if name in ("cheek_outer", "contour_hollow"):
                self._set_hair_fraction(z, geometric)
            self._add(z)

    def _eye_zones(self, clip: np.ndarray, side: str, k: str, eye_poly: np.ndarray) -> None:
        """Batch 3a: under-eye halves, eye corners, lash lines for one side.

        Halves are landmark geometry clipped to skin (coverage rule only). Corners are
        checked by coverage within a tolerance of the parser eye mask. Lash lines are
        purely geometric strokes.
        """
        f = self.f
        i33, i133 = f.indices([33, 133], side)
        centre = (f.pts[i33] + f.pts[i133]) / 2.0
        sign = 1.0 if side == "right" else -1.0  # toward the nose is +u for the subject's right eye
        ue_poly, ue_geom = self._under_eye_geometric(side, eye_poly)
        halves = {"inner": np.zeros(self.shape, dtype=bool), "outer": np.zeros(self.shape, dtype=bool)}
        win = self._window(ue_geom)
        if win is not None:
            ys, xs, yy, xx = win
            t = sign * ((xx + self.origin[0] - centre[0]) * f.u[0] + (yy + self.origin[1] - centre[1]) * f.u[1])
            halves["inner"][ys, xs] = t > 0
            halves["outer"][ys, xs] = t <= 0
        for name, half in halves.items():
            geometric = ue_geom & half
            z = self._finish(ZONE_SPEC_BY_ID[f"under_eye_{name}_{k}"], geometric & clip,
                             polygon=ue_poly, geometric=geometric)
            self._add(self._eye_area_guard(z, k, geometric))

        eye_area = mk.dilate(self.raster(eye_poly), 1)
        near_model = mk.dilate(self.region(Region.EYE_R if side == "right" else Region.EYE_L),
                               EYE_CORNER_TOLERANCE * f.ipd)
        for name, index in (("inner", 133), ("outer", 33)):
            poly = self._disc_polygon(f.point(index, side=side), EYE_CORNER_RADIUS)
            mask = self.raster(poly) & eye_area
            n = int(np.count_nonzero(mask))
            coverage = float(np.count_nonzero(mask & near_model)) / n if n else 0.0
            z = self._finish(ZONE_SPEC_BY_ID[f"eye_corner_{name}_{k}"], mask, polygon=poly,
                             agreement=(coverage, MIN_EYE_CORNER_COVERAGE, "parser eye within tolerance"))
            if z.status == "low_confidence":
                z.reason = (f"Only {coverage:.0%} of the corner lies within {EYE_CORNER_TOLERANCE} ipd "
                            "of the parser eye mask (coverage within a tolerance, not IoU).")
            self._add(self._eye_area_guard(z, k, mask))

        stroke = max(MIN_STROKE_PX, LASH_STROKE * f.ipd)
        for name, ring in (("upper", EYE_UPPER), ("lower", EYE_LOWER)):
            poly = f.pts[f.indices(ring, side)]
            mask = mk.stroke_polyline(poly - self.origin, self.shape, stroke)
            z = self._finish(ZONE_SPEC_BY_ID[f"lash_line_{name}_{k}"], mask, polygon=poly, geometric_only=True)
            self._add(self._eye_area_guard(z, k, mask))

        lower = f.pts[f.indices(EYE_LOWER, side)]
        poly = np.vstack([lower, (lower + LOWER_EYELID_HEIGHT * f.ipd * f.v)[::-1]])
        geom = self.raster(poly) & ue_geom & ~eye_area
        z = self._finish(ZONE_SPEC_BY_ID[f"lower_eyelid_{k}"], geom & clip, polygon=poly, geometric=geom)
        self._add(self._eye_area_guard(z, k, geom))

    def _under_eye(self, clip: np.ndarray, side: str, k: str, eye_poly: np.ndarray) -> DerivedZone:
        poly, geometric = self._under_eye_geometric(side, eye_poly)
        return self._finish(ZONE_SPEC_BY_ID[f"under_eye_{k}"], geometric & clip, polygon=poly, geometric=geometric)

    def _model_zone(self, zone_id: str, region: Region, lm_poly: np.ndarray, kind: str,
                    other_poly: np.ndarray) -> DerivedZone:
        """Zone = parser mask; the landmark polygon only validates it (IoU).

        Backstop against a wrong side: the zone is never `ok` if more than 20 % of its
        pixels lie inside the other side's landmark polygon, or if it is wider than
        1.5x its own landmark polygon.
        """
        mask = self.region(region)
        iou = mk.iou(mask, self.raster(lm_poly)) if mask.any() else 0.0
        zone = self._finish(ZONE_SPEC_BY_ID[zone_id], mask, agreement=(iou, MIN_IOU[kind], kind))
        if zone.status == "unavailable" or zone.mask is None:
            return zone
        n = int(np.count_nonzero(mask))
        in_other = int(np.count_nonzero(mask & self.raster(other_poly))) / n
        own_w = float(lm_poly[:, 0].max() - lm_poly[:, 0].min())
        width = zone.bbox[2] if zone.bbox else 0
        if in_other > MAX_OTHER_SIDE_FRACTION:
            zone.status = "low_confidence"
            zone.reason = f"{in_other:.0%} of the mask lies on the other side's {kind} (side mismatch)."
        elif own_w > 0 and width > MAX_WIDTH_RATIO * own_w:
            zone.status = "low_confidence"
            zone.reason = f"Mask is {width / own_w:.1f}x wider than the landmark {kind} (side mismatch)."
        return zone

    def _nose(self) -> DerivedZone:
        mask = self.region(Region.NOSE)
        hull = self.f.pts[NOSE_HULL]
        iou = mk.iou(mask, self.raster(hull)) if mask.any() else 0.0
        return self._finish(ZONE_SPEC_BY_ID["nose"], mask, agreement=(iou, MIN_IOU["nose"], "nose region"))

    def _nose_zones(self) -> None:
        """Batch 5: nose bridge, tip, sides and nostrils.

        Landmark geometry clipped to the parser nose region, validated by the coverage rule
        only. Bridge, tip and sides are made disjoint; nostrils are exclusion areas and are
        left untouched.
        """
        f = self.f
        nose = self.region(Region.NOSE)
        bridge_poly = f.polygon(NOSE_BRIDGE_SPEC)
        tip_poly = f.polygon(NOSE_TIP_SPEC)
        tip_g = self.raster(tip_poly)
        bridge_g = self.raster(bridge_poly) & ~tip_g
        z = self._finish(ZONE_SPEC_BY_ID["nose_bridge"], bridge_g & nose, polygon=bridge_poly, geometric=bridge_g)
        self._add(self._eye_area_guard(z, "", bridge_g, parent="nose"))
        z = self._finish(ZONE_SPEC_BY_ID["nose_tip"], tip_g & nose, polygon=tip_poly, geometric=tip_g)
        self._add(z)
        for side, k in (("right", "r"), ("left", "l")):
            side_poly = f.polygon(NOSE_SIDE_SPEC, side)
            side_g = self.raster(side_poly) & ~bridge_g & ~tip_g
            z = self._finish(ZONE_SPEC_BY_ID[f"nose_side_{k}"], side_g & nose, polygon=side_poly, geometric=side_g)
            self._add(self._eye_area_guard(z, k, side_g, parent="nose"))
            centre = f.point(*NOSTRIL_CENTRE, side=side)
            ang = np.linspace(0.0, 2.0 * np.pi, EYE_CORNER_SEGMENTS, endpoint=False)
            nostril_poly = centre + f.ipd * (NOSTRIL_RADIUS_U * np.cos(ang)[:, None] * f.u
                                             + NOSTRIL_RADIUS_V * np.sin(ang)[:, None] * f.v)
            nostril_g = self.raster(nostril_poly)
            self._add(self._finish(ZONE_SPEC_BY_ID[f"nostril_{k}"], nostril_g & nose, polygon=nostril_poly,
                                   geometric=nostril_g))

    def _lips(self) -> None:
        f = self.f
        upper = np.vstack([f.pts[LIP_OUTER_UPPER], f.pts[LIP_INNER_UPPER[::-1]]])
        lower = np.vstack([f.pts[LIP_OUTER_LOWER], f.pts[LIP_INNER_LOWER[::-1]]])
        up_m, lo_m = self.raster(upper), self.raster(lower)
        model_up, model_lo = self.region(Region.LIP_UPPER), self.region(Region.LIP_LOWER)

        def lip(zone_id: str, mask: np.ndarray, poly: np.ndarray, model: np.ndarray) -> None:
            iou = mk.iou(mask, model) if model.any() else 0.0
            z = self._finish(ZONE_SPEC_BY_ID[zone_id], mask, polygon=poly, agreement=(iou, MIN_IOU["lip"], "lip polygon"))
            if z.status == "unavailable":
                z.reason = "Lip polygon is outside the parsed area."
            self._add(z)

        lip("lip_upper", up_m, upper, model_up)
        lip("lip_lower", lo_m, lower, model_lo)
        lip("lip_full", up_m | lo_m, np.vstack([f.pts[LIP_OUTER_UPPER], f.pts[LIP_OUTER_LOWER[::-1]]]),
            model_up | model_lo)

        lip_full = up_m | lo_m
        for side, k in (("right", "r"), ("left", "l")):
            self._add(self._lip_corner(side, k, lip_full, model_up | model_lo))

        cupid = f.pts[CUPID_BOW_PATH]
        cupid_mask = mk.stroke_polyline(cupid - self.origin, self.shape, max(MIN_STROKE_PX, CUPID_BOW_STROKE * f.ipd))
        self._add(self._finish(ZONE_SPEC_BY_ID["cupid_bow"], cupid_mask, polygon=cupid, geometric_only=True))

        stroke = max(2.0, 0.04 * f.ipd)
        for zone_id, ring in (
            ("lip_outer_boundary", LIP_OUTER_UPPER + LIP_OUTER_LOWER[::-1][1:-1]),
            ("lip_inner_boundary", LIP_INNER_UPPER + LIP_INNER_LOWER[::-1][1:-1]),
        ):
            poly = f.pts[ring]
            mask = mk.stroke_polyline(poly - self.origin, self.shape, stroke, closed=True)
            self._add(self._finish(ZONE_SPEC_BY_ID[zone_id], mask, polygon=poly, geometric_only=True))


def derive_zones(
    frame: FaceFrame, labels: np.ndarray, confidence: np.ndarray, origin: tuple[int, int]
) -> list[DerivedZone]:
    return ZoneDeriver(frame, labels, confidence, origin).derive()
