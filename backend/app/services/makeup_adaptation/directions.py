"""Direction labels to face-frame and image-frame angles. Pure; uses only FaceFrameData (u, v, roll_deg).

Face frame: 0 degrees is outward on the subject's own side, positive toward the forehead; the right side
mirrors the left. Image frame: standard math angle with image y flipped (0 = image right, 90 = image up).
"""

import math

from app.schemas.adapted_makeup_plan import Adjustment, Direction, DirectionAngle
from app.schemas.facial_profile import FaceFrameData, FacialProfile

DIRECTION_ELEVATION_DEG: dict[str, float] = {
    "horizontal": 0.0,
    "outward": 10.0,
    "upward_outward": 25.0,
    "diagonal_up": 45.0,
}

_UNIT_TOLERANCE = 0.05


def _finite(*values: float) -> bool:
    return all(math.isfinite(x) for x in values)


def usable_frame(profile: FacialProfile) -> FaceFrameData | None:
    comps = profile.components
    if comps is None or comps.frame.data is None:
        return None
    f = comps.frame.data
    if not _finite(f.roll_deg, *f.u, *f.v):
        return None
    if abs(math.hypot(*f.u) - 1.0) > _UNIT_TOLERANCE:
        return None
    return f


def _normalise(deg: float) -> float:
    d = math.fmod(deg, 360.0)
    if d <= -180.0:
        d += 360.0
    elif d > 180.0:
        d -= 360.0
    return d


def angles_for(direction: Direction, frame: FaceFrameData) -> list[DirectionAngle]:
    elev = DIRECTION_ELEVATION_DEG[direction]
    rad = math.radians(elev)
    c, s = math.cos(rad), math.sin(rad)
    ux, uy = frame.u
    vx, vy = frame.v
    out: list[DirectionAngle] = []
    # u points from the subject's right eye to the left eye, so +u is outward on the subject's left.
    for side, sign in (("left", 1.0), ("right", -1.0)):
        dx = sign * c * ux - s * vx
        dy = sign * c * uy - s * vy
        image_deg = _normalise(math.degrees(math.atan2(-dy, dx)))
        out.append(DirectionAngle(side=side, face_deg=elev, image_deg=image_deg))
    return out


def attach_angles(adjustments: list[Adjustment], frame: FaceFrameData | None) -> list[Adjustment]:
    """Rebuild adjustments via model_validate, adding left and right angles to direction-effect ones."""
    out: list[Adjustment] = []
    for adj in adjustments:
        if frame is None or adj.effect != "direction" or adj.direction is None:
            out.append(adj)
            continue
        data = adj.model_dump()
        data["direction_angles"] = [a.model_dump() for a in angles_for(adj.direction, frame)]
        out.append(Adjustment.model_validate(data))
    return out
