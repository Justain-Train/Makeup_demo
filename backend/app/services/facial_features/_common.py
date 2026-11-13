"""Shared helpers: frame projection, banding and the never-raise guard. No logging by design."""

import math
from collections.abc import Callable, Sequence
from functools import wraps

import numpy as np

from app.schemas.facial_profile import Classified
from app.services.application_zones.frame import FaceFrame

Classifier = Callable[[np.ndarray, FaceFrame], Classified]


def unknown(reason: str, measurements: dict[str, float] | None = None) -> Classified:
    return Classified(label="unknown", confidence=0.0, measurements=measurements or {}, reason=reason)


def guarded(fn: Classifier) -> Classifier:
    """Any failure becomes label 'unknown' with a fixed reason; no input values leak."""

    @wraps(fn)
    def wrapper(pts: np.ndarray, frame: FaceFrame) -> Classified:
        try:
            return fn(pts, frame)
        except Exception:
            return unknown("degenerate_input")

    return wrapper


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def project(pts: np.ndarray, frame: FaceFrame, index: int) -> np.ndarray:
    """Landmark in frame coordinates (along u, along v), so head roll is removed."""
    p = np.asarray(pts[index], dtype=np.float64)
    return np.array([float(p @ frame.u), float(p @ frame.v)])


def agreement(a: float, b: float, scale: float) -> float:
    return clamp01(1.0 - abs(a - b) / scale)


def band(
    value: float,
    edges: Sequence[float],
    labels: Sequence[str],
    margin: float,
    scale: float,
    measurements: dict[str, float],
    confidence_factor: float = 1.0,
) -> Classified:
    if not math.isfinite(value) or not all(math.isfinite(v) for v in measurements.values()):
        return unknown("non_finite_measurement")
    distance = min(abs(value - e) for e in edges)
    confidence = clamp01(distance / scale) * clamp01(confidence_factor)
    if distance < margin:
        return Classified(
            label="unknown", confidence=confidence, measurements=measurements, reason="within_ambiguity_margin"
        )
    index = sum(1 for e in edges if value >= e)
    return Classified(label=labels[index], confidence=confidence, measurements=measurements)
