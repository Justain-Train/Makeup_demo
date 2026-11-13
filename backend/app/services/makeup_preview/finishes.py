"""Finish terms: an additive L (lightness) change per pixel. Pure; deterministic given the rng."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np

FinishKind = Literal["none", "lift", "specular", "noise"]
DEFAULT_FINISH = "natural"


@dataclass(frozen=True)
class FinishSpec:
    kind: FinishKind
    strength: float


FINISHES = MappingProxyType(
    {
        "matte": FinishSpec("none", 0.0),
        "satin": FinishSpec("lift", 1.5),
        "natural": FinishSpec("lift", 0.75),
        "velvet": FinishSpec("lift", 0.75),
        "dewy": FinishSpec("specular", 6.0),
        "glossy": FinishSpec("specular", 10.0),
        "shimmer": FinishSpec("noise", 3.0),
        "metallic": FinishSpec("noise", 4.0),
        "glitter": FinishSpec("noise", 8.0),
    }
)


def resolve_finish(name: str | None) -> str:
    return name if name is not None and name in FINISHES else DEFAULT_FINISH


def finish_delta_l(
    name: str,
    alpha: np.ndarray,
    weight: np.ndarray,
    rng: np.random.Generator,
    centroid_xy: tuple[float, float],
) -> np.ndarray:
    """float32 HxW additive L term. alpha and weight (= alpha * opacity) are HxW float32 over the ROI."""
    spec = FINISHES[name]
    if spec.kind == "none":
        return np.zeros(alpha.shape, dtype=np.float32)
    if spec.kind == "lift":
        return (weight * np.float32(spec.strength)).astype(np.float32)
    if spec.kind == "specular":
        h, w = alpha.shape
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        sigma = np.float32(max(2.0, 0.25 * max(h, w)))
        d2 = (xs - np.float32(centroid_xy[0])) ** 2 + (ys - np.float32(centroid_xy[1])) ** 2
        bump = np.exp(-d2 / (np.float32(2.0) * sigma * sigma))
        return (weight * bump * np.float32(spec.strength)).astype(np.float32)
    noise = rng.standard_normal(alpha.shape).astype(np.float32)
    return (noise * weight * np.float32(spec.strength)).astype(np.float32)
