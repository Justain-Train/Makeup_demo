"""Assign left/right to the model's side-agnostic paired regions from landmarks.

Convention (MediaPipe landmark convention, same as everywhere else): subject-RIGHT
is the side containing landmarks 33/133/468 (the image's left in an unmirrored
photo). Each pixel of a side-agnostic eye/brow/ear region goes to the side of the
nearest landmark of that feature, so merged or touching regions (brows meeting at
the glabella) are still split correctly and head roll does not matter. A mirrored
photo gets sides that follow the image geometry (swapped vs. the real person).

Canonical ids only; model class ids never appear here.
"""

import numpy as np

from app.services.segmentation.regions import Region, Unsided

# Landmark sets for the subject's RIGHT side, and their mirrors for the LEFT.
_EYE_R = [33, 246, 161, 160, 159, 158, 157, 173, 133, 7, 163, 144, 145, 153, 154, 155, 468]
_EYE_L = [263, 466, 388, 387, 386, 385, 384, 398, 362, 249, 390, 373, 374, 380, 381, 382, 473]
_BROW_R = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55]
_BROW_L = [300, 293, 334, 296, 336, 276, 283, 282, 295, 285]
_EAR_R = [234, 127, 162, 93, 132]  # face-edge points on the right
_EAR_L = [454, 356, 389, 323, 361]

_SETS = {
    Unsided.EYE: (_EYE_R, _EYE_L, Region.EYE_R, Region.EYE_L),
    Unsided.BROW: (_BROW_R, _BROW_L, Region.BROW_R, Region.BROW_L),
    Unsided.EAR: (_EAR_R, _EAR_L, Region.EAR_R, Region.EAR_L),
}


def assign_sides(labels: np.ndarray, pts: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
    """Return a copy of `labels` with every `Unsided` id replaced by a sided Region id.

    labels: (h, w) uint8 over image pixels starting at `origin`; pts: (478, 2) landmark pixels.
    """
    out = labels.copy()
    o = np.array(origin, dtype=np.float64)
    for unsided, (right_ids, left_ids, right_region, left_region) in _SETS.items():
        ys, xs = np.nonzero(labels == int(unsided))
        if len(xs) == 0:
            continue
        p = np.stack([xs, ys], axis=1).astype(np.float64) + o
        right, left = pts[right_ids], pts[left_ids]
        d_r = np.empty(len(p))
        d_l = np.empty(len(p))
        for i in range(0, len(p), 20000):  # chunked: bounded memory on large photos
            chunk = p[i : i + 20000, None, :]
            d_r[i : i + 20000] = np.min(np.sum((chunk - right[None]) ** 2, axis=2), axis=1)
            d_l[i : i + 20000] = np.min(np.sum((chunk - left[None]) ** 2, axis=2), axis=1)
        out[ys, xs] = np.where(d_r <= d_l, int(right_region), int(left_region))
    return out
