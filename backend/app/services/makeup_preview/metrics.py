"""Quality metrics for rendered previews. Pure numpy."""

import numpy as np

from app.services.makeup_preview.color import delta_e_2000, srgb_to_lab


def outside_changed_count(original: np.ndarray, rendered: np.ndarray, alpha: np.ndarray) -> int:
    """Pixels with alpha == 0 where any channel differs between original and rendered."""
    changed = np.any(original != rendered, axis=2)
    return int(np.count_nonzero(changed & (alpha == 0)))


def mean_delta_e_inside(original: np.ndarray, rendered: np.ndarray, mask: np.ndarray) -> float:
    sel = np.asarray(mask) > 0
    if not sel.any():
        return 0.0
    a = srgb_to_lab(original[sel].astype(np.float64) / 255.0)
    b = srgb_to_lab(rendered[sel].astype(np.float64) / 255.0)
    return float(np.mean(delta_e_2000(a, b)))


def delta_e_to_target(rendered: np.ndarray, mask: np.ndarray, target_lab: np.ndarray | tuple[float, float, float]) -> float:
    sel = np.asarray(mask) > 0
    if not sel.any():
        return 0.0
    lab = srgb_to_lab(rendered[sel].astype(np.float64) / 255.0)
    return float(np.mean(delta_e_2000(lab, np.asarray(target_lab, dtype=np.float64))))
