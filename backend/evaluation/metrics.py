"""Pure numpy metrics for the evaluation. No I/O, no network, no model weights.

Everything class-level is derived from a per-image confusion matrix (EP class x canonical class + NOT_PARSED),
so full-image and in-window scoring, and the A/B denominators, are sums of small matrices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from evaluation.labels import (
    CONF_COLS,
    EP_NUM_CLASSES,
    MIN_PIXELS,
    NOT_PARSED_COL,
    pred_to_columns,
)


# --- confusion matrix ---------------------------------------------------------------------------

def confusion_matrix(
    gt: np.ndarray, pred: np.ndarray, rect: tuple[int, int, int, int] | None = None, valid: np.ndarray | None = None
) -> np.ndarray:
    """(EP_NUM_CLASSES, CONF_COLS) int64 pixel counts.

    gt: (H, W) uint8 EP ids. pred: (H, W) uint8 canonical ids, 255 = not predicted.
    rect: (x0, y0, x1, y1) restricts the count to that rectangle (the in-window scoring); None = whole image.
    valid: optional bool mask of pixels that count at all (e.g. not rotation fill); others are not scored.
    """
    if gt.shape != pred.shape:
        raise ValueError("gt and pred must have the same shape")
    if gt.size and int(gt.max()) >= EP_NUM_CLASSES:
        raise ValueError("gt contains ids outside the EasyPortrait classes")
    if rect is not None:
        x0, y0, x1, y1 = rect
        gt, pred = gt[y0:y1, x0:x1], pred[y0:y1, x0:x1]
        valid = None if valid is None else valid[y0:y1, x0:x1]
    idx = gt.astype(np.intp) * CONF_COLS + pred_to_columns(pred)
    if valid is not None:
        idx = idx[valid]
    return np.bincount(idx.ravel(), minlength=EP_NUM_CLASSES * CONF_COLS).reshape(EP_NUM_CLASSES, CONF_COLS).astype(np.int64)


def not_predicted_confusion(gt: np.ndarray) -> np.ndarray:
    """Confusion matrix of an image without a prediction (all pixels NOT_PARSED)."""
    conf = np.zeros((EP_NUM_CLASSES, CONF_COLS), dtype=np.int64)
    conf[:, NOT_PARSED_COL] = np.bincount(gt.ravel(), minlength=EP_NUM_CLASSES)[:EP_NUM_CLASSES]
    return conf


@dataclass(frozen=True)
class Counts:
    tp: int
    fp: int
    fn: int

    @property
    def gt(self) -> int:
        return self.tp + self.fn

    @property
    def pred(self) -> int:
        return self.tp + self.fp


def counts_from_confusion(conf: np.ndarray, ep_ids: tuple[int, ...] | list[int], cols: list[int]) -> Counts:
    """tp/fp/fn of the union of `ep_ids` (GT) against the union of canonical `cols` (prediction)."""
    gt_total = int(conf[list(ep_ids)].sum())
    pred_total = int(conf[:, cols].sum())
    tp = int(conf[np.ix_(list(ep_ids), cols)].sum())
    return Counts(tp, pred_total - tp, gt_total - tp)


def counts_from_masks(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Counts:
    tp = int(np.count_nonzero(gt_mask & pred_mask))
    return Counts(tp, int(np.count_nonzero(pred_mask)) - tp, int(np.count_nonzero(gt_mask)) - tp)


# --- aggregate scores ---------------------------------------------------------------------------

def _ratio(num: float, den: float) -> float | None:
    return None if den <= 0 else num / den


def scores(tp: float, fp: float, fn: float) -> dict[str, float | None]:
    """Dataset-level (pixel-weighted) IoU, Dice, precision, recall from summed counts."""
    return {
        "iou": _ratio(tp, tp + fp + fn),
        "dice": _ratio(2 * tp, 2 * tp + fp + fn),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
    }


def bootstrap_scores(
    per_image: np.ndarray, n_resamples: int = 1000, seed: int = 0, chunk: int = 250
) -> dict[str, tuple[float | None, float | None]]:
    """95 % percentile bootstrap CIs over images. per_image: (n, 3) array of tp, fp, fn.

    Resampling is over whole images (the unit of independence), seeded, so it is reproducible.
    """
    arr = np.asarray(per_image, dtype=np.float64)
    names = ("iou", "dice", "precision", "recall")
    if arr.ndim != 2 or arr.shape[0] == 0:
        return {k: (None, None) for k in names}
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    samples = {k: [] for k in names}
    done = 0
    while done < n_resamples:
        b = min(chunk, n_resamples - done)
        idx = rng.integers(0, n, size=(b, n))
        s = arr[idx].sum(axis=1)  # (b, 3)
        tp, fp, fn = s[:, 0], s[:, 1], s[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            samples["iou"].append(tp / (tp + fp + fn))
            samples["dice"].append(2 * tp / (2 * tp + fp + fn))
            samples["precision"].append(tp / (tp + fp))
            samples["recall"].append(tp / (tp + fn))
        done += b
    out: dict[str, tuple[float | None, float | None]] = {}
    for k, parts in samples.items():
        v = np.concatenate(parts)
        v = v[np.isfinite(v)]
        out[k] = (None, None) if v.size == 0 else (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


def per_image_iou_stats(counts: list[Counts], min_gt: int = MIN_PIXELS) -> dict[str, float | int | None]:
    """Mean, median, p10, p90 of per-image IoU, over images whose GT region has >= min_gt pixels."""
    vals = [c.tp / (c.tp + c.fp + c.fn) for c in counts if c.gt >= min_gt]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    a = np.array(vals)
    return {
        "n": len(vals),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


def false_alarm_rate(counts: list[Counts], min_pixels: int = MIN_PIXELS) -> dict[str, float | int | None]:
    """Among images whose GT region is (nearly) absent, the share where the prediction has >= min_pixels."""
    absent = [c for c in counts if c.gt < min_pixels]
    if not absent:
        return {"n_absent": 0, "n_false_alarm": 0, "rate": None}
    fa = sum(1 for c in absent if c.pred >= min_pixels)
    return {"n_absent": len(absent), "n_false_alarm": fa, "rate": fa / len(absent)}


# --- boundary F-score ----------------------------------------------------------------------------

_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    return (m.astype(bool)) & ~cv2.erode(m, _CROSS, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def boundary_f(gt: np.ndarray, pred: np.ndarray, tau_px: float) -> float | None:
    """Boundary F-score with tolerance tau_px (pixels), on the union bounding box of both masks.

    None when both masks are empty; 0.0 when exactly one is empty.
    """
    gb, pb = _bbox(gt), _bbox(pred)
    if gb is None and pb is None:
        return None
    if gb is None or pb is None:
        return 0.0
    pad = int(math.ceil(tau_px)) + 2
    x0 = max(min(gb[0], pb[0]) - pad, 0)
    y0 = max(min(gb[1], pb[1]) - pad, 0)
    x1 = min(max(gb[2], pb[2]) + pad, gt.shape[1])
    y1 = min(max(gb[3], pb[3]) + pad, gt.shape[0])
    g, p = gt[y0:y1, x0:x1], pred[y0:y1, x0:x1]
    # Pad by one background pixel so a region touching the crop edge still has a boundary there.
    g = np.pad(g, 1)
    p = np.pad(p, 1)
    g_edge, p_edge = _boundary(g), _boundary(p)
    # distance from every pixel to the nearest boundary pixel of the other mask
    d_to_g = cv2.distanceTransform((~g_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_to_p = cv2.distanceTransform((~p_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    n_p, n_g = int(p_edge.sum()), int(g_edge.sum())
    if n_p == 0 or n_g == 0:
        return 0.0
    precision = float((d_to_g[p_edge] <= tau_px).sum()) / n_p
    recall = float((d_to_p[g_edge] <= tau_px).sum()) / n_g
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def best_shift(gt: np.ndarray, pred: np.ndarray, max_shift: int = 3) -> tuple[int, int, float] | None:
    """(dx, dy, iou) of the integer shift of `pred` (within +-max_shift) that maximises IoU with `gt`.

    Positive dx moves the prediction right, positive dy down. Ties go to the smaller shift. None if either is empty.
    """
    gb, pb = _bbox(gt), _bbox(pred)
    if gb is None or pb is None:
        return None
    pad = max_shift + 1
    x0 = max(min(gb[0], pb[0]) - pad, 0)
    y0 = max(min(gb[1], pb[1]) - pad, 0)
    x1 = min(max(gb[2], pb[2]) + pad, gt.shape[1])
    y1 = min(max(gb[3], pb[3]) + pad, gt.shape[0])
    g = np.pad(gt[y0:y1, x0:x1], max_shift)
    p = np.pad(pred[y0:y1, x0:x1], max_shift)
    h, w = g.shape
    best: tuple[float, int, int, int] | None = None
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = np.zeros_like(p)
            ys, yd = slice(max(0, -dy), h - max(0, dy)), slice(max(0, dy), h - max(0, -dy))
            xs, xd = slice(max(0, -dx), w - max(0, dx)), slice(max(0, dx), w - max(0, -dx))
            shifted[yd, xd] = p[ys, xs]
            inter = int(np.count_nonzero(g & shifted))
            union = int(np.count_nonzero(g | shifted))
            iou = inter / union if union else 0.0
            key = (iou, -(abs(dx) + abs(dy)), dx, dy)
            if best is None or key[:2] > best[:2]:
                best = key  # type: ignore[assignment]
    assert best is not None
    return best[2], best[3], best[0]


# --- geometry of the GT --------------------------------------------------------------------------

def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size < MIN_PIXELS:
        return None
    return float(xs.mean()), float(ys.mean())


def inter_ocular_distance(left_eye: np.ndarray, right_eye: np.ndarray) -> float | None:
    """Distance between the two eye-region centroids of the GT (pixels), or None."""
    a, b = centroid(left_eye), centroid(right_eye)
    return None if a is None or b is None else math.hypot(a[0] - b[0], a[1] - b[1])


def side_order_ok(first: np.ndarray, second: np.ndarray) -> bool | None:
    """True when `first` (EP LEFT_*) lies to the image-left of `second` (EP RIGHT_*), by centroid x."""
    a, b = centroid(first), centroid(second)
    return None if a is None or b is None else a[0] < b[0]


def mean_luma(rgb: np.ndarray, mask: np.ndarray) -> float | None:
    """Mean Rec.601 luma (0-255) of the masked pixels. A lighting indicator, NOT a skin-tone measure."""
    if np.count_nonzero(mask) < MIN_PIXELS:
        return None
    px = rgb[mask].astype(np.float64)
    return float((px @ np.array([0.299, 0.587, 0.114])).mean())


def side_swap(conf: np.ndarray, first_ep: int, second_ep: int, first_col: int, second_col: int) -> bool | None:
    """Is the sided prediction swapped relative to the sided GT, judged by overlap?

    first_* = the image-left GT structure and the region we map it to. Undefined (None) without any overlap.
    """
    aligned = int(conf[first_ep, first_col] + conf[second_ep, second_col])
    swapped = int(conf[first_ep, second_col] + conf[second_ep, first_col])
    if aligned + swapped == 0:
        return None
    return swapped > aligned


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.array(values), q)) if values else None
