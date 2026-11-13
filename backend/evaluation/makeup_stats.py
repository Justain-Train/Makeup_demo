"""Small statistics helpers for the makeup evaluation (pure numpy)."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from app.services.makeup_preview.color import delta_e_2000
from evaluation.report import BOOT_N, BOOT_SEED

BASELINE_MIN_DE = 1.0


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) + 0.5 P(tie), from average ranks."""
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    avg = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = avg[inv]
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def paired_auc(pos: np.ndarray, own_neg: np.ndarray) -> float:
    pos, own_neg = np.asarray(pos, dtype=np.float64), np.asarray(own_neg, dtype=np.float64)
    if pos.size == 0:
        return float("nan")
    return float(np.mean((pos > own_neg) + 0.5 * (pos == own_neg)))


def threshold_at_fp(neg: np.ndarray, max_fp: float = 0.05) -> float:
    """Smallest score threshold t (detect when score >= t) with at most max_fp of negatives at or above it."""
    neg = np.sort(np.asarray(neg, dtype=np.float64))
    n = neg.size
    if n == 0:
        return float("nan")
    k = int(math.floor(max_fp * n))
    return float(np.nextafter(neg[n - k - 1], np.inf))


def prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"precision": precision, "recall": recall, "f1": f1}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(
    stat: Callable[..., float | None], sizes: tuple[int, ...], n_boot: int = BOOT_N, seed: int = BOOT_SEED
) -> tuple[float, float] | None:
    """Percentile 95% interval. `stat` receives one resampled index array per entry of `sizes`."""
    if not sizes or any(s <= 0 for s in sizes):
        return None
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        v = stat(*(rng.integers(0, s, s) for s in sizes))
        if v is not None and math.isfinite(v):
            vals.append(float(v))
    if len(vals) < 10:
        return None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def gap_closed(preview_lab: np.ndarray, truth_lab: np.ndarray, bare_lab: np.ndarray) -> float | None:
    """1 - dE00(preview, truth) / dE00(bare, truth) on skin-relative Lab; None when the baseline is < 1 dE00."""
    baseline = float(delta_e_2000(np.asarray(bare_lab), np.asarray(truth_lab)))
    if baseline < BASELINE_MIN_DE:
        return None
    return 1.0 - float(delta_e_2000(np.asarray(preview_lab), np.asarray(truth_lab))) / baseline
