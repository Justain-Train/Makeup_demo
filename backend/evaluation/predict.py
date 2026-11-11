"""Run the real pipeline on one array and return the canonical label map embedded in the full image."""

from __future__ import annotations

import base64

import numpy as np

from app.services.face_regions import segment_face
from app.services.segmentation import masks as mk
from app.services.segmentation.regions import NOT_PARSED
from evaluation.runner import STATUS_TO_OUTCOME


def predict_full(rgb: np.ndarray, landmark_service, parser) -> tuple[str, np.ndarray | None, dict[str, float]]:
    """(outcome, (H, W) uint8 canonical labels with 255 outside the parse window or None, timings_ms)."""
    h, w = rgb.shape[:2]
    try:
        resp = segment_face(rgb, landmark_service, parser)
    except Exception:  # noqa: BLE001 - any failure is an outcome
        return "pipeline_error", None, {}
    outcome = STATUS_TO_OUTCOME.get(resp.status, "pipeline_error")
    if outcome != "ok" or resp.anatomical is None:
        return (outcome if outcome != "ok" else "pipeline_error"), None, {k: float(v) for k, v in resp.timings_ms.items()}
    a = resp.anatomical
    labels = mk.decode_png(base64.b64decode(a.label_map_png_base64))
    pred = np.full((h, w), NOT_PARSED, dtype=np.uint8)
    pred[a.origin.y : a.origin.y + a.height, a.origin.x : a.origin.x + a.width] = labels
    return "ok", pred, {k: float(v) for k, v in resp.timings_ms.items()}
