"""Decode a plan component's zone mask onto a full-image canvas. Pure; no logging."""

import base64
import binascii
import math
import struct
from typing import Literal

import cv2
import numpy as np

from app.schemas.adapted_makeup_plan import AdaptedComponent

MaskReason = Literal["no_mask", "mask_invalid"]

_MAX_PIXELS = 25_000_000
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def decode_mask_full(
    component: AdaptedComponent, image_hw: tuple[int, int]
) -> tuple[np.ndarray | None, MaskReason | None]:
    """uint8 HxW canvas (0/255) with the bbox mask pasted and clipped to the image."""
    bbox, mask = component.bbox, component.mask
    if bbox is None or mask is None:
        return None, "no_mask"
    img_h, img_w = image_hw
    if mask.encoding != "png_bbox_v1":
        return None, "mask_invalid"
    if bbox.width <= 0 or bbox.height <= 0 or bbox.width * bbox.height > _MAX_PIXELS:
        return None, "mask_invalid"
    if img_h <= 0 or img_w <= 0 or img_h * img_w > _MAX_PIXELS:
        return None, "mask_invalid"
    if len(mask.png_base64) > 4 * math.ceil((bbox.width * bbox.height + 1024) / 3):
        return None, "mask_invalid"
    try:
        raw = base64.b64decode(mask.png_base64, validate=True)
    except (ValueError, binascii.Error):
        return None, "mask_invalid"
    if (
        len(raw) < 26
        or raw[:8] != _PNG_SIGNATURE
        or raw[12:16] != b"IHDR"
        or struct.unpack(">II", raw[16:24]) != (bbox.width, bbox.height)
        or raw[24] != 8
        or raw[25] != 0
    ):
        return None, "mask_invalid"
    try:
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    except (cv2.error, ValueError):
        return None, "mask_invalid"
    if arr is None or arr.ndim != 2 or arr.shape != (bbox.height, bbox.width):
        return None, "mask_invalid"

    canvas = np.zeros((img_h, img_w), dtype=np.uint8)
    x0, y0 = max(bbox.x, 0), max(bbox.y, 0)
    x1, y1 = min(bbox.x + bbox.width, img_w), min(bbox.y + bbox.height, img_h)
    if x1 > x0 and y1 > y0:
        crop = arr[y0 - bbox.y : y1 - bbox.y, x0 - bbox.x : x1 - bbox.x]
        canvas[y0:y1, x0:x1] = (crop > 0).astype(np.uint8) * 255
    return canvas, None
