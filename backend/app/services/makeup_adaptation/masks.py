"""Union of zone masks in image coordinates. Pure; no logging."""

import base64
import binascii
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from app.schemas.face_segmentation import Zone
from app.schemas.facial_profile import ImageSize
from app.services.segmentation import masks as mk

_MAX_PIXELS = 25_000_000
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class UnionResult:
    png_b64: str | None
    bbox: tuple[int, int, int, int] | None  # x, y, width, height in image px
    centroid: tuple[float, float] | None  # image px
    pixel_count: int
    invalid_ids: tuple[str, ...]


def _decode_guarded(zone: Zone, image_size: ImageSize) -> np.ndarray | None:
    """Same pre-decode guard as cv_analysis._crop, but without its in-image bbox requirement."""
    bbox, mask = zone.bbox, zone.mask
    if bbox is None or mask is None or mask.encoding != "png_bbox_v1":
        return None
    if bbox.width <= 0 or bbox.height <= 0 or bbox.width * bbox.height > _MAX_PIXELS:
        return None
    if bbox.width > image_size.width or bbox.height > image_size.height:
        return None
    if len(mask.png_base64) > 4 * math.ceil((bbox.width * bbox.height + 1024) / 3):
        return None
    try:
        raw = base64.b64decode(mask.png_base64, validate=True)
    except (ValueError, binascii.Error):
        return None
    if (
        len(raw) < 26
        or raw[:8] != _PNG_SIGNATURE
        or raw[12:16] != b"IHDR"
        or struct.unpack(">II", raw[16:24]) != (bbox.width, bbox.height)
        or raw[24] != 8
        or raw[25] != 0
    ):
        return None
    try:
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    except (cv2.error, ValueError):
        return None
    if arr is None:
        return None
    if arr.ndim != 2 or arr.shape != (bbox.height, bbox.width):
        return None
    return arr > 0


def union_zone_masks(zones: Sequence[Zone], image_size: ImageSize) -> UnionResult | None:
    """OR the zones' masks on one canvas. None if no zone carried a mask and bbox."""
    pieces: list[tuple[np.ndarray, int, int]] = []  # clamped crop, x0, y0
    invalid: list[str] = []
    seen_candidate = False
    for zone in zones:
        if zone.status == "unavailable" or zone.mask is None or zone.bbox is None:
            continue
        seen_candidate = True
        arr = _decode_guarded(zone, image_size)
        if arr is None:
            invalid.append(zone.id)
            continue
        b = zone.bbox
        x0, y0 = max(b.x, 0), max(b.y, 0)
        x1, y1 = min(b.x + b.width, image_size.width), min(b.y + b.height, image_size.height)
        if x1 <= x0 or y1 <= y0:
            continue
        pieces.append((arr[y0 - b.y : y1 - b.y, x0 - b.x : x1 - b.x], x0, y0))
    if not seen_candidate:
        return None
    if not pieces:
        return UnionResult(None, None, None, 0, tuple(invalid))

    ux0 = min(x for _, x, _ in pieces)
    uy0 = min(y for _, _, y in pieces)
    ux1 = max(x + c.shape[1] for c, x, _ in pieces)
    uy1 = max(y + c.shape[0] for c, _, y in pieces)
    canvas = np.zeros((uy1 - uy0, ux1 - ux0), dtype=np.uint8)
    for crop, x, y in pieces:
        view = canvas[y - uy0 : y - uy0 + crop.shape[0], x - ux0 : x - ux0 + crop.shape[1]]
        view |= crop.astype(np.uint8)
    count = int(np.count_nonzero(canvas))
    if count == 0:
        return UnionResult(None, None, None, 0, tuple(invalid))
    ys, xs = np.nonzero(canvas)
    centroid = (float(xs.mean()) + ux0, float(ys.mean()) + uy0)
    png = mk.encode_png_base64(canvas * 255)
    return UnionResult(png, (ux0, uy0, ux1 - ux0, uy1 - uy0), centroid, count, tuple(invalid))
