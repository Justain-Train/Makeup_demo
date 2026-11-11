"""Small mask utilities (OpenCV-backed). Masks are 2D bool arrays unless noted.

All drawing functions take points in the mask's own pixel coordinates. Callers
subtract the canvas origin first.
"""

import base64

import cv2
import numpy as np

PNG_COMPRESSION = 1  # low level: fast to encode, payload stays small enough for sparse masks


def rasterize_polygon(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Filled polygon -> bool mask of `shape` (h, w). Points are (N, 2) floats."""
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) >= 3:
        # fillPoly takes integer vertices; shift=4 keeps 1/16 px precision.
        pts = np.round(np.asarray(points, dtype=np.float64) * 16).astype(np.int32)
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 1, lineType=cv2.LINE_8, shift=4)
    return mask.astype(bool)


def stroke_polyline(
    points: np.ndarray, shape: tuple[int, int], thickness: float, closed: bool = False
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) >= 2:
        pts = np.round(np.asarray(points, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [pts], closed, 1, thickness=max(1, int(round(thickness))), lineType=cv2.LINE_8)
    return mask.astype(bool)


def dilate(mask: np.ndarray, radius_px: float) -> np.ndarray:
    r = int(round(radius_px))
    if r <= 0:
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected component (empty in -> empty out)."""
    if not mask.any():
        return mask.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask.copy()
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


def bbox_of(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x, y, width, height) of the True pixels, or None if empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def polygon_area(points: np.ndarray) -> float:
    """Shoelace area (absolute)."""
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def encode_png(array: np.ndarray) -> bytes:
    """8-bit grayscale PNG. Bool arrays become 0/255 so the PNG is viewable."""
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    ok, buf = cv2.imencode(".png", array, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def encode_png_base64(array: np.ndarray) -> str:
    return base64.b64encode(encode_png(array)).decode("ascii")


def decode_png(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("PNG decoding failed")
    return arr
