"""Face parsing: one square crop around the face -> canonical region label map.

This module (with `regions.py`) is the only place that knows the parsing
model's class ids. Callers get canonical region ids and never see raw logits.

Pipeline for `OnnxFaceParser.parse`:
  1. Square *parse window* = landmark bbox centre, side = crop_scale * bbox side.
     It may extend past the image; those pixels are padded with the ImageNet
     mean (which is 0 after normalisation).
  2. warpAffine window -> 512x512, normalise, run the network.
  3. Logits are upsampled bilinearly to the window (long side capped at
     `MAX_WORK_SIZE`), argmax'd, mapped to canonical ids, and nearest-mapped to
     the original pixels of (window ∩ image). The result is exact: window
     coordinates are integers, so the map back to the image is a pure shift
     when the window is <= the cap.
"""

import hashlib
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import onnxruntime as ort

from app.services.segmentation.regions import NOT_PARSED, NUM_MODEL_CLASSES, build_label_lut

logger = logging.getLogger(__name__)

INPUT_SIZE = 512
MAX_WORK_SIZE = 1024
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


class FaceParserError(Exception):
    """Base class for parser failures."""


class FaceParserUnavailableError(FaceParserError):
    """Weights are missing, corrupt or not the expected model."""


class FaceParserBusyError(FaceParserError):
    """All inference slots are in use and the wait timed out."""


class FaceParserInferenceError(FaceParserError):
    """Inference or post-processing failed."""


@dataclass(frozen=True)
class ParseWindow:
    x: int
    y: int
    size: int


@dataclass
class ParseResult:
    labels: np.ndarray  # (h, w) uint8 canonical region ids over (window ∩ image)
    confidence: np.ndarray  # (h, w) float32 softmax prob of the winning class
    window: ParseWindow
    origin: tuple[int, int]  # image-pixel position of labels[0, 0]
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ParserInfo:
    id: str
    architecture: str
    weights_sha256: str
    input_size: int


class FaceParser(Protocol):
    info: ParserInfo

    def parse(self, image: np.ndarray, face_bbox: tuple[float, float, float, float]) -> ParseResult:
        """`face_bbox` = (x_min, y_min, x_max, y_max) of the face landmarks, in image pixels."""
        ...


def compute_parse_window(
    face_bbox: tuple[float, float, float, float], crop_scale: float
) -> ParseWindow:
    x0, y0, x1, y1 = face_bbox
    side = max(int(math.ceil(crop_scale * max(x1 - x0, y1 - y0))), 16)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return ParseWindow(x=int(round(cx - side / 2.0)), y=int(round(cy - side / 2.0)), size=side)


def covered_region(window: ParseWindow, width: int, height: int) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of window ∩ image. Empty if x1 <= x0 or y1 <= y0."""
    return (
        max(window.x, 0),
        max(window.y, 0),
        min(window.x + window.size, width),
        min(window.y + window.size, height),
    )


class OnnxFaceParser:
    """BiSeNet-style 19-class parser running on onnxruntime (CPU)."""

    def __init__(
        self,
        model_path: Path,
        expected_sha256: str,
        crop_scale: float = 2.0,
        max_concurrency: int = 2,
        queue_timeout_s: float = 5.0,
        intra_op_threads: int = 0,
        model_id: str = "bisenet-resnet18-celebamask-hq",
        architecture: str = "BiSeNet (ResNet18), 19 classes",
    ) -> None:
        if not model_path.is_file():
            raise FaceParserUnavailableError(
                f"Face parser weights not found at {model_path}. "
                "Run `python scripts/download_face_parser.py` from backend/."
            )
        # Read once, hash the bytes, and build the session from those same
        # bytes, so the file cannot change between verification and loading.
        try:
            data = model_path.read_bytes()
        except OSError as exc:  # permission denied, is a directory, I/O error, ...
            logger.error("Face parser weights unreadable: %s", type(exc).__name__)
            raise FaceParserUnavailableError(
                f"Face parser weights at {model_path} could not be read ({type(exc).__name__}). "
                "Check the file permissions or re-download with `python scripts/download_face_parser.py`."
            ) from exc
        digest = hashlib.sha256(data).hexdigest()
        if not expected_sha256 or digest != expected_sha256.lower():
            raise FaceParserUnavailableError(
                f"Face parser weights at {model_path} do not match the expected SHA-256 "
                "(FACE_PARSER_SHA256). Re-download with `python scripts/download_face_parser.py`."
            )
        self.info = ParserInfo(
            id=model_id,
            architecture=architecture,
            weights_sha256=digest,
            input_size=INPUT_SIZE,
        )
        self._init_pipeline(crop_scale, max_concurrency, queue_timeout_s)

        opts = ort.SessionOptions()
        if intra_op_threads > 0:
            opts.intra_op_num_threads = intra_op_threads
        try:
            self._session = ort.InferenceSession(
                data, sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # noqa: BLE001 - onnxruntime raises many types
            raise FaceParserUnavailableError(
                f"Face parser weights could not be loaded ({type(exc).__name__})."
            ) from exc
        del data

        # Input/output names come from the model, not from memory. The first
        # output is the main head; the others are auxiliary training heads.
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self._output_name = self._session.get_outputs()[0].name
        shape = inp.shape
        if len(shape) != 4 or shape[1] != 3 or shape[2] != INPUT_SIZE or shape[3] != INPUT_SIZE:
            raise FaceParserUnavailableError(f"Unexpected face parser input shape {shape}.")
        # Warm-up also validates the output layout (19 channels at 512x512).
        try:
            out = self._infer(np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            raise FaceParserUnavailableError(
                f"Face parser warm-up failed ({type(exc).__name__})."
            ) from exc
        if out.shape != (NUM_MODEL_CLASSES, INPUT_SIZE, INPUT_SIZE):
            raise FaceParserUnavailableError(
                f"Unexpected face parser output shape {out.shape}; expected "
                f"({NUM_MODEL_CLASSES}, {INPUT_SIZE}, {INPUT_SIZE})."
            )

    def _init_pipeline(self, crop_scale: float, max_concurrency: int, queue_timeout_s: float) -> None:
        # Separate from __init__ so tests can build a parser around a stub `_infer`.
        self._crop_scale = crop_scale
        self._lut = np.array(build_label_lut(), dtype=np.uint8)
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._queue_timeout_s = queue_timeout_s

    # -- model call (the only place the session is touched; tests override it) --
    def _infer(self, x: np.ndarray) -> np.ndarray:
        """(1, 3, 512, 512) float32 -> (19, 512, 512) float32 logits."""
        return self._session.run([self._output_name], {self._input_name: x})[0][0]

    def parse(self, image: np.ndarray, face_bbox: tuple[float, float, float, float]) -> ParseResult:
        if not self._slots.acquire(timeout=self._queue_timeout_s):
            raise FaceParserBusyError("Face parser is busy.")
        try:
            return self._parse(image, face_bbox)
        except FaceParserError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Log the type only: messages could contain image-derived data.
            logger.error("Face parsing failed: %s", type(exc).__name__)
            raise FaceParserInferenceError("Face parsing failed.") from exc
        finally:
            self._slots.release()

    def _parse(self, image: np.ndarray, face_bbox: tuple[float, float, float, float]) -> ParseResult:
        height, width = image.shape[:2]
        window = compute_parse_window(face_bbox, self._crop_scale)
        rx0, ry0, rx1, ry1 = covered_region(window, width, height)
        if rx1 <= rx0 or ry1 <= ry0:
            raise FaceParserInferenceError("Face is outside the image.")

        t0 = time.perf_counter()
        k_in = INPUT_SIZE / window.size
        # Only the covered part of the window is read. Heavy down-scaling
        # (large photos) is pre-shrunk with INTER_AREA to avoid aliasing,
        # because warpAffine's bilinear sampling does not average.
        sub = image[ry0:ry1, rx0:rx1]
        shrink = 1.0 if k_in >= 0.5 else 2.0 * k_in
        if shrink < 1.0:
            # Explicit dsize: with fx/fy OpenCV's INTER_AREA uses the nominal factor for its
            # averaging footprint even though the output size is rounded.
            new_size = (max(1, round(sub.shape[1] * shrink)), max(1, round(sub.shape[0] * shrink)))
            sub = cv2.resize(sub, new_size, interpolation=cv2.INTER_AREA)
        sub = np.ascontiguousarray(sub)
        # Pixel-centre (half-pixel) convention throughout, like cv2.resize and the inverse
        # map below. The pre-shrink output size is rounded, so use the scale actually produced
        # (sub.shape), not the nominal `shrink`.
        sx = sub.shape[1] / (rx1 - rx0)
        sy = sub.shape[0] / (ry1 - ry0)
        # sub pixel q = sx * (image_px - rx0 + 0.5) - 0.5
        # dst       = k_in * (image_px - window_origin + 0.5) - 0.5
        m = np.array(
            [
                [k_in / sx, 0.0, k_in * (rx0 - window.x + 0.5 / sx) - 0.5],
                [0.0, k_in / sy, k_in * (ry0 - window.y + 0.5 / sy) - 0.5],
            ],
            dtype=np.float64,
        )
        crop = cv2.warpAffine(
            sub,
            m,
            (INPUT_SIZE, INPUT_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=tuple(float(v) for v in IMAGENET_MEAN),
        )
        x = ((crop.astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
        t1 = time.perf_counter()

        logits = self._infer(np.ascontiguousarray(x))
        t2 = time.perf_counter()

        # Softmax confidence of the winning class, computed at 512 (cheap).
        shifted = np.exp(logits - logits.max(axis=0, keepdims=True))
        conf512 = (1.0 / shifted.sum(axis=0)).astype(np.float32)

        work = min(window.size, MAX_WORK_SIZE)
        up = cv2.resize(
            np.ascontiguousarray(logits.transpose(1, 2, 0)),
            (work, work),
            interpolation=cv2.INTER_LINEAR,
        )
        labels_work = self._lut[up.argmax(axis=2).astype(np.uint8)]
        conf_work = cv2.resize(conf512, (work, work), interpolation=cv2.INTER_LINEAR)

        # Map the working-resolution window onto the original pixels of the
        # covered region. For work == window.size this is an exact integer shift.
        k = work / window.size
        rw, rh = rx1 - rx0, ry1 - ry0
        mm = np.array(
            [
                [k, 0.0, (rx0 - window.x + 0.5) * k - 0.5],
                [0.0, k, (ry0 - window.y + 0.5) * k - 0.5],
            ],
            dtype=np.float64,
        )
        flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
        labels = cv2.warpAffine(
            labels_work, mm, (rw, rh), flags=flags,
            borderMode=cv2.BORDER_CONSTANT, borderValue=NOT_PARSED,
        )
        conf = cv2.warpAffine(
            conf_work, mm, (rw, rh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        t3 = time.perf_counter()
        return ParseResult(
            labels=labels,
            confidence=conf,
            window=window,
            origin=(rx0, ry0),
            timings_ms={
                "preprocess": (t1 - t0) * 1000,
                "inference": (t2 - t1) * 1000,
                "postprocess": (t3 - t2) * 1000,
            },
        )
