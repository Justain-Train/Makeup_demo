"""Offline run: score the real pipeline (`segment_face`) against EasyPortrait annotations.

Sequential, no network. Services are built the way `app/main.py`'s lifespan builds them (without editing it).
Per-image rows go to `per_image.jsonl` OUTSIDE the repository; logs carry ids, outcome categories, timings
and exception type names only.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import BACKEND_DIR, get_settings
from app.services.application_zones import derive as dz
from app.services.application_zones.frame import FaceFrame
from app.services.face_regions import segment_face
from app.services.facial_analysis.landmarks import FaceLandmarkService
from app.services.image_loading import ImageValidationError, load_rgb_image
from app.services.segmentation import masks as mk
from app.services.segmentation.face_parser import FaceParser, OnnxFaceParser
from app.services.segmentation.regions import NOT_PARSED, Region
from evaluation import metrics
from evaluation.dataset import (
    InvalidSample,
    SampleEntry,
    annotation_path,
    decode_annotation,
    image_path,
    refuse_inside_repo,
)
from evaluation.labels import (
    COMPARISONS,
    EP_LEFT_BROW,
    EP_LEFT_EYE,
    EP_LIPS,
    EP_RIGHT_BROW,
    EP_RIGHT_EYE,
    EP_SKIN,
    EP_TEETH,
    ep_mask,
    pred_mask,
)

logger = logging.getLogger("evaluation")

OUTCOMES = ("ok", "no_face", "multiple_faces", "pipeline_error", "invalid_sample")
STATUS_TO_OUTCOME = {"ok": "ok", "no_face_detected": "no_face", "multiple_faces_detected": "multiple_faces"}
TAUS = (0.02, 0.05)  # boundary tolerance as a fraction of the GT inter-ocular distance
MAX_INVALID_FRACTION = 0.01
GLASSES_WARNING = "glasses_detected"


def build_services(crop_scale: float = 2.0, threads: int | None = None) -> tuple[FaceLandmarkService, FaceParser]:
    s = get_settings()
    landmarks = FaceLandmarkService(
        model_path=s.resolved_model_path,
        min_face_detection_confidence=s.min_face_detection_confidence,
        min_face_presence_confidence=s.min_face_presence_confidence,
        secondary_face_area_ratio=s.secondary_face_area_ratio,
    )
    parser = OnnxFaceParser(
        model_path=s.resolved_face_parser_path,
        expected_sha256=s.face_parser_sha256,
        crop_scale=crop_scale,
        max_concurrency=1,
        queue_timeout_s=s.face_parser_queue_timeout_s,
        intra_op_threads=s.face_parser_threads if threads is None else threads,
    )
    return landmarks, parser


# --- landmark polygons (for the landmark-vs-parser comparison) -----------------------------------

def landmark_polygons(pts: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Eye and outer-lip polygons from landmark pixels (same index lists the zone engine uses)."""
    # right-eye index lists are mirrored to the left eye by the engine's own mirror map
    frame = FaceFrame.from_landmarks(pts)
    eyes = [frame.pts[frame.indices(dz.EYE_UPPER + dz.EYE_LOWER[::-1][1:-1], side)] for side in ("right", "left")]
    mouth = np.vstack([pts[dz.LIP_OUTER_UPPER], pts[dz.LIP_OUTER_LOWER[::-1][1:-1]]])
    return {"eye": eyes, "mouth": [mouth]}


def polygon_counts(polys: list[np.ndarray], gt_mask: np.ndarray) -> metrics.Counts:
    shape = gt_mask.shape
    m = np.zeros(shape, dtype=bool)
    for p in polys:
        m |= mk.rasterize_polygon(p, shape)
    return metrics.counts_from_masks(gt_mask, m)


# --- one image -----------------------------------------------------------------------------------

def _counts_dict(c: metrics.Counts) -> list[int]:
    return [c.tp, c.fp, c.fn]


def gt_summary(gt: np.ndarray, rgb: np.ndarray) -> dict[str, Any]:
    iod = metrics.inter_ocular_distance(ep_mask(gt, (EP_LEFT_EYE,)), ep_mask(gt, (EP_RIGHT_EYE,)))
    luma = metrics.mean_luma(rgb, gt == EP_SKIN)
    return {
        "gt_counts": np.bincount(gt.ravel(), minlength=9)[:9].tolist(),
        "gt_iod": iod,
        "gt_skin_luma": luma,
        "order_ok": {
            "eye": metrics.side_order_ok(gt == EP_LEFT_EYE, gt == EP_RIGHT_EYE),
            "brow": metrics.side_order_ok(gt == EP_LEFT_BROW, gt == EP_RIGHT_BROW),
        },
    }


def small_region_metrics(gt: np.ndarray, pred_full: np.ndarray, iod: float | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for comp in COMPARISONS:
        if not comp.small or comp.name in ("eye_R", "eye_L", "brow_R", "brow_L"):
            continue
        g, p = ep_mask(gt, comp.ep_ids), pred_mask(pred_full, comp.canonical)
        entry: dict[str, Any] = {}
        if iod:
            entry["bf"] = {str(t): metrics.boundary_f(g, p, t * iod) for t in TAUS}
        shift = metrics.best_shift(g, p) if g.sum() >= metrics.MIN_PIXELS and p.sum() >= metrics.MIN_PIXELS else None
        entry["shift"] = list(shift) if shift else None
        out[comp.name] = entry
    return out


def evaluate_image(
    image_bytes: bytes, annotation_bytes: bytes, landmark_service: FaceLandmarkService, parser: FaceParser
) -> dict[str, Any]:
    """Row for one sample. Raises InvalidSample if the sample itself is unusable."""
    t0 = time.perf_counter()
    try:
        rgb = load_rgb_image(image_bytes)
    except ImageValidationError as exc:
        raise InvalidSample("image_undecodable") from exc
    decode_ms = (time.perf_counter() - t0) * 1000
    h, w = rgb.shape[:2]
    gt = decode_annotation(annotation_bytes, (h, w))
    row: dict[str, Any] = {"height": h, "width": w, "decode_ms": decode_ms, **gt_summary(gt, rgb)}

    t1 = time.perf_counter()
    try:
        resp = segment_face(rgb, landmark_service, parser)
    except Exception as exc:  # noqa: BLE001 - any failure of the pipeline is an outcome, not a crash
        row.update(outcome="pipeline_error", error_type=type(exc).__name__, wall_ms=(time.perf_counter() - t1) * 1000)
        row["conf_full"] = metrics.not_predicted_confusion(gt).tolist()
        row["conf_win"] = None
        return row
    row["wall_ms"] = (time.perf_counter() - t1) * 1000
    row["timings_ms"] = {k: float(v) for k, v in resp.timings_ms.items()}
    outcome = STATUS_TO_OUTCOME.get(resp.status, "pipeline_error")
    row["outcome"] = outcome
    if outcome != "ok" or resp.anatomical is None:
        if outcome == "ok":
            row["outcome"] = "pipeline_error"
            row["error_type"] = "MissingAnatomicalLayer"
        row["conf_full"] = metrics.not_predicted_confusion(gt).tolist()
        row["conf_win"] = None
        return row

    a = resp.anatomical
    labels = mk.decode_png(base64.b64decode(a.label_map_png_base64))
    if labels.shape != (a.height, a.width):
        raise InvalidSample("label_map_shape")  # would be a pipeline contract break; never expected
    ox, oy = a.origin.x, a.origin.y
    pred = np.full((h, w), NOT_PARSED, dtype=np.uint8)
    pred[oy : oy + a.height, ox : ox + a.width] = labels
    rect = (ox, oy, ox + a.width, oy + a.height)
    row["rect"] = list(rect)
    row["window"] = [resp.parse_window.x, resp.parse_window.y, resp.parse_window.size] if resp.parse_window else None
    row["conf_full"] = metrics.confusion_matrix(gt, pred).tolist()
    row["conf_win"] = metrics.confusion_matrix(gt, pred, rect).tolist()
    row["warnings"] = sorted({wn.code for wn in resp.warnings})
    row["alignment_passed"] = bool(resp.alignment_check.passed) if resp.alignment_check else None
    row["small"] = small_region_metrics(gt, pred, row["gt_iod"])

    lm = resp.landmarks.landmarks
    if len(lm) >= 478:
        pts = np.array([[p.x * w, p.y * h] for p in sorted(lm, key=lambda p: p.index)], dtype=np.float64)
        polys = landmark_polygons(pts)
        row["landmark_polygons"] = {
            "eye": _counts_dict(polygon_counts(polys["eye"], ep_mask(gt, (EP_LEFT_EYE, EP_RIGHT_EYE)))),
            "mouth": _counts_dict(polygon_counts(polys["mouth"], ep_mask(gt, (EP_LIPS, EP_TEETH)))),
        }
    return row


# --- environment and fingerprints ------------------------------------------------------------------

def source_fingerprint() -> str:
    """SHA-256 over the relative paths and contents of backend/app and backend/evaluation Python files."""
    h = hashlib.sha256()
    for root in ("app", "evaluation"):
        for p in sorted((BACKEND_DIR / root).rglob("*.py")):
            h.update(str(p.relative_to(BACKEND_DIR)).encode())
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def _cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def environment(crop_scale: float, parser: FaceParser) -> dict[str, Any]:
    s = get_settings()
    pkgs = {}
    for name in ("numpy", "opencv-python", "opencv-python-headless", "onnxruntime", "mediapipe", "Pillow"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {
        "python": platform.python_version(),
        "packages": pkgs,
        "os": platform.platform(),  # no hostname, no user name, no paths
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "parser_id": parser.info.id,
        "parser_weights_sha256": parser.info.weights_sha256,
        "landmarker_sha256": _file_sha256(s.resolved_model_path),
        "crop_scale": crop_scale,
        "parser_threads_setting": s.face_parser_threads,
        "opencv_threads": cv2.getNumThreads(),
        "source_fingerprint_sha256": source_fingerprint(),
        "git": "not a git repository: no commit hash available",
    }


def peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1e6 if sys.platform == "darwin" else ru / 1e3  # bytes on macOS, KB on Linux


# --- the run -----------------------------------------------------------------------------------------

@dataclass
class RunSummary:
    n: int
    outcomes: dict[str, int]
    elapsed_s: float
    peak_rss_mb: float
    stopped_early: bool


def run_evaluation(
    entries: list[SampleEntry],
    cache: Path,
    out_dir: Path,
    crop_scale: float = 2.0,
    warmups: int = 3,
    limit: int | None = None,
    services: tuple[FaceLandmarkService, FaceParser] | None = None,
    progress_every: int = 25,
) -> RunSummary:
    out_dir = refuse_inside_repo(out_dir, "output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)
    entries = entries[:limit] if limit else entries
    landmark_service, parser = services or build_services(crop_scale)
    own_services = services is None
    try:
        # warm-ups on the first readable sample (not recorded)
        done_warm = 0
        for e in entries:
            if done_warm >= warmups:
                break
            try:
                rgb = load_rgb_image(image_path(cache, e.image_id).read_bytes())
                segment_face(rgb, landmark_service, parser)
                done_warm += 1
            except Exception:  # noqa: BLE001
                continue

        outcomes = {k: 0 for k in OUTCOMES}
        stopped = False
        path = out_dir / "per_image.jsonl"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        t_start = time.perf_counter()
        with os.fdopen(fd, "w") as f:
            for k, e in enumerate(entries, 1):
                try:
                    row = evaluate_image(
                        image_path(cache, e.image_id).read_bytes(),
                        annotation_path(cache, e.image_id).read_bytes(),
                        landmark_service,
                        parser,
                    )
                except InvalidSample as exc:
                    row = {"outcome": "invalid_sample", "reason": exc.reason}
                except OSError:
                    row = {"outcome": "invalid_sample", "reason": "missing_file"}
                row["id"] = e.image_id
                outcomes[row["outcome"]] += 1
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
                f.flush()
                if k % progress_every == 0 or k == len(entries):
                    logger.info("run %d/%d %s", k, len(entries), outcomes)
                if outcomes["invalid_sample"] > MAX_INVALID_FRACTION * len(entries):
                    logger.error("invalid samples exceed %.0f%%: stopping", MAX_INVALID_FRACTION * 100)
                    stopped = True
                    break
        elapsed = time.perf_counter() - t_start
        summary = RunSummary(sum(outcomes.values()), outcomes, elapsed, peak_rss_mb(), stopped)
        meta = {
            "environment": environment(crop_scale, parser),
            "summary": summary.__dict__,
            "n_sample": len(entries),
            "warmups": done_warm,
        }
        (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
        os.chmod(out_dir / "run_meta.json", 0o600)
        return summary
    finally:
        if own_services:
            landmark_service.close()
