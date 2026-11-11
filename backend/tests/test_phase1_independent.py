"""Independent Phase 1 verification of the makeup-aware facial segmentation feature.

Written by the Testing Agent, separate from the implementation. Uses real MediaPipe
and the real BiSeNet ONNX weights (tests skip if they are missing), the portrait
fixture, and synthetic variants generated in memory. Nothing is downloaded and no
image is written to disk.

Defects found in iteration 1 (BUG-1 mirrored left/right, BUG-2 unreadable weights, BUG-3
leaked landmarker) are fixed and their tests are now ordinary tests.
"""

from __future__ import annotations

import base64
import io
import math
import os
import shutil
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from fastapi.testclient import TestClient
from mediapipe.tasks.python import vision
from PIL import Image, ImageEnhance, ImageOps

from app import main as app_main
from app.api.facial_analysis import get_landmark_service
from app.api.face_segmentation import get_face_parser
from app.config import get_settings
from app.main import create_app
from app.schemas.face_segmentation import LICENSE_WARNING
from app.services.facial_analysis import landmarks as landmarks_module
from app.services.facial_analysis.landmarks import select_primary_face
from app.services.segmentation import face_parser as face_parser_module
from app.services.segmentation import masks as mk
from app.services.segmentation.face_parser import (
    FaceParserBusyError,
    IMAGENET_MEAN,
    IMAGENET_STD,
    OnnxFaceParser,
    ParserInfo,
    compute_parse_window,
    covered_region,
)

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
FIX = BACKEND / "tests" / "fixtures" / "portrait.jpg"
SEG = "/api/facial-analysis/segmentation"
LMK = "/api/facial-analysis/landmarks"
SETTINGS = get_settings()
MODELS_PRESENT = SETTINGS.resolved_model_path.is_file() and SETTINGS.resolved_face_parser_path.is_file()
needs_models = pytest.mark.skipif(not MODELS_PRESENT, reason="MediaPipe / BiSeNet weights not downloaded")

FIXTURE_BYTES = FIX.read_bytes()
BASE = Image.open(FIX).convert("RGB")
W0, H0 = BASE.size  # 820 x 1024

# Independent copy of the canonical vocabulary (name -> label-map id).
CANON = {
    "background": 0, "skin": 1, "nose": 2, "brow_l": 3, "brow_r": 4, "eye_l": 5, "eye_r": 6,
    "lip_upper": 7, "lip_lower": 8, "mouth_interior": 9, "ear_l": 10, "ear_r": 11, "hair": 12,
    "neck": 13, "cloth": 14, "hat": 15, "glasses": 16, "earring": 17, "necklace": 18,
}
VALID_LABELS = set(CANON.values()) | {255}
PHASE1_ZONES = {
    "face_skin", "forehead", "cheek_l", "cheek_r", "nose", "chin", "jawline", "under_eye_l", "under_eye_r",
    "eye_l", "eye_r", "brow_l", "brow_r", "lip_upper", "lip_lower", "lip_full", "lip_outer_boundary",
    "lip_inner_boundary",
    "chin_center", "under_lower_lip", "mouth_surround", "cupid_bow", "lip_corner_l", "lip_corner_r",
    "forehead_center", "forehead_perimeter", "temple_l", "temple_r", "jawline_outer_l", "jawline_outer_r",
    "under_eye_inner_l", "under_eye_inner_r", "under_eye_outer_l", "under_eye_outer_r",
    "eye_corner_inner_l", "eye_corner_inner_r", "eye_corner_outer_l", "eye_corner_outer_r",
    "lash_line_upper_l", "lash_line_upper_r", "lash_line_lower_l", "lash_line_lower_r",
    "eyelid_upper_l", "eyelid_upper_r", "brow_head_l", "brow_head_r", "brow_arch_l", "brow_arch_r",
    "brow_tail_l", "brow_tail_r", "brow_bone_l", "brow_bone_r",
    "cheek_apple_l", "cheek_apple_r", "cheek_upper_l", "cheek_upper_r", "cheek_outer_l", "cheek_outer_r",
    "cheekbone_l", "cheekbone_r", "contour_hollow_l", "contour_hollow_r",
    "nose_bridge", "nose_tip", "nose_side_l", "nose_side_r", "nostril_l", "nostril_r",
    "lower_eyelid_l", "lower_eyelid_r",
}
# Phase 2 zones from the requirements / plan. Must not appear anywhere in Phase 1 output.
PHASE2_ZONES = {
    "cheek_hollow_l", "cheek_hollow_r",
    "nose_crease_l", "nose_crease_r",
    "cupids_bow",
    "eyelid_lower_l", "eyelid_lower_r",
}
MODEL_ONLY_ZONES = {"eye_r", "eye_l", "brow_r", "brow_l"}
LANDMARK_EYE_ZONES = [f"{base}_{k}" for base in ("eye_corner_inner", "eye_corner_outer",
                                                  "lash_line_upper", "lash_line_lower") for k in "lr"]
WARNING_CODES = {
    "face_partially_outside_image", "low_resolution_face", "head_tilt", "glasses_detected",
    "parser_landmark_mismatch", "low_overlap_zone", "hair_occludes_forehead", "background_face_ignored",
}
ZONE_KEYS = {"id", "name", "group", "side", "sources", "status", "reason", "confidence", "coverage",
             "clipped_to", "bbox", "pixel_count", "mask", "polygon"}
# Clip regions as the zone spec declares them (canonical names -> label ids).
CLIP_IDS = {"skin": {1}, "nose": {2}, "eye_l": {5}, "eye_r": {6}, "brow_l": {3}, "brow_r": {4}}
# MediaPipe's own connection tables, used as the authoritative landmark groups.
_C = vision.FaceLandmarksConnections


def _idx(conns) -> list[int]:
    return sorted({c.start for c in conns} | {c.end for c in conns})


EYE_R_IDX = _idx(_C.FACE_LANDMARKS_RIGHT_EYE)
EYE_L_IDX = _idx(_C.FACE_LANDMARKS_LEFT_EYE)
BROW_R_IDX = _idx(_C.FACE_LANDMARKS_RIGHT_EYEBROW)
BROW_L_IDX = _idx(_C.FACE_LANDMARKS_LEFT_EYEBROW)
LIPS_IDX = _idx(_C.FACE_LANDMARKS_LIPS)


# --- image helpers (all in memory) ------------------------------------------------

def png(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def jpg(img: Image.Image, q: int = 95) -> bytes:
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=q)
    return b.getvalue()


def pad(img: Image.Image, left: int, right: int, top: int, bottom: int) -> Image.Image:
    c = Image.new("RGB", (img.width + left + right, img.height + top + bottom), (128, 128, 128))
    c.paste(img, (left, top))
    return c


def two_side_by_side(a: Image.Image, b: Image.Image, gap: int = 0) -> Image.Image:
    c = Image.new("RGB", (a.width + b.width + gap, max(a.height, b.height)), (128, 128, 128))
    c.paste(a, (0, 0))
    c.paste(b, (a.width + gap, 0))
    return c


def bad_png_header(width: int, height: int) -> bytes:
    """A PNG whose header claims a huge bitmap, with a tiny compressed IDAT that holds almost nothing."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def bomb_jpeg(width: int, height: int) -> bytes:
    """The real fixture with its SOF0 header rewritten to claim a huge frame."""
    data = bytearray(FIXTURE_BYTES)
    i = data.find(b"\xff\xc2")  # the fixture is progressive (SOF2)
    if i < 0:
        i = data.find(b"\xff\xc0")
    assert i > 0
    data[i + 5 : i + 7] = struct.pack(">H", height)
    data[i + 7 : i + 9] = struct.pack(">H", width)
    return bytes(data)


def rotated(angle: float) -> bytes:
    return png(BASE.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0)))


def exif_rotated_jpeg() -> bytes:
    """Stored sideways with EXIF orientation 6; displays upright, like a phone photo."""
    stored = BASE.rotate(90, expand=True)
    exif = stored.getexif()
    exif[0x0112] = 6
    b = io.BytesIO()
    stored.save(b, format="JPEG", quality=95, exif=exif)
    return b.getvalue()


# --- client helpers ---------------------------------------------------------------

def post(client: TestClient, url: str, data: bytes, ctype: str = "image/png", name: str = "upload.bin"):
    return client.post(url, files={"file": (name, data, ctype)})


def decode(b64: str) -> np.ndarray:
    return mk.decode_png(base64.b64decode(b64))


def label_map(body: dict) -> tuple[np.ndarray, tuple[int, int]]:
    a = body["anatomical"]
    return decode(a["label_map_png_base64"]), (a["origin"]["x"], a["origin"]["y"])


def full_labels(body: dict) -> np.ndarray:
    """Label map placed in full image coordinates; 255 outside the covered area."""
    lab, (ox, oy) = label_map(body)
    full = np.full((body["image"]["height"], body["image"]["width"]), 255, dtype=np.uint8)
    full[oy : oy + lab.shape[0], ox : ox + lab.shape[1]] = lab
    return full


def landmark_px(body: dict) -> dict[int, tuple[float, float]]:
    w, h = body["image"]["width"], body["image"]["height"]
    return {p["index"]: (p["x"] * w, p["y"] * h) for p in body["landmarks"]["landmarks"]}


def zone_full_mask(body: dict, zid: str) -> np.ndarray:
    """Zone mask in full image coordinates (bool, H x W)."""
    z = next(z for z in body["zones"] if z["id"] == zid)
    out = np.zeros((body["image"]["height"], body["image"]["width"]), dtype=bool)
    b = z["bbox"]
    m = decode(z["mask"]["png_base64"]) == 255
    out[b["y"] : b["y"] + b["height"], b["x"] : b["x"] + b["width"]] = m
    return out


def class_at(labels_full: np.ndarray, px: tuple[float, float]) -> int:
    x, y = int(round(px[0])), int(round(px[1]))
    return int(labels_full[y, x])


def class_near(labels_full: np.ndarray, px: tuple[float, float], r: int = 3) -> set[int]:
    x, y = int(round(px[0])), int(round(px[1]))
    h, w = labels_full.shape
    win = labels_full[max(y - r, 0) : min(y + r + 1, h), max(x - r, 0) : min(x + r + 1, w)]
    return {int(v) for v in np.unique(win)}


def check_invariants(body: dict) -> None:
    """Properties that must hold for every response, whatever the input.

    Warnings are checked against the landmarks and the label map in the same response,
    so an honest response passes and a silent one fails.
    """
    assert body["model"]["license_warning"] == LICENSE_WARNING
    assert body["status"] in {"ok", "no_face_detected", "multiple_faces_detected"}
    codes = {w["code"] for w in body["warnings"]}
    assert codes <= WARNING_CODES, codes - WARNING_CODES
    if body["status"] != "ok":
        assert body["anatomical"] is None and body["zones"] == [] and body["alignment_check"] is None
        assert body["landmarks"]["landmarks"] == []
        return

    W, H = body["image"]["width"], body["image"]["height"]
    assert len(body["landmarks"]["landmarks"]) == 478
    lm = landmark_px(body)
    pts = np.array([lm[i] for i in range(478)])

    # alignment and warnings agree with each other
    al = body["alignment_check"]
    assert 0.0 <= al["fraction"] <= 1.0
    # Iris-in-eye is a hard requirement: passed needs BOTH irises in their eye region AND >= 0.6.
    full = full_labels(body)
    irises_ok = (CANON["eye_r"] in class_near(full, lm[468], 1)) and (CANON["eye_l"] in class_near(full, lm[473], 1))
    assert al["passed"] == (irises_ok and al["fraction"] >= 0.6)
    assert ("parser_landmark_mismatch" in codes) == (not al["passed"])
    assert ("background_face_ignored" in codes) == (body["landmarks"]["face_count"] > 1)

    # geometric warnings: bands are generous where the plan gives only "about"
    outside = (pts[:, 0] < -2) | (pts[:, 0] > W + 2) | (pts[:, 1] < -2) | (pts[:, 1] > H + 2)
    if outside.any():
        assert "face_partially_outside_image" in codes
    if (pts[:, 0] >= 0).all() and (pts[:, 1] >= 0).all() and (pts[:, 0] <= W).all() and (pts[:, 1] <= H).all():
        assert "face_partially_outside_image" not in codes
    d = np.array(lm[473]) - np.array(lm[468])
    ipd = float(np.hypot(*d))
    roll = math.degrees(math.atan2(d[1], d[0]))
    if abs(roll) > 28:
        assert "head_tilt" in codes, roll
    if abs(roll) < 22:
        assert "head_tilt" not in codes, roll
    if ipd < 20:
        assert "low_resolution_face" in codes, ipd
    if ipd > 60:
        assert "low_resolution_face" not in codes, ipd
    full = full_labels(body)
    glasses_frac = np.count_nonzero(full == CANON["glasses"]) / full.size
    assert ("glasses_detected" in codes) == (glasses_frac >= 0.01)

    # anatomical label map
    lab, (ox, oy) = label_map(body)
    a = body["anatomical"]
    assert lab.dtype == np.uint8 and lab.shape == (a["height"], a["width"])
    assert set(np.unique(lab).tolist()) <= VALID_LABELS
    assert 0 <= ox and ox + a["width"] <= W and 0 <= oy and oy + a["height"] <= H
    pw = body["parse_window"]
    assert (ox, oy) == (max(pw["x"], 0), max(pw["y"], 0))
    assert a["width"] == min(pw["x"] + pw["size"], W) - ox and a["height"] == min(pw["y"] + pw["size"], H) - oy
    assert a["not_parsed_value"] == 255
    assert {r["name"]: r["id"] for r in a["regions"]} == CANON
    present = {r["id"] for r in a["regions"] if r["present"]}
    assert present == set(np.unique(lab).tolist()) - {255}
    assert sum(r["pixel_count"] for r in a["regions"]) == lab.size

    # zones
    zones = {z["id"]: z for z in body["zones"]}
    assert set(zones) <= PHASE1_ZONES and not (set(zones) & PHASE2_ZONES)
    for z in body["zones"]:
        assert set(z) == ZONE_KEYS, z["id"]
        assert z["status"] in {"ok", "low_confidence", "unavailable"}
        assert z["sources"] and set(z["sources"]) <= {"segmentation", "landmarks"}
        if z["id"].endswith("_l"):
            assert z["side"] == "left", z["id"]
        elif z["id"].endswith("_r"):
            assert z["side"] == "right", z["id"]
        else:
            assert z["side"] is None, z["id"]
        if z["confidence"] is not None:
            assert 0.0 <= z["confidence"] <= 1.0
        if z["coverage"] is not None:
            assert 0.0 <= z["coverage"] <= 1.0
        if z["status"] == "unavailable":
            assert z["mask"] is None and z["bbox"] is None and z["pixel_count"] == 0 and z["reason"], z["id"]
            continue
        assert z["mask"] is not None and z["bbox"] is not None, z["id"]
        b = z["bbox"]
        assert b["width"] >= 1 and b["height"] >= 1
        assert 0 <= b["x"] and b["x"] + b["width"] <= W and 0 <= b["y"] and b["y"] + b["height"] <= H, z["id"]
        m = decode(z["mask"]["png_base64"])
        assert m.shape == (b["height"], b["width"]) and set(np.unique(m).tolist()) <= {0, 255}, z["id"]
        assert int(np.count_nonzero(m == 255)) == z["pixel_count"] >= 16, z["id"]  # never an empty "ok" mask
        if z["status"] == "ok":
            assert z["reason"] is None, z["id"]
        else:
            assert z["reason"], z["id"]
        if z["coverage"] is not None and z["coverage"] < 0.5 and z["status"] == "low_confidence":
            assert "low_overlap_zone" in codes, z["id"]
        if z["polygon"] is not None:
            assert all(len(p) == 2 for p in z["polygon"])

    # stages: every referenced zone exists in the response
    assert [s["number"] for s in body["stages"]] == list(range(1, 12))
    referenced = {zid for s in body["stages"] for ids in s["layers"].values() for zid in ids}
    assert referenced <= set(zones), referenced - set(zones)


# --- fixtures ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Real app: MediaPipe + BiSeNet loaded through the normal lifespan, once per module."""
    if not MODELS_PRESENT:
        pytest.skip("MediaPipe / BiSeNet weights not downloaded")
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture(scope="module")
def parser():
    if not MODELS_PRESENT:
        pytest.skip("weights not downloaded")
    return OnnxFaceParser(SETTINGS.resolved_face_parser_path, SETTINGS.face_parser_sha256)


@pytest.fixture(scope="module")
def seg_fixture(client):
    return post(client, SEG, FIXTURE_BYTES, "image/jpeg").json()


# --- 1. contract, schema, documentation -------------------------------------------

@needs_models
def test_ac1_license_warning_in_every_response_and_docs(client):
    body = post(client, SEG, FIXTURE_BYTES, "image/jpeg").json()
    assert body["model"]["license_warning"] == LICENSE_WARNING
    assert "non-commercial" in LICENSE_WARNING and "Not cleared for commercial use" in LICENSE_WARNING
    readme = (REPO / "README.md").read_text()
    docs = (REPO / "docs" / "face_segmentation.md").read_text()
    assert "CelebAMask-HQ" in readme and "non-commercial" in readme.lower()
    assert "CelebAMask-HQ" in docs and "non-commercial" in docs.lower()
    assert "19" in docs and "BiSeNet" in docs


@needs_models
def test_ac1_license_warning_on_non_ok_responses(client):
    app = create_app(use_lifespan=False)
    app.dependency_overrides[get_landmark_service] = lambda: _FakeLandmarks("no_face_detected")
    app.dependency_overrides[get_face_parser] = lambda: client.app.state.face_parser
    body = post(TestClient(app), SEG, png(Image.new("RGB", (64, 64), (5, 5, 5)))).json()
    assert body["status"] == "no_face_detected" and body["model"]["license_warning"] == LICENSE_WARNING


class _FakeLandmarks:
    def __init__(self, status: str, face_count: int = 1):
        self.status, self.face_count = status, face_count

    def detect_landmarks(self, image):
        from app.schemas.facial_landmarks import DetectionThresholds, FacialLandmarkResult, ImageSize

        h, w = image.shape[:2]
        return FacialLandmarkResult(
            face_detected=self.status != "no_face_detected",
            status=self.status,
            message="Please upload a photo with only one face." if self.status == "multiple_faces_detected" else "No face found.",
            face_count=0 if self.status == "no_face_detected" else self.face_count,
            image=ImageSize(width=w, height=h),
            thresholds=DetectionThresholds(min_face_detection_confidence=0.5, min_face_presence_confidence=0.5),
            landmark_count=0,
            landmarks=[],
        )


@needs_models
def test_ac7_fixture_has_exactly_the_expected_zones_all_ok(seg_fixture):
    check_invariants(seg_fixture)
    assert seg_fixture["status"] == "ok" and seg_fixture["alignment_check"]["passed"]
    zones = {z["id"]: z for z in seg_fixture["zones"]}
    assert set(zones) == PHASE1_ZONES
    bad = {k: (z["status"], z["reason"]) for k, z in zones.items() if z["status"] != "ok"}
    assert not bad, bad


@needs_models
def test_ac8_phase2_zones_absent_from_zones_and_stages(seg_fixture):
    ids = {z["id"] for z in seg_fixture["zones"]}
    stage_ids = {zid for s in seg_fixture["stages"] for ids_ in s["layers"].values() for zid in ids_}
    assert not (ids & PHASE2_ZONES) and not (stage_ids & PHASE2_ZONES)
    assert "eyelid_crease" not in ids


@needs_models
def test_stage_map_covers_eleven_stages_with_the_specified_layers(seg_fixture):
    st = {s["key"]: s for s in seg_fixture["stages"]}
    assert [s["number"] for s in seg_fixture["stages"]] == list(range(1, 12))
    assert len(st) == 11
    assert list(st["bronzer_contour"]["layers"]) == ["contour", "bronzer"]
    assert st["bronzer_contour"]["layers"]["bronzer"] == []  # Phase 2, listed but empty
    assert st["bronzer_contour"]["layers"]["contour"] == [
        "jawline", "contour_hollow_r", "contour_hollow_l", "nose_side_r", "nose_side_l"]
    assert list(st["lips"]["layers"]) == ["lipstick", "lip_liner", "gloss"]
    assert st["lips"]["layers"]["lip_liner"] == ["lip_outer_boundary", "cupid_bow", "lip_corner_r", "lip_corner_l"]
    assert st["lips"]["layers"]["gloss"] == ["lip_full"]
    assert set(st["lips"]["layers"]["lipstick"]) >= {"lip_upper", "lip_lower", "lip_full"}
    assert st["eyebrow_eye"]["layers"] == {
        "eyebrow": ["brow_r", "brow_l", "brow_head_r", "brow_arch_r", "brow_tail_r",
                    "brow_head_l", "brow_arch_l", "brow_tail_l"],
        "eye": ["eye_r", "eye_l", "eye_corner_inner_r", "eye_corner_inner_l", "eye_corner_outer_r",
                "eye_corner_outer_l", "lash_line_upper_r", "lash_line_upper_l", "lash_line_lower_r",
                "lash_line_lower_l", "eyelid_upper_r", "eyelid_upper_l",
                "lower_eyelid_r", "lower_eyelid_l"]}
    assert st["blush"]["layers"] == {"blush": ["cheek_apple_r", "cheek_apple_l", "cheek_upper_r", "cheek_upper_l",
                                               "cheek_outer_r", "cheek_outer_l"]}
    assert st["highlighter"]["layers"] == {
        "highlighter": ["brow_bone_r", "brow_bone_l", "cheekbone_r", "cheekbone_l", "nose_bridge", "nose_tip"]}


@needs_models
def test_stage_membership_against_requirement_text(seg_fixture):
    """Requirement section 7 (Phase 1 subset) must be satisfied by the stage map."""
    layers = {s["key"]: set(z for ids in s["layers"].values() for z in ids) for s in seg_fixture["stages"]}
    face = {"face_skin", "forehead", "cheek_l", "cheek_r", "nose", "chin", "jawline"}
    required = {
        "primer": face | {"under_eye_l", "under_eye_r"},
        "color_corrector": {"under_eye_l", "under_eye_r", "cheek_l", "cheek_r", "nose", "chin", "forehead"},
        "foundation": face,
        "concealer": {"under_eye_l", "under_eye_r"},
        "setting_powder": {"face_skin", "under_eye_l", "under_eye_r", "chin"},
        "bronzer_contour": {"jawline"},
        "setting_spray": face,
    }
    for key, need in required.items():
        assert need <= layers[key], (key, need - layers[key])
    assert {"brow_l", "brow_r", "eye_l", "eye_r"} <= layers["eyebrow_eye"]
    assert {"lip_upper", "lip_lower", "lip_outer_boundary", "lip_inner_boundary"} <= layers["lips"]


def test_stage_membership_against_plan_table_documents_chin_difference():
    """Plan table lists chin for stages 1, 2, 3, 11. Requirement section 7 also puts chin in
    setting powder, and the catalog follows the requirement. Recorded here, not failed."""
    from app.services.application_zones.catalog import STAGES

    membership: dict[str, set[int]] = {}
    for st in STAGES:
        for ids in st.layers.values():
            for zid in ids:
                membership.setdefault(zid, set()).add(st.number)
    assert membership["face_skin"] == {1, 3, 5, 11}
    assert membership["forehead"] == {1, 2, 3, 11}
    assert membership["cheek_l"] == membership["cheek_r"] == {1, 2, 3, 11}
    assert membership["nose"] == {1, 2, 3, 11}
    assert membership["jawline"] == {1, 3, 6, 11}
    assert membership["under_eye_l"] == {1, 2, 4, 5}
    assert membership["eye_l"] == {9} and membership["brow_r"] == {9}
    assert membership["lip_full"] == {10} and membership["lip_outer_boundary"] == {10}
    assert membership["chin"] - {1, 2, 3, 11} == {5}  # documented deviation (requirement-backed)


@needs_models
def test_response_schema_and_timings_contract(seg_fixture):
    assert set(seg_fixture) >= {"status", "message", "image", "model", "landmarks", "parse_window",
                                "anatomical", "zones", "stages", "alignment_check", "warnings", "timings_ms"}
    assert seg_fixture["model"]["input_size"] == 512 and len(seg_fixture["model"]["weights_sha256"]) == 64
    t = seg_fixture["timings_ms"]
    assert {"preprocess", "inference", "postprocess", "zones", "encode", "total", "landmarks"} <= set(t)
    assert all(v >= 0 for v in t.values())
    assert t["total"] >= max(t["inference"], t["zones"], t["encode"])
    assert seg_fixture["anatomical"]["encoding"] == "png_label_map_v1"
    assert seg_fixture["anatomical"]["not_parsed_value"] == 255
    for z in seg_fixture["zones"]:
        if z["mask"]:
            assert z["mask"]["encoding"] == "png_bbox_v1"


@needs_models
def test_landmarks_endpoint_is_backwards_compatible(client):
    body = post(client, LMK, FIXTURE_BYTES, "image/jpeg").json()
    original = {"face_detected", "status", "message", "image", "thresholds", "landmark_count", "landmarks",
                "measurements"}
    assert original <= set(body)
    assert "face_count" in body and body["face_count"] == 1
    assert body["status"] == "ok" and body["landmark_count"] == 478
    assert {"index", "x", "y", "z"} == set(body["landmarks"][0])
    assert set(body["measurements"]) >= {"face_bounding_box", "interpupillary_distance_px",
                                         "inner_eye_corner_distance_px", "left_eye_width_px",
                                         "right_eye_width_px", "mouth_width_px", "face_width_px",
                                         "face_height_px"}


def test_openapi_contract_has_new_status_and_face_count():
    app = create_app(use_lifespan=False)
    with TestClient(app) as c:
        spec = c.get("/openapi.json").json()
    assert c.get("/health").json() == {"status": "ok"}
    schemas = spec["components"]["schemas"]
    lm = schemas["FacialLandmarkResponse"]["properties"]
    assert "face_count" in lm
    assert "multiple_faces_detected" in lm["status"].get("enum", []) or "multiple_faces_detected" in str(lm["status"])
    assert "/api/facial-analysis/segmentation" in spec["paths"]
    assert "/api/facial-analysis/topology" in spec["paths"]


def test_topology_regression():
    app = create_app(use_lifespan=False)
    body = TestClient(app).get("/api/facial-analysis/topology").json()
    assert len(body["tesselation"]) > 2000
    assert set(body["contours"]) == {"face_oval", "lips", "left_eye", "right_eye", "left_eyebrow",
                                     "right_eyebrow", "left_iris", "right_iris", "nose"}


# --- 2. model loading, class mapping, alignment (AC 2, 3, 4, 6) ------------------

@needs_models
def test_ac2_weights_verified_and_pinned_hash_matches_file():
    import hashlib

    data_hash = hashlib.sha256(SETTINGS.resolved_face_parser_path.read_bytes()).hexdigest()
    assert data_hash == SETTINGS.face_parser_sha256
    assert data_hash == "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f"


@needs_models
def test_ac2_weights_loaded_once_at_startup_not_per_request(monkeypatch):
    created: list[str] = []
    real = ort.InferenceSession

    class Counting(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **kw):
            created.append("session")
            super().__init__(*a, **kw)

    monkeypatch.setattr(face_parser_module.ort, "InferenceSession", Counting)
    with TestClient(create_app()) as c:
        startup = len(created)
        for _ in range(3):
            assert post(c, SEG, FIXTURE_BYTES, "image/jpeg").status_code == 200
        assert startup == 1 and len(created) == 1
        assert c.app.state.face_parser is not None


@needs_models
def test_ac3_raw_model_class_order_and_left_right_mapping(client, parser, monkeypatch):
    """Independent check of the model's raw logits (before any mapping)."""
    lm_body = post(client, LMK, FIXTURE_BYTES, "image/jpeg").json()
    lm = {p["index"]: (p["x"] * W0, p["y"] * H0) for p in lm_body["landmarks"]}
    pts = np.array([lm[i] for i in range(478)])
    img = np.asarray(BASE, dtype=np.uint8)
    bbox = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())

    captured: dict[str, np.ndarray] = {}
    orig = parser._infer

    def spy(x):
        out = orig(x)
        captured["logits"] = out
        return out

    monkeypatch.setattr(parser, "_infer", spy)
    res = parser.parse(img, bbox)
    logits = captured["logits"]
    assert logits.shape == (19, 512, 512)  # 19 channels, logits at 512 px

    def raw_class(p):
        # crop index = (image px - window origin) * 512 / window size
        j = (p[0] - res.window.x) * 512 / res.window.size
        i = (p[1] - res.window.y) * 512 / res.window.size
        jj, ii = int(round(j)), int(round(i))
        patch = logits[:, max(ii - 2, 0) : ii + 3, max(jj - 2, 0) : jj + 3].reshape(19, -1).mean(axis=1)
        return int(np.argmax(patch))

    assert raw_class(lm[468]) == 5  # model class 5 = r_eye: subject's right iris (image left)
    assert raw_class(lm[473]) == 4  # model class 4 = l_eye: subject's left iris (image right)
    from app.services.segmentation.regions import MODEL_ID_TO_CANONICAL

    # Side-agnostic grouping: the model's l_/r_ pairs map to ONE unsided id; sides come from landmarks.
    assert MODEL_ID_TO_CANONICAL[5] is MODEL_ID_TO_CANONICAL[4]
    assert MODEL_ID_TO_CANONICAL[2] is MODEL_ID_TO_CANONICAL[3]


@needs_models
def test_ac3_irises_land_in_correct_side_eye_classes(seg_fixture):
    full = full_labels(seg_fixture)
    lm = landmark_px(seg_fixture)
    assert class_at(full, lm[468]) == CANON["eye_r"]
    assert class_at(full, lm[473]) == CANON["eye_l"]
    assert class_near(full, lm[468], 1) == {CANON["eye_r"]}
    assert class_near(full, lm[473], 1) == {CANON["eye_l"]}


@needs_models
def test_ac3_alignment_check_recomputed_independently(seg_fixture):
    """Recompute the six alignment checks from the response and compare with the reported fraction."""
    full = full_labels(seg_fixture)
    lm = landmark_px(seg_fixture)

    def near(p, allowed, r=1):
        return bool(class_near(full, p, r) & allowed)

    mid = lambda a, b: ((lm[a][0] + lm[b][0]) / 2, (lm[a][1] + lm[b][1]) / 2)
    checks = [
        near(lm[468], {CANON["eye_r"]}),
        near(lm[473], {CANON["eye_l"]}),
        near(lm[1], {CANON["nose"], CANON["skin"]}),
        near(lm[13], {CANON["mouth_interior"], CANON["lip_upper"], CANON["lip_lower"]}),
        near(mid(105, 52), {CANON["brow_r"]}),
        near(mid(334, 282), {CANON["brow_l"]}),
    ]
    assert seg_fixture["alignment_check"]["fraction"] == round(sum(checks) / 6, 3) == 1.0


@needs_models
def test_ac3_eye_brow_lip_landmarks_sit_on_their_classes(seg_fixture):
    """Brow and lip contours sit within 3 px of their class. Eye contours within 6 px: the parser
    eye mask stops about 5 px short of the inner canthus (measured 5.0-5.6 px for landmarks 133, 155)."""
    full = full_labels(seg_fixture)
    lm = landmark_px(seg_fixture)
    for idx, cls, r in ((EYE_R_IDX, CANON["eye_r"], 6), (EYE_L_IDX, CANON["eye_l"], 6),
                        (BROW_R_IDX, CANON["brow_r"], 3), (BROW_L_IDX, CANON["brow_l"], 3)):
        miss = [i for i in idx if cls not in class_near(full, lm[i], r)]
        assert not miss, (cls, miss)
    inner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324]
    lip_ok = {CANON["lip_upper"], CANON["lip_lower"], CANON["mouth_interior"]}
    miss = [i for i in inner if not class_near(full, lm[i], 3) & lip_ok]
    assert not miss, miss
    outer = [i for i in LIPS_IDX if i not in inner]
    miss = [i for i in outer if not class_near(full, lm[i], 3) & (lip_ok | {CANON["skin"]})]
    assert not miss, miss


@needs_models
def test_ac3_nose_tip_is_in_nose_class(seg_fixture):
    full = full_labels(seg_fixture)
    lm = landmark_px(seg_fixture)
    assert class_at(full, lm[1]) == CANON["nose"]


@needs_models
@pytest.mark.parametrize("name,data,ctype", [
    ("exif_jpeg", "exif", "image/jpeg"),
    ("rotate_15", "rot15", "image/png"),
    ("rotate_minus_15", "rot-15", "image/png"),
    ("half_scale", "half", "image/png"),
    ("upscale_5_4", "up5.4", "image/jpeg"),
])
def test_ac3_irises_correct_on_transformed_inputs(client, name, data, ctype):
    payload = {
        "exif": exif_rotated_jpeg, "rot15": lambda: rotated(15), "rot-15": lambda: rotated(-15),
        "half": lambda: png(BASE.resize((W0 // 2, H0 // 2), Image.LANCZOS)),
        "up5.4": lambda: jpg(BASE.resize((round(W0 * 5.4), round(H0 * 5.4)), Image.BICUBIC), 85),
    }[data]()
    body = post(client, SEG, payload, ctype).json()
    assert body["status"] == "ok", name
    check_invariants(body)
    assert body["alignment_check"]["passed"], name
    full = full_labels(body)
    lm = landmark_px(body)
    assert class_at(full, lm[468]) == CANON["eye_r"], name
    assert class_at(full, lm[473]) == CANON["eye_l"], name


@needs_models
def test_ac4_landmarks_and_label_map_share_coordinates_after_exif(client):
    """EXIF-rotated upload must give the same landmarks as the upright file (same coordinate space)."""
    a = post(client, LMK, exif_rotated_jpeg(), "image/jpeg").json()
    b = post(client, LMK, FIXTURE_BYTES, "image/jpeg").json()
    assert (a["image"]["width"], a["image"]["height"]) == (b["image"]["width"], b["image"]["height"]) == (W0, H0)

    def px(body):  # /landmarks has a flat landmark list (no wrapper object)
        return {p["index"]: (p["x"] * W0, p["y"] * H0) for p in body["landmarks"]}

    pa, pb = px(a), px(b)
    d = np.array([np.hypot(pa[i][0] - pb[i][0], pa[i][1] - pb[i][1]) for i in range(478)])
    assert np.median(d) <= 1.0 and d.max() <= 2.0, (float(np.median(d)), float(d.max()))


# Stub model: class "skin" where the crop is bright, "background" elsewhere. Lets the
# crop / warp / inverse-map path be checked against exact pixel geometry.
class BrightnessStub(OnnxFaceParser):
    def __init__(self, crop_scale: float = 2.0) -> None:  # noqa: D401 - no weights needed
        self.info = ParserInfo("stub", "stub", "0" * 64, 512)
        self._init_pipeline(crop_scale, 1, 1.0)

    def _infer(self, x: np.ndarray) -> np.ndarray:
        pix = x[0].transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
        bright = pix.mean(axis=2) > 128
        out = np.zeros((19, 512, 512), dtype=np.float32)
        out[1] = np.where(bright, 10.0, 0.0)
        out[0] = np.where(bright, 0.0, 10.0)
        return out


@pytest.mark.parametrize("side", [1000, 1500, 2400, 3200])
def test_ac4_stub_label_map_aligns_with_image_within_1px(side):
    """Reconstructed full-size map must match a known rectangle within 1 px. Several
    sub-pixel phases and window sizes, including windows above the 1024 px cap."""
    parser = BrightnessStub()
    worst = 0
    for phase in (0, 3, 5):
        S = side  # window side
        W = H = S + 200
        cx, cy = W // 2 + phase, H // 2 - phase
        bbox_side = S // 2
        img = np.zeros((H, W, 3), np.uint8)
        q = S // 8
        rx0, ry0, rx1, ry1 = cx - q, cy - q, cx + q, cy + q
        img[ry0:ry1, rx0:rx1] = 255
        bbox = (cx - bbox_side / 2, cy - bbox_side / 2, cx + bbox_side / 2, cy + bbox_side / 2)
        res = parser.parse(img, bbox)
        full = np.full((H, W), 255, np.uint8)
        ox, oy = res.origin
        full[oy : oy + res.labels.shape[0], ox : ox + res.labels.shape[1]] = res.labels
        sk = full == CANON["skin"]
        xs, ys = np.nonzero(sk.any(axis=0))[0], np.nonzero(sk.any(axis=1))[0]
        errs = [xs.min() - rx0, xs.max() - (rx1 - 1), ys.min() - ry0, ys.max() - (ry1 - 1)]
        worst = max(worst, max(abs(e) for e in errs))
    assert worst <= 1, worst


@pytest.mark.parametrize("rect", [(60, 80, 200, 250), (0, 0, 120, 140)])
def test_ac4_stub_window_sticking_out_of_image_aligns(rect):
    """Face at or past the image edge: the window is padded and the map covers only window ∩ image."""
    parser = BrightnessStub()
    W, H = 700, 600
    rx0, ry0, rx1, ry1 = rect
    img = np.zeros((H, W, 3), np.uint8)
    img[ry0:ry1, rx0:rx1] = 255
    bbox = (rx0 - 40, ry0 - 40, rx1 + 40, ry1 + 40)
    res = parser.parse(img, bbox)
    full = np.full((H, W), 255, np.uint8)
    ox, oy = res.origin
    full[oy : oy + res.labels.shape[0], ox : ox + res.labels.shape[1]] = res.labels
    sk = full == CANON["skin"]
    xs, ys = np.nonzero(sk.any(axis=0))[0], np.nonzero(sk.any(axis=1))[0]
    assert abs(xs.min() - rx0) <= 1 and abs(xs.max() - (rx1 - 1)) <= 1
    assert abs(ys.min() - ry0) <= 1 and abs(ys.max() - (ry1 - 1)) <= 1


# --- 3. left / right distinguishability (AC 6) ------------------------------------

@needs_models
def test_ac6_left_right_zones_are_on_the_subjects_sides(seg_fixture):
    lm = landmark_px(seg_fixture)
    nose_x = lm[1][0]
    for zid in ("cheek_l", "under_eye_l", "eye_l", "brow_l"):
        ys, xs = np.nonzero(zone_full_mask(seg_fixture, zid))
        assert xs.mean() > nose_x, zid  # subject's left = image right
    for zid in ("cheek_r", "under_eye_r", "eye_r", "brow_r"):
        ys, xs = np.nonzero(zone_full_mask(seg_fixture, zid))
        assert xs.mean() < nose_x, zid


@needs_models
def test_ac6_side_fields_and_names_are_consistent(seg_fixture):
    regions = {r["name"]: r for r in seg_fixture["anatomical"]["regions"]}
    assert regions["eye_l"]["side"] == "left" and regions["eye_r"]["side"] == "right"
    assert regions["brow_l"]["side"] == "left" and regions["brow_r"]["side"] == "right"
    zones = {z["id"]: z for z in seg_fixture["zones"]}
    for zid in ("cheek_l", "under_eye_l", "eye_l", "brow_l"):
        assert zones[zid]["side"] == "left" and zones[zid]["name"].startswith("Left"), zid
    for zid in ("cheek_r", "under_eye_r", "eye_r", "brow_r"):
        assert zones[zid]["side"] == "right" and zones[zid]["name"].startswith("Right"), zid
    for zid in ("face_skin", "forehead", "chin", "jawline", "nose", "lip_full"):
        assert zones[zid]["side"] is None


def _mirrored_body(client):
    return post(client, SEG, png(ImageOps.mirror(BASE))).json()


def _zone_full(body, zid):
    return zone_full_mask(body, zid)


@needs_models
def test_ac6_mirrored_selfie_eyes_get_different_classes(client):
    body = _mirrored_body(client)
    full = full_labels(body)
    lm = landmark_px(body)
    assert class_at(full, lm[468]) == CANON["eye_r"]
    assert class_at(full, lm[473]) == CANON["eye_l"]
    assert body["alignment_check"]["passed"] is True


@needs_models
def test_ac6_mirrored_selfie_ok_eye_zone_does_not_cover_the_other_eye(client):
    body = _mirrored_body(client)
    lm = landmark_px(body)
    zones = {z["id"]: z for z in body["zones"]}
    for side, own, other in (("l", 473, 468), ("r", 468, 473)):
        z = zones[f"eye_{side}"]
        assert z["status"] == "ok", z
        m = zone_full_mask(body, f"eye_{side}")
        assert m[int(round(lm[own][1])), int(round(lm[own][0]))], f"eye_{side} ok but misses its iris"
        assert not m[int(round(lm[other][1])), int(round(lm[other][0]))], f"eye_{side} ok but covers the other eye"


@needs_models
def test_mirrored_selfie_each_eye_and_brow_covers_exactly_one_iris_on_its_own_side(client):
    """Eyes AND brows: each side's mask sits on its own side of the roll-aware midline (10-152)."""
    body = _mirrored_body(client)
    lm = landmark_px(body)
    a, b = np.array(lm[10]), np.array(lm[152])
    t = (b - a) / np.linalg.norm(b - a)
    n = np.array([-t[1], t[0]])  # signed offset along n: sign tells the side
    right_iris_side = np.sign(np.dot(np.array(lm[468]) - a, n))
    zones = {z["id"]: z for z in body["zones"]}
    for kind in ("eye", "brow"):
        for side, own, other in (("r", 468, 473), ("l", 473, 468)):
            zid = f"{kind}_{side}"
            assert zones[zid]["status"] == "ok", (zid, zones[zid])
            ys, xs = np.nonzero(zone_full_mask(body, zid))
            off = (np.stack([xs, ys], 1) - a) @ n
            want = right_iris_side if side == "r" else -right_iris_side
            assert (np.sign(off) == want).all(), f"{zid} crosses the midline"
            # widths stay near their landmark polygon (no merged both-eyes mask)
            assert zones[zid]["bbox"]["width"] < 90, zones[zid]["bbox"]
            if kind == "eye":
                m = zone_full_mask(body, zid)
                assert m[int(round(lm[own][1])), int(round(lm[own][0]))]
                assert not m[int(round(lm[other][1])), int(round(lm[other][0]))]


@needs_models
def test_original_fixture_result_unchanged_by_side_assignment(seg_fixture):
    """Same zone ids/statuses and eye/brow sides as before the BUG-1 change (baseline recorded from iteration 1)."""
    zones = {z["id"]: z for z in seg_fixture["zones"]}
    assert set(zones) == PHASE1_ZONES and all(z["status"] == "ok" for z in zones.values())
    assert seg_fixture["alignment_check"] == {"passed": True, "fraction": 1.0}
    assert seg_fixture["warnings"] == []
    # iteration-1 baseline: bbox centre x and width of each eye/brow zone (tolerance 2 px)
    baseline = {"eye_l": (450, 32), "eye_r": (365, 30), "brow_l": (452, 62), "brow_r": (359, 64)}
    for zid, (cx, w) in baseline.items():
        bb = zones[zid]["bbox"]
        assert abs(bb["x"] + bb["width"] / 2 - cx) <= 2 and abs(bb["width"] - w) <= 2, (zid, bb)


@needs_models
def test_alignment_check_is_false_when_exactly_one_iris_is_wrong():
    from app.services.face_regions import alignment_check, iris_checks
    from app.services.segmentation.regions import Region as R

    pts = np.array([[0.0, 0.0]] * 478)
    for i, xy in {468: (20, 20), 473: (60, 20), 1: (40, 40), 13: (40, 60), 105: (20, 8), 52: (20, 8),
                  334: (60, 8), 282: (60, 8)}.items():
        pts[i] = xy
    labels = np.full((80, 80), int(R.SKIN), dtype=np.uint8)
    labels[18:23, 18:23] = int(R.EYE_R)
    labels[18:23, 58:63] = int(R.EYE_L)
    labels[38:43, 38:43] = int(R.NOSE)
    labels[58:63, 38:43] = int(R.LIP_UPPER)
    labels[6:11, 18:23] = int(R.BROW_R)
    labels[6:11, 58:63] = int(R.BROW_L)
    assert alignment_check(pts, labels, (0, 0)).passed is True
    broken = labels.copy()
    broken[18:23, 18:23] = int(R.SKIN)  # one iris (468) no longer in an eye: 5/6 = 0.83 >= 0.6
    result = alignment_check(pts, broken, (0, 0))
    assert result.fraction >= 0.6 and iris_checks(pts, broken, (0, 0)) == (False, True)
    assert result.passed is False


# --- 4. mask validity on the real fixture (AC 5, 7) --------------------------------

@needs_models
def test_ac5_label_values_binary_and_regions_match_vocabulary(seg_fixture):
    lab, _ = label_map(seg_fixture)
    assert set(np.unique(lab).tolist()) <= VALID_LABELS
    assert {r["name"] for r in seg_fixture["anatomical"]["regions"]} == set(CANON)


@needs_models
def test_ac5_zone_masks_are_binary_and_decode_to_declared_bbox(seg_fixture):
    for z in seg_fixture["zones"]:
        assert z["status"] == "ok"
        m = decode(z["mask"]["png_base64"])
        b = z["bbox"]
        assert m.shape == (b["height"], b["width"]), z["id"]
        assert set(np.unique(m).tolist()) <= {0, 255}, z["id"]
        assert int(np.count_nonzero(m == 255)) == z["pixel_count"] > 0


@needs_models
def test_ac5_face_zones_are_subsets_of_skin_and_eyes_brows_nose_equal_their_class(seg_fixture):
    full = full_labels(seg_fixture)
    zones = {z["id"]: z for z in seg_fixture["zones"]}
    skin = full == CANON["skin"]
    skin_or_nose = skin | (full == CANON["nose"])
    for zid in ("forehead", "cheek_l", "cheek_r", "chin", "jawline", "under_eye_l", "under_eye_r"):
        m = zone_full_mask(seg_fixture, zid)
        assert not (m & ~skin).any(), f"{zid} leaves the skin class"
    assert not (zone_full_mask(seg_fixture, "face_skin") & ~skin_or_nose).any()
    for zid, cls in (("eye_l", "eye_l"), ("eye_r", "eye_r"), ("brow_l", "brow_l"), ("brow_r", "brow_r"), ("nose", "nose")):
        m = zone_full_mask(seg_fixture, zid)
        region = full == CANON[cls]
        assert (m == region).all(), zid
    assert zones["nose"]["clipped_to"] == ["nose"] and zones["face_skin"]["clipped_to"] == ["skin", "nose"]


@needs_models
def test_ac5_declared_clip_regions_are_real_canonical_names(seg_fixture):
    names = set()
    for z in seg_fixture["zones"]:
        names |= set(z["clipped_to"])
    assert names <= set(CLIP_IDS)


@needs_models
def test_ac5_eyes_do_not_leak_into_face_zones(seg_fixture):
    full = full_labels(seg_fixture)
    forbidden = np.isin(full, [CANON[k] for k in ("eye_l", "eye_r", "lip_upper", "lip_lower", "brow_l", "brow_r", "glasses", "hair")])
    for zid in ("forehead", "cheek_l", "cheek_r", "chin", "jawline", "under_eye_l", "under_eye_r", "face_skin"):
        m = zone_full_mask(seg_fixture, zid)
        assert not (m & forbidden).any(), zid


@needs_models
def test_ac5_landmark_zone_polygons_are_in_image_and_model_zones_have_none(seg_fixture):
    W, H = seg_fixture["image"]["width"], seg_fixture["image"]["height"]
    for z in seg_fixture["zones"]:
        if z["polygon"] is not None:
            p = np.array(z["polygon"])
            assert (p[:, 0] >= -2).all() and (p[:, 0] <= W + 2).all() and (p[:, 1] >= -2).all() and (p[:, 1] <= H + 2).all(), z["id"]
    zones = {z["id"]: z for z in seg_fixture["zones"]}
    assert MODEL_ONLY_ZONES <= set(zones)
    for zid in MODEL_ONLY_ZONES:
        assert zones[zid]["polygon"] is None, zid
    for zid in LANDMARK_EYE_ZONES:
        assert zid in zones and zones[zid]["polygon"], zid


@needs_models
def test_ac7_lip_zones_declare_no_clip_region_and_are_validated_by_iou(seg_fixture):
    zones = {z["id"]: z for z in seg_fixture["zones"]}
    for zid in ("lip_upper", "lip_lower", "lip_full"):
        assert zones[zid]["clipped_to"] == [] and zones[zid]["confidence"] is not None
    for zid in ("lip_outer_boundary", "lip_inner_boundary"):
        assert zones[zid]["confidence"] is None and zones[zid]["polygon"]


@needs_models
def test_ac7_model_zone_confidence_is_iou_and_face_zone_confidence_is_probability(seg_fixture):
    zones = {z["id"]: z for z in seg_fixture["zones"]}
    regions = {r["name"]: r for r in seg_fixture["anatomical"]["regions"]}
    # Parser-backed zones use mean probability over the zone (not calibrated; range check only)
    assert 0.5 < zones["face_skin"]["confidence"] <= 1.0
    assert 0.0 < zones["eye_l"]["confidence"] <= 1.0
    assert regions["skin"]["mean_probability"] is not None and regions["background"]["mean_probability"] is not None


# --- 5. robustness on real images (AC 9, requirement 12) ---------------------------

@needs_models
@pytest.mark.parametrize("name,payload,ctype,allowed", [
    ("rotate_15", lambda: rotated(15), "image/png", {"ok"}),
    ("rotate_minus_15", lambda: rotated(-15), "image/png", {"ok"}),
    ("rotate_35", lambda: rotated(35), "image/png", {"ok", "no_face_detected"}),
    ("rotate_minus_35", lambda: rotated(-35), "image/png", {"ok", "no_face_detected"}),
    ("cut_top_forehead", lambda: png(BASE.crop((0, int(H0 * 0.12), W0, H0))), "image/png", {"ok"}),
    ("cut_right_half", lambda: png(BASE.crop((0, 0, int(W0 * 0.45), H0))), "image/png", {"ok", "no_face_detected"}),
    ("cut_left_half", lambda: png(BASE.crop((int(W0 * 0.55), 0, W0, H0))), "image/png", {"ok", "no_face_detected"}),
    ("cut_bottom_chin", lambda: png(BASE.crop((0, 0, W0, int(H0 * 0.55)))), "image/png", {"ok"}),
    ("half_scale", lambda: png(BASE.resize((W0 // 2, H0 // 2), Image.LANCZOS)), "image/png", {"ok"}),
    ("quarter_scale", lambda: png(BASE.resize((W0 // 4, H0 // 4), Image.LANCZOS)), "image/png", {"ok"}),
    ("eighth_scale", lambda: png(BASE.resize((W0 // 8, H0 // 8), Image.LANCZOS)), "image/png", {"ok", "no_face_detected"}),
    ("aspect_1.3", lambda: png(pad(BASE, 120, 120, 0, 0)), "image/png", {"ok"}),
    ("aspect_1.6", lambda: png(pad(BASE, 300, 300, 0, 0)), "image/png", {"ok", "no_face_detected"}),
    ("aspect_2.4", lambda: png(pad(BASE, 790, 790, 0, 0)), "image/png", {"ok", "no_face_detected"}),
    ("aspect_1_to_3.6", lambda: png(pad(BASE, 0, 0, 1100, 1100)), "image/png", {"ok", "no_face_detected"}),
    ("grayscale", lambda: png(BASE.convert("L")), "image/png", {"ok"}),
    ("dark_x0.25", lambda: png(ImageEnhance.Brightness(BASE).enhance(0.25)), "image/png", {"ok"}),
    ("dark_x0.08", lambda: png(ImageEnhance.Brightness(BASE).enhance(0.08)), "image/png", {"ok", "no_face_detected"}),
    ("overexposed_x1.8", lambda: png(ImageEnhance.Brightness(BASE).enhance(1.8)), "image/png", {"ok", "no_face_detected"}),
    ("blank_scene", lambda: png(Image.new("RGB", (400, 300), (90, 90, 90))), "image/png", {"no_face_detected"}),
])
def test_robustness_status_and_warnings_are_honest(client, name, payload, ctype, allowed):
    r = post(client, SEG, payload(), ctype)
    assert r.status_code == 200, (name, r.status_code, r.text[:200])
    body = r.json()
    assert body["status"] in allowed, (name, body["status"])
    check_invariants(body)


@needs_models
def test_robustness_tilted_face_flags_head_tilt(client):
    body = post(client, SEG, rotated(35), "image/png").json()
    if body["status"] == "ok":
        assert "head_tilt" in {w["code"] for w in body["warnings"]}


@needs_models
def test_robustness_small_face_flags_low_resolution_and_unavailable_zones_are_honest(client):
    body = post(client, SEG, png(BASE.resize((W0 // 4, H0 // 4), Image.LANCZOS)), "image/png").json()
    assert body["status"] == "ok"
    assert "low_resolution_face" in {w["code"] for w in body["warnings"]}
    for z in body["zones"]:
        if z["status"] == "unavailable":
            assert z["mask"] is None and z["reason"]


@needs_models
def test_robustness_partial_face_flags_outside_image(client):
    body = post(client, SEG, png(BASE.crop((0, int(H0 * 0.12), W0, H0))), "image/png").json()
    assert body["status"] == "ok"
    assert "face_partially_outside_image" in {w["code"] for w in body["warnings"]}


@needs_models
def test_robustness_half_face_is_not_reported_as_ok_with_garbage(client):
    body = post(client, SEG, png(BASE.crop((0, 0, int(W0 * 0.45), H0))), "image/png").json()
    assert body["status"] in {"ok", "no_face_detected"}
    if body["status"] == "ok":
        assert body["alignment_check"]["passed"] or "parser_landmark_mismatch" in {w["code"] for w in body["warnings"]}


@needs_models
def test_robustness_grayscale_and_dark_inputs_keep_alignment(client):
    for img in (BASE.convert("L"), ImageEnhance.Brightness(BASE).enhance(0.25)):
        body = post(client, SEG, png(img), "image/png").json()
        assert body["status"] == "ok" and body["alignment_check"]["passed"]


# --- 6. large images, the 25 MP cap, and decode limits (AC 9) ----------------------

@needs_models
def test_large_upscale_near_25mp_is_ok_and_aligned(client):
    up = BASE.resize((round(W0 * 5.4), round(H0 * 5.4)), Image.BICUBIC)  # 24.5 MP
    data = jpg(up, 85)
    assert len(data) < 10 * 1024 * 1024  # inside the default upload cap
    r = post(client, SEG, data, "image/jpeg")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["image"] == {"width": up.width, "height": up.height}
    check_invariants(body)
    full = full_labels(body)
    lm = landmark_px(body)
    assert class_at(full, lm[468]) == CANON["eye_r"] and class_at(full, lm[473]) == CANON["eye_l"]
    assert body["parse_window"]["size"] > 1024  # exercises the nearest-neighbour mapping path


@needs_models
def test_pixel_cap_boundary_25mp_allowed_and_25mp_plus_one_rejected(client):
    ok_img = Image.new("L", (5000, 5000), 128)  # exactly 25,000,000 px
    big = Image.new("L", (5001, 5000), 128)  # 25,005,000 px
    for url in (LMK, SEG):
        assert post(client, url, png(ok_img), "image/png").status_code == 200, url
        r = post(client, url, png(big), "image/png")
        assert r.status_code == 400 and "too large" in r.json()["detail"], url


@needs_models
@pytest.mark.parametrize("make,msg", [
    (lambda: bad_png_header(6000, 6000), "too large"),  # 36 MP: our 25 MP cap must reject it by header
    (lambda: bomb_jpeg(6000, 6000), "too large"),
    # 3.6 GP: Pillow's own bomb guard (about 179 MP) raises at open time, which is also a 400 without decoding
    (lambda: bad_png_header(60000, 60000), None),
    (lambda: bomb_jpeg(60000, 60000), None),
])
def test_pixel_bomb_header_is_rejected_before_decoding(client, make, msg):
    data = make()  # a few hundred bytes claiming a huge bitmap
    ctype = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
    for url in (LMK, SEG):
        r = post(client, url, data, ctype)
        assert r.status_code == 400, (url, r.status_code)
        if msg:
            assert msg in r.json()["detail"], (url, r.json())
        assert "Traceback" not in r.text


# --- 7. multiple faces (AC 9, plan amendment) --------------------------------------

@needs_models
def test_two_comparable_faces_rejected_on_both_endpoints(client):
    head = BASE.crop((230, 0, 580, 400))
    data = png(two_side_by_side(head, head))
    body = post(client, SEG, data, "image/png").json()
    assert body["status"] == "multiple_faces_detected"
    assert body["message"] == "Please upload a photo with only one face."
    assert body["anatomical"] is None and body["zones"] == [] and body["landmarks"]["landmarks"] == []
    assert body["model"]["license_warning"] == LICENSE_WARNING
    lmk = post(client, LMK, data, "image/png").json()
    assert lmk["status"] == "multiple_faces_detected" and lmk["face_count"] == 2
    assert lmk["landmarks"] == [] and lmk["measurements"] is None and lmk["face_detected"] is True


@needs_models
def test_two_full_portraits_side_by_side_rejected(client):
    data = png(two_side_by_side(BASE, BASE))
    assert post(client, SEG, data, "image/png").json()["status"] == "multiple_faces_detected"
    assert post(client, LMK, data, "image/png").json()["status"] == "multiple_faces_detected"


@needs_models
def test_small_background_face_ignored_with_warning(client):
    head = BASE.crop((230, 0, 580, 400))
    small = head.resize((int(head.width * 0.42), int(head.height * 0.42)), Image.LANCZOS)  # area ratio ~0.18
    data = png(two_side_by_side(head, small, gap=20))
    body = post(client, SEG, data, "image/png").json()
    assert body["status"] == "ok" and body["landmarks"]["face_count"] == 2
    assert "background_face_ignored" in {w["code"] for w in body["warnings"]}
    check_invariants(body)
    lm = landmark_px(body)
    assert np.median([lm[i][0] for i in range(478)]) < head.width  # analysed the large face on the left
    lmk = post(client, LMK, data, "image/png").json()
    assert lmk["status"] == "ok" and lmk["face_count"] == 2


@needs_models
@pytest.mark.parametrize("scale,expected", [(0.40, "ok"), (0.55, "multiple_faces_detected")])
def test_significance_threshold_on_real_detector(client, scale, expected):
    """Area ratio ~0.16 at scale 0.40 is ignored; ~0.30 at 0.55 is rejected. Scales are kept away
    from the 0.20 boundary (linear ~0.447): the detector's landmark bbox shifts slightly with a gap,
    so 0.44-0.46 flip between the two outcomes depending on layout."""
    head = BASE.crop((230, 0, 580, 400))
    small = head.resize((int(head.width * scale), int(head.height * scale)), Image.LANCZOS)
    body = post(client, LMK, png(two_side_by_side(head, small, gap=20)), "image/png").json()
    assert body["face_count"] == 2, "MediaPipe must see both faces for the rule to apply"
    assert body["status"] == expected, scale


def test_primary_face_selection_and_ratio_rule_unit():
    assert select_primary_face([100.0], 0.2) == (0, False)
    assert select_primary_face([100.0, 19.9], 0.2) == (0, False)
    assert select_primary_face([100.0, 20.0], 0.2) == (0, True)
    assert select_primary_face([30.0, 100.0], 0.2) == (1, True)
    assert select_primary_face([100.0, 10.0, 25.0], 0.2) == (0, True)


def test_landmark_service_requests_two_faces():
    """The landmarker must look for two faces, otherwise multiple faces cannot be detected."""
    src = Path(landmarks_module.__file__).read_text()
    assert "num_faces=2" in src


# --- 8. invalid inputs on both endpoints (AC 9) -----------------------------------

@pytest.mark.parametrize("url", [SEG, LMK])
def test_invalid_inputs_return_documented_status(client, url):
    cases = [
        ("no file", None, None, 422),
        ("empty file", b"", "image/png", 400),
        ("text/plain declared", FIXTURE_BYTES, "text/plain", 415),
        ("octet-stream declared", FIXTURE_BYTES, "application/octet-stream", 415),
        ("image/gif declared (not allowed)", FIXTURE_BYTES, "image/gif", 415),
        ("random bytes as jpeg", os.urandom(4096), "image/jpeg", 400),
        ("gif bytes as png", _gif(), "image/png", 400),
        ("bmp bytes as png", _bmp(), "image/png", 400),
        ("tiff bytes as png", _tiff(), "image/png", 400),
        ("truncated jpeg", FIXTURE_BYTES[: len(FIXTURE_BYTES) // 2], "image/jpeg", 400),
        ("svg as png", b"<svg xmlns='http://www.w3.org/2000/svg'><rect/></svg>", "image/png", 400),
        ("bad png header", b"\x89PNG\r\n\x1a\nnot really", "image/png", 400),
    ]
    for label, data, ctype, want in cases:
        if data is None:
            r = client.post(url, files={})
        else:
            r = post(client, url, data, ctype)
        assert r.status_code == want, (label, r.status_code, r.text[:120])
        if want >= 400:
            detail = str(r.json().get("detail"))
            assert "Traceback" not in detail and "File \"" not in detail, label


def _gif() -> bytes:
    b = io.BytesIO()
    Image.new("P", (40, 40)).save(b, format="GIF")
    return b.getvalue()


def _bmp() -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (40, 40)).save(b, format="BMP")
    return b.getvalue()


def _tiff() -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (40, 40)).save(b, format="TIFF")
    return b.getvalue()


@pytest.mark.parametrize("url", [SEG, LMK])
@needs_models
def test_oversize_upload_is_413_on_both_endpoints(client, url):
    r = post(client, url, b"\x89PNG\r\n\x1a\n" + b"0" * (10 * 1024 * 1024), "image/png")
    assert r.status_code == 413


@pytest.mark.parametrize("url", [SEG, LMK])
@needs_models
def test_content_is_checked_not_only_the_declared_type(client, url):
    """A PNG labelled image/jpeg is still a PNG, so it is accepted. The decoder checks real bytes."""
    assert post(client, url, png(BASE), "image/jpeg").status_code == 200
    assert post(client, url, jpg(BASE), "image/png").status_code == 200
    assert post(client, url, png(BASE.resize((200, 250))), "image/webp").status_code == 200  # PNG is allow-listed
    webp = io.BytesIO()
    BASE.resize((200, 250)).save(webp, format="WEBP", quality=90)
    assert post(client, url, webp.getvalue(), "image/webp").status_code == 200


# --- 9. parser availability and error handling (AC 2, 9) --------------------------

@needs_models
@pytest.mark.parametrize("kind", ["missing", "directory", "dangling_symlink", "wrong_hash"])
def test_ac2_parser_unavailable_gives_503_and_landmarks_keep_working(monkeypatch, tmp_path, kind):
    path = tmp_path / "w.onnx"
    if kind == "directory":
        path.mkdir()
    elif kind == "dangling_symlink":
        path.symlink_to(tmp_path / "nope.onnx")
    elif kind == "wrong_hash":
        shutil.copyfile(SETTINGS.resolved_face_parser_path, path)
        with path.open("ab") as f:
            f.write(b"\x00")  # one byte changes the SHA-256
    settings = SETTINGS.model_copy(update={"face_parser_model_path": path})
    monkeypatch.setattr(app_main, "get_settings", lambda: settings)
    with TestClient(create_app()) as c:
        assert c.app.state.face_parser is None
        r = post(c, SEG, FIXTURE_BYTES, "image/jpeg")
        assert r.status_code == 503 and "download_face_parser" in r.json()["detail"]
        lm = post(c, LMK, FIXTURE_BYTES, "image/jpeg")
        assert lm.status_code == 200 and lm.json()["status"] == "ok"


@needs_models
def test_validation_precedes_parser_availability(monkeypatch, tmp_path):
    settings = SETTINGS.model_copy(update={"face_parser_model_path": tmp_path / "missing.onnx"})
    monkeypatch.setattr(app_main, "get_settings", lambda: settings)
    with TestClient(create_app()) as c:
        assert post(c, SEG, b"", "image/png").status_code == 400


@needs_models
@pytest.mark.skipif(os.geteuid() == 0, reason="root can read chmod 000 files")
def test_unreadable_weights_do_not_prevent_startup(monkeypatch, tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root can read chmod 000 files")
    path = tmp_path / "w.onnx"
    shutil.copyfile(SETTINGS.resolved_face_parser_path, path)
    path.chmod(0)
    # The failed lifespan leaks the landmarker it already built. Record it and close it here,
    # so the leaked MediaPipe object cannot be garbage-collected later in another test.
    created: list = []
    real_service = app_main.FaceLandmarkService

    def recording_service(*a, **kw):
        svc = real_service(*a, **kw)
        created.append(svc)
        return svc

    monkeypatch.setattr(app_main, "FaceLandmarkService", recording_service)
    monkeypatch.setattr(app_main, "get_settings", lambda: SETTINGS.model_copy(update={"face_parser_model_path": path}))
    try:
        with TestClient(create_app()) as c:
            assert post(c, SEG, FIXTURE_BYTES, "image/jpeg").status_code == 503
            assert post(c, LMK, FIXTURE_BYTES, "image/jpeg").status_code == 200
    finally:
        path.chmod(0o644)
        for svc in created:
            svc.close()


_LEAK_REPRO = r"""
import gc, shutil, sys
from pathlib import Path
from fastapi.testclient import TestClient
from app import main as app_main
from app.main import create_app
from app.config import get_settings
S = get_settings()
real, tmp = Path(sys.argv[1]), Path(sys.argv[2])
ok = tmp / "ok.onnx"
shutil.copyfile(real, ok)
app_main.get_settings = lambda: S.model_copy(update={"face_parser_model_path": ok})
with TestClient(create_app()):
    pass  # a normal start and stop
bad = tmp / "unreadable.onnx"
shutil.copyfile(real, bad)
bad.chmod(0)
app_main.get_settings = lambda: S.model_copy(update={"face_parser_model_path": bad})
try:
    with TestClient(create_app()):
        pass
except BaseException:
    pass  # startup failed before yield; the landmarker it built is never closed
bad.chmod(0o644)
gc.collect()  # the leaked MediaPipe landmarker's __del__ -> close() blocks here
print("DONE", flush=True)
"""


@needs_models
@pytest.mark.skipif(os.geteuid() == 0, reason="root can read chmod 000 files")
def test_failed_startup_does_not_leave_a_landmarker_that_hangs_on_gc(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root can read chmod 000 files")
    import subprocess
    import sys as _sys

    # Subprocess with a hard timeout, so the hang fails this test instead of freezing pytest.
    try:
        out = subprocess.run(
            [_sys.executable, "-c", _LEAK_REPRO, str(SETTINGS.resolved_face_parser_path), str(tmp_path)],
            cwd=BACKEND, capture_output=True, text=True, timeout=90,
        )
        finished, stdout = out.returncode == 0, out.stdout
    except subprocess.TimeoutExpired:
        finished, stdout = False, "timed out"
    assert finished and "DONE" in stdout, "process hung or crashed after a failed startup"


@needs_models
def test_inference_failure_is_generic_500_and_does_not_leak(client, monkeypatch):
    app = create_app(use_lifespan=False)
    app.dependency_overrides[get_landmark_service] = lambda: client.app.state.landmark_service
    broken = BrightnessStub()

    def boom(*a, **k):
        raise RuntimeError("secret-internal-detail /private/path")

    monkeypatch.setattr(broken, "_infer", boom)
    app.dependency_overrides[get_face_parser] = lambda: broken
    r = post(TestClient(app), SEG, FIXTURE_BYTES, "image/jpeg")
    assert r.status_code == 500 and r.json()["detail"] == "Face segmentation failed."
    assert "secret" not in r.text and "/private" not in r.text


@needs_models
def test_missing_mediapipe_landmarks_do_not_produce_an_ok_result(client, parser):
    """A detector that reports ok with too few points must not yield a usable segmentation."""

    class Truncated(_FakeLandmarks):
        def detect_landmarks(self, image):
            from app.schemas.facial_landmarks import DetectionThresholds, FacialLandmarkResult, ImageSize, Landmark

            h, w = image.shape[:2]
            lms = [Landmark(index=i, x=0.5, y=0.5, z=0.0) for i in range(300)]
            return FacialLandmarkResult(
                face_detected=True, status="ok", face_count=1, image=ImageSize(width=w, height=h),
                thresholds=DetectionThresholds(min_face_detection_confidence=0.5, min_face_presence_confidence=0.5),
                landmark_count=300, landmarks=lms,
            )

    app = create_app(use_lifespan=False)
    app.dependency_overrides[get_landmark_service] = lambda: Truncated("ok")
    app.dependency_overrides[get_face_parser] = lambda: parser
    r = post(TestClient(app), SEG, FIXTURE_BYTES, "image/jpeg")
    assert not (r.status_code == 200 and r.json()["status"] == "ok")
    assert "IndexError" not in r.text and "Traceback" not in r.text


@needs_models
def test_parser_busy_under_real_contention_is_busy_not_crash(parser):
    tight = OnnxFaceParser(SETTINGS.resolved_face_parser_path, SETTINGS.face_parser_sha256,
                           max_concurrency=1, queue_timeout_s=0.01)
    img = np.asarray(BASE, dtype=np.uint8)
    bbox = (300.0, 100.0, 520.0, 360.0)
    results: list[object] = []
    lock = threading.Lock()

    def run():
        try:
            tight.parse(img, bbox)
            out: object = "ok"
        except FaceParserBusyError:
            out = "busy"
        except Exception as exc:  # noqa: BLE001 - we want to see any other failure type
            out = exc
        with lock:
            results.append(out)

    threads = [threading.Thread(target=run) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert len(results) == 6
    assert all(r in ("ok", "busy") for r in results), results
    assert "ok" in results


@needs_models
def test_segmentation_runs_mediapipe_exactly_once_per_request(client, monkeypatch):
    """Landmarks are embedded in the segmentation response, so MediaPipe must not run twice."""
    svc = client.app.state.landmark_service
    calls: list[int] = []
    real_detect = svc.detect_landmarks

    def counting(image):
        calls.append(1)
        return real_detect(image)

    monkeypatch.setattr(svc, "detect_landmarks", counting)
    assert post(client, SEG, FIXTURE_BYTES, "image/jpeg").status_code == 200
    assert len(calls) == 1


@needs_models
def test_saturated_parser_gives_503_with_retry_after_over_http(client, monkeypatch):
    """Real parser class, real route: the only inference slot is held, so the request must get 503."""
    slot = OnnxFaceParser(SETTINGS.resolved_face_parser_path, SETTINGS.face_parser_sha256,
                          max_concurrency=1, queue_timeout_s=0.05)
    entered, release = threading.Event(), threading.Event()
    real_infer = slot._infer

    def gated(x):
        entered.set()
        release.wait(30)
        return real_infer(x)

    monkeypatch.setattr(slot, "_infer", gated)
    app = create_app(use_lifespan=False)
    app.dependency_overrides[get_landmark_service] = lambda: client.app.state.landmark_service
    app.dependency_overrides[get_face_parser] = lambda: slot
    c = TestClient(app)
    holder = threading.Thread(target=lambda: slot.parse(np.asarray(BASE), (300.0, 100.0, 520.0, 360.0)))
    holder.start()
    try:
        assert entered.wait(30)
        r = post(c, SEG, FIXTURE_BYTES, "image/jpeg")
        assert r.status_code == 503 and r.headers.get("retry-after"), r.status_code
        assert "busy" in r.json()["detail"].lower()
    finally:
        release.set()
        holder.join(30)
    assert post(c, SEG, FIXTURE_BYTES, "image/jpeg").status_code == 200  # slot released


@needs_models
def test_concurrent_http_requests_never_return_500(client):
    def go(_):
        return post(client, SEG, FIXTURE_BYTES, "image/jpeg")

    with ThreadPoolExecutor(max_workers=4) as ex:
        responses = list(ex.map(go, range(8)))
    codes = [r.status_code for r in responses]
    assert set(codes) <= {200, 503}, codes
    for r in responses:
        if r.status_code == 503:
            assert r.headers.get("retry-after")
    assert codes.count(200) >= 1


@needs_models
def test_weights_path_comes_from_settings_only():
    """Weights location and hash come from settings, not from request data."""
    src = (BACKEND / "app" / "api" / "face_segmentation.py").read_text()
    assert "Form(" not in src and "Query(" not in src and "Header(" not in src
    assert SETTINGS.face_parser_sha256 == "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f"


def test_gitignore_excludes_weights_and_mentions_onnx():
    text = (REPO / ".gitignore").read_text()
    assert "backend/models/*.onnx" in text


def test_documentation_mentions_phase_scope_and_model():
    doc = (REPO / "docs" / "face_segmentation.md").read_text()
    for needle in ("SHA-256", "BiSeNet", "CelebAMask-HQ", "19"):
        assert needle in doc, needle


# --- 10. leaking and forbidden data (AC 9, security basics) ------------------------

@needs_models
def test_response_has_no_model_class_names(seg_fixture):
    import re

    text = str(seg_fixture)
    # "ear_r" is excluded on purpose: it is the canonical name of the subject's right ear.
    hits = re.findall(r"\b(l_eye|r_eye|u_lip|l_lip|neck_l|l_brow|r_brow|eye_g)\b", text)
    assert not hits, hits


@needs_models
def test_response_has_no_filename_or_path_fields(client):
    body = post(client, SEG, FIXTURE_BYTES, "image/jpeg", name="secret_name_123.jpg").json()
    assert "secret_name_123" not in str(body)


# --- 10. iteration-2 regression: mirrored robustness and end-to-end wrong-iris (BUG-1) ---

MIR = ImageOps.mirror(BASE)


def _mirrored_exif_jpeg() -> bytes:
    stored = MIR.rotate(90, expand=True)
    exif = stored.getexif()
    exif[0x0112] = 6
    b = io.BytesIO()
    stored.save(b, format="JPEG", quality=95, exif=exif)
    return b.getvalue()


def _mask_near(full_mask: np.ndarray, p: tuple[float, float], r: int = 3) -> bool:
    x, y = int(round(p[0])), int(round(p[1]))
    return bool(full_mask[max(y - r, 0) : y + r + 1, max(x - r, 0) : x + r + 1].any())


def _assert_mirrored_sides_follow_geometry(body: dict) -> None:
    """Each OK eye/brow zone lies on its own side of the roll-aware 10-152 midline, and each
    OK eye zone covers its own iris but not the other one (checked with a 3 px tolerance)."""
    lm = landmark_px(body)
    a, b = np.array(lm[10]), np.array(lm[152])
    t = (b - a) / np.linalg.norm(b - a)
    n = np.array([-t[1], t[0]])
    right_side = np.sign(np.dot(np.array(lm[468]) - a, n))
    assert right_side != 0 and np.sign(np.dot(np.array(lm[473]) - a, n)) == -right_side, "irises on one side"
    zones = {z["id"]: z for z in body["zones"]}
    for kind in ("eye", "brow"):
        for side, own, other in (("r", 468, 473), ("l", 473, 468)):
            zid = f"{kind}_{side}"
            if zones[zid]["status"] != "ok":
                continue
            ys, xs = np.nonzero(zone_full_mask(body, zid))
            assert len(xs) > 0, zid
            off = (np.stack([xs, ys], 1) - a) @ n
            want = right_side if side == "r" else -right_side
            assert (np.sign(off) == want).all(), f"{zid} crosses the midline"
            if kind == "eye":
                m = zone_full_mask(body, zid)
                assert _mask_near(m, lm[own]), f"{zid} ok but misses its own iris"
                assert not _mask_near(m, lm[other]), f"{zid} ok but covers the other iris"


MIRROR_VARIANTS = [
    # name, payload factory, content type, allowed statuses, must be ok with both eyes/brows ok
    ("rotate_15", lambda: png(MIR.rotate(15, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0))), "image/png", {"ok"}, True),
    ("rotate_minus_15", lambda: png(MIR.rotate(-15, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0))), "image/png", {"ok"}, True),
    ("rotate_35", lambda: png(MIR.rotate(35, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0))), "image/png", {"ok", "no_face_detected"}, False),
    ("rotate_minus_35", lambda: png(MIR.rotate(-35, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0))), "image/png", {"ok", "no_face_detected"}, False),
    ("exif_sideways", _mirrored_exif_jpeg, "image/jpeg", {"ok"}, True),
    ("half_scale", lambda: png(MIR.resize((W0 // 2, H0 // 2), Image.LANCZOS)), "image/png", {"ok"}, True),
    ("quarter_scale", lambda: png(MIR.resize((W0 // 4, H0 // 4), Image.LANCZOS)), "image/png", {"ok"}, False),
    ("up_5.4", lambda: jpg(MIR.resize((round(W0 * 5.4), round(H0 * 5.4)), Image.BICUBIC), 85), "image/jpeg", {"ok"}, True),
    ("cut_top_forehead", lambda: png(MIR.crop((0, int(H0 * 0.12), W0, H0))), "image/png", {"ok"}, True),
    ("cut_right_half", lambda: png(MIR.crop((0, 0, int(W0 * 0.45), H0))), "image/png", {"ok", "no_face_detected"}, False),
    ("cut_left_half", lambda: png(MIR.crop((int(W0 * 0.55), 0, W0, H0))), "image/png", {"ok", "no_face_detected"}, False),
    ("grayscale", lambda: png(MIR.convert("L")), "image/png", {"ok"}, True),
    ("dark_x0.25", lambda: png(ImageEnhance.Brightness(MIR).enhance(0.25)), "image/png", {"ok"}, True),
    ("dark_x0.08", lambda: png(ImageEnhance.Brightness(MIR).enhance(0.08)), "image/png", {"ok", "no_face_detected"}, False),
]


@needs_models
@pytest.mark.parametrize("name,payload,ctype,allowed,must_be_ok", MIRROR_VARIANTS, ids=[v[0] for v in MIRROR_VARIANTS])
def test_mirrored_robustness_variants_keep_each_side_on_its_own_side(client, name, payload, ctype, allowed, must_be_ok):
    r = post(client, SEG, payload(), ctype)
    assert r.status_code == 200, (name, r.status_code)
    body = r.json()
    assert body["status"] in allowed, (name, body["status"])
    check_invariants(body)
    if body["status"] != "ok":
        return
    _assert_mirrored_sides_follow_geometry(body)
    if must_be_ok:
        assert body["alignment_check"]["passed"], name
        zones = {z["id"]: z for z in body["zones"]}
        for zid in ("eye_l", "eye_r", "brow_l", "brow_r"):
            assert zones[zid]["status"] == "ok", (name, zid, zones[zid]["status"], zones[zid]["reason"])


class _OneWrongIrisParser:
    """Wraps the real parser and erases ONE side's eye mask, so exactly one iris lands outside an
    eye class. The erased pixels become skin. Side 'l' = subject left = image right."""

    def __init__(self, inner, side: str):
        self._inner = inner
        self.info = inner.info
        self._side = side

    def parse(self, image, face_bbox):
        from app.services.segmentation.regions import Region, Unsided

        res = self._inner.parse(image, face_bbox)
        cx = (face_bbox[0] + face_bbox[2]) / 2
        ox, oy = res.origin
        ys, xs = np.nonzero(res.labels == int(Unsided.EYE))
        gx = xs + ox
        bad = gx > cx if self._side == "l" else gx < cx
        res.labels[ys[bad], xs[bad]] = int(Region.SKIN)
        return res


@needs_models
@pytest.mark.parametrize("side", ["l", "r"])
def test_one_wrong_iris_end_to_end_fails_check_and_degrades_only_that_side(client, parser, side):
    wrapped = _OneWrongIrisParser(parser, side)
    client.app.dependency_overrides[get_face_parser] = lambda: wrapped
    try:
        r = post(client, SEG, FIXTURE_BYTES, "image/jpeg")
    finally:
        client.app.dependency_overrides.pop(get_face_parser, None)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    check_invariants(body)
    al = body["alignment_check"]
    assert al["passed"] is False
    assert round(al["fraction"], 3) == 0.833  # only the one iris check fails; the fraction alone would pass
    assert "parser_landmark_mismatch" in {w["code"] for w in body["warnings"]}
    zones = {z["id"]: z for z in body["zones"]}
    other = "r" if side == "l" else "l"
    for zid in (f"eye_{side}", f"under_eye_{side}"):
        assert zones[zid]["status"] != "ok", (zid, zones[zid]["status"])
        assert zones[zid]["reason"], zid
    assert zones[f"eye_{side}"]["status"] in {"low_confidence", "unavailable"}, zones[f"eye_{side}"]
    assert zones[f"under_eye_{side}"]["status"] == "low_confidence", zones[f"under_eye_{side}"]
    assert zones[f"eye_{other}"]["status"] == "ok", zones[f"eye_{other}"]
    assert zones[f"under_eye_{other}"]["status"] == "ok", zones[f"under_eye_{other}"]


@needs_models
@pytest.mark.skipif(os.geteuid() == 0, reason="root can read chmod 000 files")
def test_failed_startup_closes_the_landmark_service_itself(monkeypatch, tmp_path):
    """No test-side cleanup: the lifespan must close the landmarker it created after a failed parser start."""
    path = tmp_path / "w.onnx"
    shutil.copyfile(SETTINGS.resolved_face_parser_path, path)
    path.chmod(0)
    closed: list = []
    real_close = app_main.FaceLandmarkService.close

    def spy_close(self):
        closed.append(self)
        real_close(self)

    monkeypatch.setattr(app_main.FaceLandmarkService, "close", spy_close)
    monkeypatch.setattr(app_main, "get_settings", lambda: SETTINGS.model_copy(update={"face_parser_model_path": path}))
    try:
        with TestClient(create_app()) as c:
            assert post(c, SEG, FIXTURE_BYTES, "image/jpeg").status_code == 503
            assert post(c, LMK, FIXTURE_BYTES, "image/jpeg").status_code == 200
        assert len(closed) == 1, "lifespan did not close the landmark service after the parser failed"
    finally:
        path.chmod(0o644)
