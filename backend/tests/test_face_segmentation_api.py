"""Segmentation endpoint, multiple-face handling and upload hardening."""

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.face_segmentation import get_face_parser
from app.api.facial_analysis import get_landmark_service
from app.config import get_settings
from app.main import create_app
from app.schemas.face_segmentation import LICENSE_WARNING
from app.services.facial_analysis.landmarks import FaceLandmarkService, select_primary_face
from app.services.image_loading import ImageValidationError, load_rgb_image
from app.services.segmentation import masks as mk
from app.services.segmentation.face_parser import FaceParserBusyError, FaceParserInferenceError
from app.services.segmentation.regions import NOT_PARSED, Region
from tests.synthetic import FIXTURES, FakeLandmarkService, FakeParser

SEG = "/api/facial-analysis/segmentation"
LMK = "/api/facial-analysis/landmarks"
SETTINGS = get_settings()


def _png(w: int = 820, h: int = 1024, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format=fmt)
    return buf.getvalue()


def _client(landmarks=None, parser="default", settings=None) -> TestClient:
    app = create_app(use_lifespan=False)
    app.dependency_overrides[get_landmark_service] = lambda: landmarks or FakeLandmarkService()
    app.dependency_overrides[get_face_parser] = lambda: FakeParser() if parser == "default" else parser
    if settings:
        app.dependency_overrides[get_settings] = lambda: SETTINGS.model_copy(update=settings)
    return TestClient(app)


def _post(client: TestClient, url: str = SEG, data: bytes | None = None, ctype: str = "image/png"):
    return client.post(url, files={"file": ("a.png", _png() if data is None else data, ctype)})


# --- segmentation response ----------------------------------------------------------

def test_segmentation_ok_response_contract():
    r = _post(_client())
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok"
    assert body["model"]["license_warning"] == LICENSE_WARNING
    assert body["image"] == {"width": 820, "height": 1024}
    assert body["landmarks"]["landmark_count"] == 478 and body["landmarks"]["face_count"] == 1
    assert set(body["timings_ms"]) >= {"preprocess", "inference", "postprocess", "zones", "encode", "total"}
    assert body["alignment_check"]["passed"] is True
    stages = body["stages"]
    assert [st["number"] for st in stages] == list(range(1, 12))
    lips = stages[9]
    assert (lips["key"], lips["name"]) == ("lips", "Lips")
    assert list(lips["layers"]) == ["lipstick", "lip_liner", "gloss"]
    assert list(stages[5]["layers"]) == ["contour", "bronzer"] and (
        stages[5]["layers"]["bronzer"] == [
            "temple_r", "temple_l", "forehead_perimeter", "cheek_outer_r", "cheek_outer_l",
            "jawline_outer_r", "jawline_outer_l"])

    a = body["anatomical"]
    labels = mk.decode_png(base64.b64decode(a["label_map_png_base64"]))
    assert labels.shape == (a["height"], a["width"]) and labels.dtype == np.uint8
    pw = body["parse_window"]
    # label map = parse window ∩ image
    assert a["origin"] == {"x": max(pw["x"], 0), "y": max(pw["y"], 0)}
    assert a["origin"]["x"] + a["width"] <= 820 and a["origin"]["y"] + a["height"] <= 1024
    assert a["not_parsed_value"] == NOT_PARSED
    assert {r["name"] for r in a["regions"]} == {r.name.lower() for r in Region}
    present = {r["id"] for r in a["regions"] if r["present"]}
    assert set(np.unique(labels).tolist()) == present
    assert sum(r["pixel_count"] for r in a["regions"]) == labels.size
    assert {r["name"]: r["side"] for r in a["regions"] if r["side"]}["eye_l"] == "left"


def test_zone_masks_are_valid_bbox_pngs():
    body = _post(_client()).json()
    assert body["zones"]
    for z in body["zones"]:
        assert z["id"] and z["name"] and z["sources"] and z["status"] in ("ok", "low_confidence", "unavailable")
        if z["status"] == "unavailable":
            assert z["mask"] is None and z["reason"]
            continue
        b = z["bbox"]
        m = mk.decode_png(base64.b64decode(z["mask"]["png_base64"]))
        assert m.shape == (b["height"], b["width"]) and set(np.unique(m).tolist()) <= {0, 255}
        assert int((m == 255).sum()) == z["pixel_count"] > 0  # an empty mask is never reported as usable
        assert 0 <= b["x"] and b["x"] + b["width"] <= 820 and 0 <= b["y"] and b["y"] + b["height"] <= 1024
    landmark_zone = next(z for z in body["zones"] if z["id"] == "lip_full")
    assert landmark_zone["polygon"] and all(len(p) == 2 for p in landmark_zone["polygon"])
    assert next(z for z in body["zones"] if z["id"] == "eye_l")["polygon"] is None  # model zone: no polygon


def test_no_face_returns_200_with_status_and_no_anatomy():
    body = _post(_client(FakeLandmarkService("no_face_detected"))).json()
    assert body["status"] == "no_face_detected"
    assert body["anatomical"] is None and body["zones"] == [] and body["alignment_check"] is None
    assert body["model"]["license_warning"] == LICENSE_WARNING


def test_multiple_faces_returns_200_with_status():
    r = _post(_client(FakeLandmarkService("multiple_faces_detected", face_count=2)))
    body = r.json()
    assert r.status_code == 200 and body["status"] == "multiple_faces_detected"
    assert body["message"] == "Please upload a photo with only one face."
    assert body["anatomical"] is None and body["zones"] == [] and body["landmarks"]["landmarks"] == []


def test_background_face_warning_when_a_second_small_face_was_ignored():
    body = _post(_client(FakeLandmarkService("ok", face_count=2))).json()
    assert body["status"] == "ok"
    assert "background_face_ignored" in {w["code"] for w in body["warnings"]}
    assert "background_face_ignored" not in {w["code"] for w in _post(_client()).json()["warnings"]}


class _GlassesParser(FakeParser):
    """FakeParser that paints a controllable fraction of the label map as GLASSES."""

    def __init__(self, fraction: float):
        super().__init__()
        self.fraction = fraction

    def parse(self, image, face_bbox):
        result = super().parse(image, face_bbox)
        result.labels.flat[: int(round(self.fraction * result.labels.size))] = int(Region.GLASSES)
        return result


def _glasses_codes(body: dict) -> bool:
    return "glasses_detected" in {w["code"] for w in body["warnings"]}


def test_glasses_at_or_above_threshold_suggests_retake():
    body = _post(_client(parser=_GlassesParser(0.03))).json()
    assert body["status"] == "ok"
    assert _glasses_codes(body) and body["retake_suggested"] == ["glasses"]


def test_glasses_below_threshold_has_no_warning_and_no_retake():
    body = _post(_client(parser=_GlassesParser(0.005))).json()
    assert body["status"] == "ok"
    assert not _glasses_codes(body) and body["retake_suggested"] == []


def test_no_face_and_multiple_faces_have_no_retake_suggestion():
    assert _post(_client(FakeLandmarkService("no_face_detected"))).json()["retake_suggested"] == []
    multi = _post(_client(FakeLandmarkService("multiple_faces_detected", face_count=2))).json()
    assert multi["retake_suggested"] == []


@pytest.mark.parametrize("fraction", [0.0, 0.005, 0.02, 0.05])
def test_retake_hint_present_exactly_when_glasses_warning_present(fraction):
    body = _post(_client(parser=_GlassesParser(fraction))).json()
    assert (body["retake_suggested"] == ["glasses"]) == _glasses_codes(body)


def test_parser_unavailable_is_503_and_landmarks_still_work():
    client = _client(parser=None)
    r = _post(client)
    assert r.status_code == 503 and "download_face_parser" in r.json()["detail"]
    assert _post(client, LMK).status_code == 200


def test_validation_happens_before_parser_availability():
    assert _post(_client(parser=None), data=b"nope").status_code == 400
    assert _post(_client(parser=None), data=b"x", ctype="text/plain").status_code == 415


def test_busy_is_503_with_retry_after():
    r = _post(_client(parser=FakeParser(error=FaceParserBusyError("busy"))))
    assert r.status_code == 503 and r.headers["retry-after"]


def test_inference_error_is_generic_500(caplog):
    secret = "SECRET-LANDMARK-DETAIL"
    r = _post(_client(parser=FakeParser(error=FaceParserInferenceError(secret))))
    assert r.status_code == 500
    assert r.json()["detail"] == "Face segmentation failed."
    assert secret not in r.text and secret not in caplog.text
    assert "FaceParserInferenceError" in caplog.text  # type only


def test_unexpected_error_is_generic_500():
    r = _post(_client(parser=FakeParser(error=RuntimeError("boom with path /etc/passwd"))))
    assert r.status_code == 500 and "passwd" not in r.text


# --- upload hardening (both endpoints) ---------------------------------------------

@pytest.mark.parametrize("url", [SEG, LMK])
def test_pixel_cap_rejects_with_400(url):
    r = _post(_client(settings={"max_image_pixels": 1000}), url, data=_png(100, 50))
    assert r.status_code == 400 and "too large" in r.json()["detail"]


@pytest.mark.parametrize("url", [SEG, LMK])
@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_real_format_outside_allow_list_is_rejected_even_if_labelled_png(url, fmt):
    r = _post(_client(), url, data=_png(40, 40, fmt), ctype="image/png")
    assert r.status_code == 400


def test_decompression_bomb_error_maps_to_400_not_500(monkeypatch):
    # Pillow raises DecompressionBombError (not an OSError) above 2x MAX_IMAGE_PIXELS.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    data = _png(100, 50)  # 5000 px > 2 * 1000
    with pytest.raises(ImageValidationError):
        load_rgb_image(data, max_pixels=10**9)
    assert _post(_client(), data=data).status_code == 400


def test_allowed_formats_still_load():
    for fmt in ("PNG", "JPEG", "WEBP"):
        assert load_rgb_image(_png(30, 20, fmt)).shape == (20, 30, 3)


def test_pixel_cap_is_checked_before_decoding():
    # A 30 MP 1-bit PNG is tiny on disk; it must be rejected from the header, before decoding.
    buf = io.BytesIO()
    Image.new("1", (6000, 5000)).save(buf, format="PNG")
    with pytest.raises(ImageValidationError, match="too large"):
        load_rgb_image(buf.getvalue())


# --- multiple faces on /landmarks ----------------------------------------------------

def test_landmarks_endpoint_reports_multiple_faces_and_face_count():
    client = _client(FakeLandmarkService("multiple_faces_detected", face_count=2))
    body = _post(client, LMK).json()
    assert body["status"] == "multiple_faces_detected" and body["face_detected"] is True
    assert body["landmarks"] == [] and body["measurements"] is None and body["face_count"] == 2
    ok = _post(_client(), LMK).json()
    assert ok["face_count"] == 1 and ok["status"] == "ok" and ok["measurements"] is not None


def test_primary_face_is_the_largest_and_ratio_rule():
    assert select_primary_face([100.0], 0.2) == (0, False)
    assert select_primary_face([100.0, 19.9], 0.2) == (0, False)
    assert select_primary_face([100.0, 20.0], 0.2) == (0, True)  # >= is significant
    assert select_primary_face([30.0, 100.0], 0.2) == (1, True)  # index 0 is not assumed primary
    assert select_primary_face([10.0, 100.0], 0.2) == (1, False)


def _fake_landmarker(face_specs):
    """face_specs: list of (cx, cy, half_size) in normalized coords."""
    faces = []
    for cx, cy, hs in face_specs:
        lms = [SimpleNamespace(x=cx + (i % 7 - 3) / 3 * hs, y=cy + (i % 5 - 2) / 2 * hs, z=0.0) for i in range(478)]
        faces.append(lms)
    return SimpleNamespace(detect=lambda _img: SimpleNamespace(face_landmarks=faces))


def _service_with(face_specs, ratio=0.2) -> FaceLandmarkService:
    import threading

    from app.schemas.facial_landmarks import DetectionThresholds

    svc = object.__new__(FaceLandmarkService)
    svc._thresholds = DetectionThresholds(min_face_detection_confidence=0.5, min_face_presence_confidence=0.5)
    svc._secondary_ratio = ratio
    svc._lock = threading.Lock()
    svc._landmarker = _fake_landmarker(face_specs)
    return svc


def test_service_picks_largest_face_even_when_listed_second():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    r = _service_with([(0.2, 0.5, 0.03), (0.7, 0.5, 0.2)]).detect_landmarks(img)  # small, then big
    assert r.status == "ok" and r.face_count == 2
    assert r.landmarks[0].x == pytest.approx(0.7 - 0.2, abs=1e-6)  # the big face's landmarks


def test_service_rejects_two_comparable_faces():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    r = _service_with([(0.25, 0.5, 0.15), (0.75, 0.5, 0.15)]).detect_landmarks(img)
    assert (r.status, r.face_detected, r.face_count, r.landmarks) == ("multiple_faces_detected", True, 2, [])
    assert r.message == "Please upload a photo with only one face."


def _service_with_n_points(n: int) -> FaceLandmarkService:
    svc = _service_with([(0.5, 0.5, 0.2)])
    face = [SimpleNamespace(x=0.3 + (i % 7) * 0.01, y=0.3 + (i % 5) * 0.01, z=0.0) for i in range(n)]
    svc._landmarker = SimpleNamespace(detect=lambda _img: SimpleNamespace(face_landmarks=[face]))
    return svc


INCOMPLETE = "Face landmarks were incomplete; please retake the photo with your whole face visible."


@pytest.mark.parametrize("n", [477, 1, 0])
def test_truncated_landmarks_give_a_status_not_a_500(n, caplog):
    svc = _service_with_n_points(n)
    r = svc.detect_landmarks(np.zeros((100, 200, 3), dtype=np.uint8))
    assert (r.status, r.face_detected, r.landmark_count, r.landmarks) == ("no_face_detected", False, 0, [])
    assert r.message == INCOMPLETE
    # through both endpoints (the real service with a stubbed landmarker)
    client = _client(landmarks=svc)
    lm = _post(client, LMK)
    assert lm.status_code == 200 and lm.json()["status"] == "no_face_detected"
    assert lm.json()["message"] == INCOMPLETE and lm.json()["landmarks"] == []
    seg = _post(client)
    body = seg.json()
    assert seg.status_code == 200 and body["status"] == "no_face_detected" and body["message"] == INCOMPLETE
    assert body["anatomical"] is None and body["zones"] == []
    # the log line carries a count only
    assert all("x=" not in m and "0.3" not in m for m in caplog.messages)


def test_478_landmarks_are_unchanged():
    r = _service_with_n_points(478).detect_landmarks(np.zeros((100, 200, 3), dtype=np.uint8))
    assert (r.status, r.face_detected, r.landmark_count) == ("ok", True, 478) and r.message is None


# --- real MediaPipe / real parser ----------------------------------------------------

REAL = pytest.mark.skipif(
    not (SETTINGS.resolved_model_path.is_file() and SETTINGS.resolved_face_parser_path.is_file()),
    reason="model weights not downloaded",
)


def _two_faces(second_scale: float) -> bytes:
    # Head crops, not the full portrait: MediaPipe's detector only sees a second
    # face if it is not tiny relative to the whole frame.
    head = Image.open(FIXTURES / "portrait.jpg").convert("RGB").crop((230, 0, 580, 400))
    small = head.resize((int(head.width * second_scale), int(head.height * second_scale)))
    canvas = Image.new("RGB", (head.width + small.width, head.height), (128, 128, 128))
    canvas.paste(head, (0, 0))
    canvas.paste(small, (head.width, 0))
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG")
    return buf.getvalue()


@REAL
def test_real_pipeline_on_portrait_fixture():
    with TestClient(create_app()) as client:
        r = _post(client, data=(FIXTURES / "portrait.jpg").read_bytes(), ctype="image/jpeg")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok" and body["alignment_check"]["passed"]
    assert body["landmarks"]["face_count"] == 1
    a = body["anatomical"]
    labels = mk.decode_png(base64.b64decode(a["label_map_png_base64"]))
    assert labels.shape == (a["height"], a["width"])
    zones = {z["id"]: z for z in body["zones"]}
    assert all(zones[z]["status"] == "ok" for z in zones), {k: v["reason"] for k, v in zones.items() if v["status"] != "ok"}
    # every parser-backed zone mask is a subset of its label-map region
    for zid, region in (("eye_l", Region.EYE_L), ("eye_r", Region.EYE_R), ("brow_l", Region.BROW_L),
                        ("brow_r", Region.BROW_R), ("nose", Region.NOSE)):
        b, o = zones[zid]["bbox"], a["origin"]
        m = mk.decode_png(base64.b64decode(zones[zid]["mask"]["png_base64"])) == 255
        crop = labels[b["y"] - o["y"] : b["y"] - o["y"] + b["height"], b["x"] - o["x"] : b["x"] - o["x"] + b["width"]]
        assert (crop[m] == int(region)).all(), zid
    for zid in ("forehead", "cheek_l", "cheek_r", "chin", "jawline", "under_eye_l", "under_eye_r"):
        b, o = zones[zid]["bbox"], a["origin"]
        m = mk.decode_png(base64.b64decode(zones[zid]["mask"]["png_base64"])) == 255
        crop = labels[b["y"] - o["y"] : b["y"] - o["y"] + b["height"], b["x"] - o["x"] : b["x"] - o["x"] + b["width"]]
        assert (crop[m] == int(Region.SKIN)).all(), zid


@REAL
def test_real_two_comparable_faces_are_rejected_on_both_endpoints():
    data = _two_faces(1.0)
    with TestClient(create_app()) as client:
        for url in (SEG, LMK):
            body = _post(client, url, data=data, ctype="image/jpeg").json()
            assert body["status"] == "multiple_faces_detected", url
            lm = body["landmarks"] if url == SEG else body
            assert lm["landmarks"] == [] and lm["face_count"] == 2


@REAL
def test_real_small_background_face_is_ignored_with_warning():
    data = _two_faces(0.4)  # bbox area ratio ~0.15 < 0.20
    with TestClient(create_app()) as client:
        body = _post(client, data=data, ctype="image/jpeg").json()
        lm = _post(client, LMK, data=data, ctype="image/jpeg").json()
    assert body["status"] == "ok" and body["landmarks"]["face_count"] == 2
    assert "background_face_ignored" in {w["code"] for w in body["warnings"]}
    assert lm["status"] == "ok" and lm["face_count"] == 2
    # the analyzed face is the large one (left half of the canvas)
    assert body["landmarks"]["landmarks"][1]["x"] < 0.5


@pytest.mark.skipif(not SETTINGS.resolved_model_path.is_file(), reason="landmarker model not downloaded")
@pytest.mark.parametrize("corrupt", [False, True])
def test_startup_survives_missing_or_corrupt_parser_weights(monkeypatch, tmp_path, corrupt):
    path = tmp_path / "w.onnx"
    if corrupt:
        path.write_bytes(b"garbage")
    monkeypatch.setenv("FACE_PARSER_MODEL_PATH", str(path))
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            assert client.app.state.face_parser is None
            assert _post(client, data=(FIXTURES / "portrait.jpg").read_bytes(), ctype="image/jpeg").status_code == 503
            lm = _post(client, LMK, data=(FIXTURES / "portrait.jpg").read_bytes(), ctype="image/jpeg")
            assert lm.status_code == 200 and lm.json()["status"] == "ok"  # landmarks keep working
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
