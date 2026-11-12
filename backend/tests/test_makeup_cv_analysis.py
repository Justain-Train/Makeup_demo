import base64
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.schemas.face_segmentation import SegmentationWarning, Zone, ZoneBBox, ZoneMask
from app.schemas.facial_landmarks import ImageSize
from app.schemas.facial_profile import (
    Component,
    ComponentMeta,
    ComponentSet,
    FacialProfile,
    ParsingData,
    ProfileQuality,
    ProviderInfo,
    ZonesData,
)
from app.services.application_zones import catalog
from app.services.makeup_blueprint import cv_analysis as cv
from app.services.makeup_blueprint.cv_analysis import (
    TARGETS,
    CvAnalysisError,
    CvReferenceAnalyzer,
    undertone_from_lab,
)
from app.services.makeup_mapping import CatalogDriftError, map_zones
from app.services.makeup_vocabulary.color import hex_to_lab, parse_hex

W = H = 300
SKIN = (205, 160, 140)

RECTS = {
    "face_skin": (20, 20, 260, 260),
    "lip_full": (120, 220, 60, 30),
    "cheek_apple_l": (50, 120, 40, 40),
    "cheek_apple_r": (210, 120, 40, 40),
    "eyelid_upper_l": (70, 80, 40, 15),
    "eyelid_upper_r": (190, 80, 40, 15),
    "cheekbone_l": (50, 170, 30, 20),
    "cheekbone_r": (220, 170, 30, 20),
    "contour_hollow_l": (60, 200, 30, 20),
    "contour_hollow_r": (210, 200, 30, 20),
}


def _zone(zid, rect, status="ok", confidence=0.9):
    x, y, w, h = rect
    ok, buf = cv2.imencode(".png", np.full((h, w), 255, dtype=np.uint8))
    assert ok
    return Zone(
        id=zid, name=zid, group="g", side=None, sources=["landmarks"], status=status,
        confidence=confidence, clipped_to=[], bbox=ZoneBBox(x=x, y=y, width=w, height=h),
        pixel_count=w * h,
        mask=ZoneMask(png_base64=base64.b64encode(buf.tobytes()).decode()),
    )


def _meta(kind, status="ok"):
    return ComponentMeta(
        component_version=1, status=status,
        provider=ProviderInfo(id="t", kind=kind, algorithm_version="1"),
    )


def _profile(rects=None, zone_status="ok", status="ok", warnings=(), zones_status="ok", parsing_status="ok"):
    rects = RECTS if rects is None else rects
    if status != "ok":
        return FacialProfile(status=status, image=ImageSize(width=W, height=H))
    zones = [_zone(z, r, zone_status if z != "face_skin" else "ok") for z, r in rects.items()]
    zdata = ZonesData(zones=zones, stages=[]) if zones_status != "unavailable" else None
    pdata = ParsingData(includes_masks=True, regions=[]) if parsing_status != "unavailable" else None
    comps = ComponentSet(
        landmarks=Component(meta=_meta("landmarks", "unavailable")),
        frame=Component(meta=_meta("landmarks", "unavailable")),
        parsing=Component(meta=_meta("parsing", parsing_status), data=pdata),
        zones=Component(meta=_meta("zones", zones_status), data=zdata),
    )
    return FacialProfile(
        status="ok", image=ImageSize(width=W, height=H), components=comps,
        quality=ProfileQuality(warnings=[SegmentationWarning(code=c, message="m") for c in warnings]),
    )


def _image(paint=None):
    img = np.empty((H, W, 3), dtype=np.uint8)
    img[:] = SKIN
    for zid, color in (paint or {}).items():
        x, y, w, h = RECTS[zid]
        img[y : y + h, x : x + w] = color
    return img


def _de(h1, h2):
    return math.dist(hex_to_lab(h1), hex_to_lab(h2))


def _by_pt(result):
    return {c.component.product_type: c for c in result.components}


def test_lipstick_detected():
    red = (200, 20, 40)
    res = CvReferenceAnalyzer().analyze(_image({"lip_full": red}), _profile())
    assert list(_by_pt(res)) == ["lipstick"]
    c = res.components[0]
    assert _de(c.component.color_hex, "#C81428") < 3
    assert 0.0 <= c.component.intensity <= 1.0
    assert c.component.target_zones == ["lip_full"]
    assert set(c.sources) == {"category", "product_type", "color_hex", "undertone", "intensity", "target_zones"}
    assert set(c.sources.values()) == {"cv"}
    assert c.component.finish is None and c.component.shade_name is None


def test_plain_skin_no_makeup():
    res = CvReferenceAnalyzer().analyze(_image(), _profile())
    assert res.components == []
    assert [w.code for w in res.warnings] == ["no_makeup_detected"]
    assert res.confidence is None


def test_blush_detected_in_layer():
    res = CvReferenceAnalyzer().analyze(
        _image({"cheek_apple_l": (225, 130, 140), "cheek_apple_r": (225, 130, 140)}), _profile()
    )
    comp = _by_pt(res)["powder_blush"].component
    layer = set(map_zones("powder_blush").zone_ids)
    assert comp.target_zones and set(comp.target_zones) <= layer


def test_highlighter_and_contour():
    res = CvReferenceAnalyzer().analyze(
        _image({"cheekbone_l": (250, 215, 195), "cheekbone_r": (250, 215, 195),
                "contour_hollow_l": (150, 110, 90), "contour_hollow_r": (150, 110, 90)}),
        _profile(),
    )
    assert {"powder_highlighter", "contour_powder"} <= set(_by_pt(res))


def test_glasses_skip_eyeshadow():
    img = _image({"eyelid_upper_l": (60, 80, 200), "eyelid_upper_r": (60, 80, 200)})
    res = CvReferenceAnalyzer().analyze(img, _profile(warnings=["glasses_detected"]))
    assert "eyeshadow" not in _by_pt(res)
    assert "glasses_present" in [w.code for w in res.warnings]
    res2 = CvReferenceAnalyzer().analyze(img, _profile())
    assert "eyeshadow" in _by_pt(res2)


def test_reference_fallback_low_confidence():
    rects = dict(RECTS)
    rects["face_skin"] = (100, 100, 40, 40)  # 1600 px < MIN_REF_PIXELS
    res = CvReferenceAnalyzer().analyze(_image(), _profile(rects))
    assert "low_confidence" in [w.code for w in res.warnings]


def test_small_zone_skipped():
    rects = dict(RECTS)
    rects["lip_full"] = (120, 220, 10, 10)
    res = CvReferenceAnalyzer().analyze(_image({"lip_full": (200, 20, 40)}), _profile(rects))
    assert "lipstick" not in _by_pt(res)
    assert "low_confidence" in [w.code for w in res.warnings]


def test_bad_bbox_dropped():
    rects = dict(RECTS)
    rects["lip_full"] = (280, 280, 60, 30)
    res = CvReferenceAnalyzer().analyze(_image(), _profile(rects))
    assert "lipstick" not in _by_pt(res)


@pytest.mark.parametrize(
    "kwargs",
    [{"status": "no_face_detected"}, {"zones_status": "unavailable"}, {"parsing_status": "unavailable"}],
)
def test_unusable_profile_empty(kwargs):
    res = CvReferenceAnalyzer().analyze(_image(), _profile(**kwargs))
    assert res.components == []
    assert [w.code for w in res.warnings] == ["low_confidence"]


def test_table_zones_map():
    for row in TARGETS:
        zm = map_zones(row.product_type, None, row.zones)
        assert set(zm.zone_ids) == set(row.zones)
        assert row.category == cv.MakeupComponent(
            category=row.category, product_type=row.product_type
        ).category


def test_catalog_drift(monkeypatch):
    stages = [
        catalog.StageSpec(s.number, s.key, s.name, {k: v for k, v in s.layers.items() if k != "lipstick"})
        for s in catalog.STAGES
    ]
    monkeypatch.setattr(catalog, "STAGES", stages)
    with pytest.raises(CatalogDriftError):
        CvReferenceAnalyzer().analyze(_image({"lip_full": (200, 20, 40)}), _profile())


def test_schema_failure_no_input_leak(monkeypatch):
    monkeypatch.setattr(cv, "undertone_from_lab", lambda a, b: "SECRETVALUE")
    with pytest.raises(CvAnalysisError) as ei:
        CvReferenceAnalyzer().analyze(_image({"lip_full": (200, 20, 40)}), _profile())
    assert "SECRETVALUE" not in str(ei.value)


def test_bad_image_rejected():
    with pytest.raises(CvAnalysisError):
        CvReferenceAnalyzer().analyze(np.zeros((10, 10), dtype=np.uint8), _profile())


def test_info():
    info = CvReferenceAnalyzer().info()
    assert info.kind == "cv" and info.id == "cv_zone_color" and info.model is None and info.enabled


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (5.0, 0.0, "neutral"),
        (7.9, 0.0, "neutral"),
        (8.0, 0.0, "cool"),
        (10 * math.cos(math.radians(24.9)), 10 * math.sin(math.radians(24.9)), "cool"),
        (10 * math.cos(math.radians(25.1)), 10 * math.sin(math.radians(25.1)), "neutral"),
        (10 * math.cos(math.radians(54.9)), 10 * math.sin(math.radians(54.9)), "neutral"),
        (10 * math.cos(math.radians(55.1)), 10 * math.sin(math.radians(55.1)), "warm"),
        (10 * math.cos(math.radians(84.9)), 10 * math.sin(math.radians(84.9)), "warm"),
        (10 * math.cos(math.radians(85.1)), 10 * math.sin(math.radians(85.1)), "olive"),
        (10 * math.cos(math.radians(136.0)), 10 * math.sin(math.radians(136.0)), "olive"),  # green
        (10 * math.cos(math.radians(199.9)), 10 * math.sin(math.radians(199.9)), "olive"),
        (10 * math.cos(math.radians(200.1)), 10 * math.sin(math.radians(200.1)), "cool"),
        (10 * math.cos(math.radians(306.0)), 10 * math.sin(math.radians(306.0)), "cool"),  # blue
        (10.0, -1.0, "cool"),
    ],
)
def test_undertone_boundaries(a, b, expected):
    assert undertone_from_lab(a, b) == expected


def test_fallback_reference_color_exact(monkeypatch):
    rects = dict(RECTS)
    rects["face_skin"] = (100, 100, 40, 40)
    seen = []
    real = cv.hex_to_lab

    def spy(h):
        seen.append(h)
        return real(h)

    monkeypatch.setattr(cv, "hex_to_lab", spy)
    CvReferenceAnalyzer().analyze(_image(), _profile(rects))
    assert seen[0] == "#CDA08C"


@pytest.mark.parametrize("status", ["multiple_faces_detected"])
def test_other_non_ok_status_empty(status):
    res = CvReferenceAnalyzer().analyze(_image(), _profile(status=status))
    assert res.components == []
    assert [w.code for w in res.warnings] == ["low_confidence"]
    assert res.confidence is None


def _crop_zone(b64, w=40, h=30):
    z = _zone("lip_full", (10, 10, w, h))
    return Zone(
        id=z.id, name=z.name, group=z.group, side=None, sources=["landmarks"], status="ok",
        confidence=0.9, clipped_to=[], bbox=ZoneBBox(x=10, y=10, width=w, height=h),
        pixel_count=w * h, mask=ZoneMask(png_base64=b64),
    )


def _png_b64(w, h):
    ok, buf = cv2.imencode(".png", np.full((h, w), 255, dtype=np.uint8))
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def _spy_decode(monkeypatch):
    calls = []
    monkeypatch.setattr(cv.mk, "decode_png", lambda data: calls.append(1) or np.zeros((1, 1)))
    return calls


def test_crop_valid_png_decodes():
    out = cv._crop((100, 100, 3), _crop_zone(_png_b64(40, 30)))
    assert out is not None and out[0].shape == (30, 40)


def test_crop_rejects_ihdr_dim_mismatch(monkeypatch):
    calls = _spy_decode(monkeypatch)
    assert cv._crop((100, 100, 3), _crop_zone(_png_b64(41, 30))) is None
    assert calls == []


def test_crop_rejects_oversized_base64(monkeypatch):
    calls = _spy_decode(monkeypatch)
    limit = 4 * math.ceil((40 * 30 + 1024) / 3)
    big = "A" * (limit + 4)
    assert cv._crop((100, 100, 3), _crop_zone(big)) is None
    assert calls == []


def test_crop_rejects_truncated_and_non_png(monkeypatch):
    calls = _spy_decode(monkeypatch)
    good = base64.b64decode(_png_b64(40, 30))
    for payload in (good[:20], b"GIF89a" + b"\x00" * 40, b"\x89PNG\r\n\x1a\n" + b"\x00" * 16):
        assert cv._crop((100, 100, 3), _crop_zone(base64.b64encode(payload).decode())) is None
    assert calls == []


def test_crop_rejects_invalid_base64(monkeypatch):
    calls = _spy_decode(monkeypatch)
    assert cv._crop((100, 100, 3), _crop_zone("!!!not base64!!!")) is None
    assert calls == []


def test_no_model_construct_in_module():
    src = Path(cv.__file__).read_text()
    assert "model_construct" not in src and "model_copy" not in src


# --- build_profile integration (local providers only, no network) -------------------------
from app.config import get_settings  # noqa: E402
from app.services.facial_analysis.landmarks import FaceLandmarkService  # noqa: E402
from app.services.facial_profile.builder import build_profile  # noqa: E402
from app.services.facial_profile.providers import (  # noqa: E402
    DefaultParsingProvider,
    DefaultZoneProvider,
    MediaPipeLandmarkProvider,
)
from app.services.segmentation.face_parser import OnnxFaceParser  # noqa: E402

_SETTINGS = get_settings()
_LOCAL_MODELS_PRESENT = _SETTINGS.resolved_model_path.exists() and _SETTINGS.resolved_face_parser_path.exists()


def _deny_network(*_a, **_k):
    raise AssertionError("network access attempted")


def _fixture_rgb() -> np.ndarray:
    bgr = cv2.imread(str(Path(__file__).parent / "fixtures" / "portrait.jpg"), cv2.IMREAD_COLOR)
    assert bgr is not None
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


@pytest.mark.skipif(not _LOCAL_MODELS_PRESENT, reason="local model weights not downloaded")
def test_build_profile_integration_deterministic(monkeypatch):
    import socket

    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _deny_network)
    monkeypatch.setattr(socket, "create_connection", _deny_network)
    img = _fixture_rgb()
    analyzer = CvReferenceAnalyzer()

    def run():
        lm = FaceLandmarkService(
            model_path=_SETTINGS.resolved_model_path,
            min_face_detection_confidence=_SETTINGS.min_face_detection_confidence,
            min_face_presence_confidence=_SETTINGS.min_face_presence_confidence,
            secondary_face_area_ratio=_SETTINGS.secondary_face_area_ratio,
        )
        try:
            parser = OnnxFaceParser(
                model_path=_SETTINGS.resolved_face_parser_path,
                expected_sha256=_SETTINGS.face_parser_sha256,
            )
            profile = build_profile(
                img,
                MediaPipeLandmarkProvider(lm, _SETTINGS.resolved_model_path),
                DefaultParsingProvider(parser),
                DefaultZoneProvider(),
                include_masks=True,
            )
        finally:
            lm.close()
        assert profile.status == "ok"
        return analyzer.analyze(img, profile)

    first, second = run(), run()
    dump = lambda r: [c.model_dump(mode="json") for c in r.components]  # noqa: E731
    assert dump(first) == dump(second)
    assert [w.model_dump(mode="json") for w in first.warnings] == [w.model_dump(mode="json") for w in second.warnings]
    assert first.confidence == second.confidence

    for bc in first.components:
        assert set(bc.sources.values()) == {"cv"}
        assert 0.0 <= bc.component.intensity <= 1.0
        layer = set(map_zones(bc.component.product_type).zone_ids)
        assert set(bc.component.target_zones) <= layer


@pytest.mark.parametrize(
    "hue_deg,expected_below,expected_above",
    [
        (25.0, "cool", "neutral"),
        (55.0, "neutral", "warm"),
        (85.0, "warm", "olive"),
        (200.0, "olive", "cool"),
    ],
)
def test_undertone_exact_hue_thresholds(hue_deg, expected_below, expected_above):
    eps = 1e-6
    for hue, expected in ((hue_deg - eps, expected_below), (hue_deg + eps, expected_above)):
        r = math.radians(hue)
        assert undertone_from_lab(10 * math.cos(r), 10 * math.sin(r)) == expected


@pytest.mark.parametrize("chroma,expected", [(8.0 - 1e-9, "neutral"), (8.0, "cool")])
def test_undertone_exact_chroma_threshold(chroma, expected):
    # hue 0 (a-axis): chroma alone decides neutral vs chromatic
    assert undertone_from_lab(chroma, 0.0) == expected
