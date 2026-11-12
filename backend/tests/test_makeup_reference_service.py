"""Tests for the reference analysis service. Never touches the network."""

import itertools
import socket
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import Settings, get_settings
from app.schemas.facial_landmarks import ImageSize
from app.schemas.facial_profile import FacialProfile
from app.schemas.makeup_blueprint import (
    BlueprintStyle,
    BlueprintWarning,
    MakeupBlueprint,
    ReferenceProviderInfo,
)
from app.services.makeup_blueprint import cv_analysis as cv
from app.services.makeup_blueprint import service as svc
from app.services.makeup_blueprint.cv_analysis import (
    CvAnalysis,
    CvAnalysisError,
    CvReferenceAnalyzer,
)
from app.services.makeup_blueprint.service import (
    ParserUnavailableError,
    ReferenceAnalysisError,
    ReferenceDeps,
    analyze_reference,
    build_reference_deps,
)
from app.services.makeup_blueprint.vlm import (
    StubVlmProvider,
    VlmAnalysis,
    VlmAnnotationResult,
)
from app.services.makeup_mapping import CatalogDriftError
from app.services.segmentation.face_parser import FaceParserBusyError
from tests.test_makeup_cv_analysis import RECTS, _image, _profile

CANARY = "CANARY-SECRET-xyz123"
SERVICE_SRC = Path(svc.__file__)


def _w(code, message="m"):
    return BlueprintWarning(code=code, message=message)


class FakeVlm:
    def __init__(self, analysis=None, exc=None, info_exc=None):
        self.analysis, self.exc, self.info_exc = analysis, exc, info_exc
        self.seen = None

    def info(self):
        if self.info_exc:
            raise self.info_exc
        return ReferenceProviderInfo(
            id="fake_vlm", kind="vlm", algorithm_version="1.0", model="m", enabled=True
        )

    def analyze(self, image, cv_components):
        self.seen = list(cv_components)
        if self.exc:
            raise self.exc
        return self.analysis if self.analysis is not None else VlmAnalysis(info=self.info())


class FakeCv:
    def __init__(self, result=None, exc=None):
        self.result, self.exc = result, exc

    def info(self):
        return CvReferenceAnalyzer().info()

    def analyze(self, image, profile):
        if self.exc:
            raise self.exc
        return self.result


def _deps(vlm=None, cv_an=None, parsing=object()):
    return ReferenceDeps(
        landmark_provider=object(),
        parsing_provider=parsing,
        zone_provider=object(),
        cv_analyzer=cv_an or CvReferenceAnalyzer(),
        vlm_provider=vlm or StubVlmProvider(),
    )


@pytest.fixture
def patch_profile(monkeypatch):
    def apply(profile):
        monkeypatch.setattr(svc, "build_profile", lambda *a, **k: profile)

    return apply


def _comp(product_type, category, zone, conf):
    return cv._make(category, product_type, "#C81428", "warm", 0.5, conf, (zone,))


def _two():
    return [_comp("lipstick", "lip", "lip_full", 0.8), _comp("powder_blush", "blush", "cheek_apple_l", 0.6)]


def _ok_profile():
    return _profile()


def _red_lips_image():
    return _image({"lip_full": (200, 20, 40)})


@pytest.mark.parametrize("status", ["no_face_detected", "multiple_faces_detected"])
def test_non_ok_status_passthrough(patch_profile, status):
    p = FacialProfile(status=status, message="Pass me through.", image=ImageSize(width=30, height=20))
    patch_profile(p)
    clock = iter([0.0, 2.0])
    bp = analyze_reference(np.zeros((20, 30, 3), np.uint8), deps=_deps(), clock=lambda: next(clock))
    assert bp.status == status and bp.message == "Pass me through."
    assert bp.components == [] and bp.style is None and bp.providers == []
    assert bp.image.width == 30 and bp.image.height == 20
    assert bp.timings_ms == {"profile": 2000.0, "total": 2000.0}


def test_ok_path_merges_and_orders(patch_profile):
    patch_profile(_ok_profile())
    style = BlueprintStyle(labels=["natural"], confidence=0.7)
    vlm = FakeVlm(
        VlmAnalysis(
            info=FakeVlm().info(),
            annotations=[VlmAnnotationResult(0, finish="matte")],
            style=style,
        )
    )
    clock = iter([0.0, 1.0, 3.0, 6.0])
    bp = analyze_reference(_red_lips_image(), deps=_deps(vlm), clock=lambda: next(clock))
    assert bp.status == "ok"
    (lip,) = bp.components
    assert lip.component.product_type == "lipstick"
    assert lip.component.finish == "matte" and lip.sources["finish"] == "vlm"
    assert lip.sources["color_hex"] == "cv"
    assert bp.style == style
    assert [p.kind for p in bp.providers] == ["cv", "vlm"]
    assert [p.id for p in bp.providers] == ["cv_zone_color", "fake_vlm"]
    assert list(bp.timings_ms) == ["profile", "cv", "vlm", "total"]
    assert bp.timings_ms == {"profile": 1000.0, "cv": 2000.0, "vlm": 3000.0, "total": 6000.0}
    assert vlm.seen is not None and len(vlm.seen) == 1


def test_stub_vlm_warning_and_provider(patch_profile):
    patch_profile(_ok_profile())
    bp = analyze_reference(_red_lips_image(), deps=_deps())
    assert "vlm_disabled" in [w.code for w in bp.quality.warnings]
    vlm_info = bp.providers[1]
    assert vlm_info.id == "vlm_stub" and vlm_info.enabled is False


def test_warnings_deduped_first_wins_and_capped(patch_profile):
    patch_profile(_ok_profile())
    result = CvAnalysis(_two(), [_w("low_confidence", "cv one"), _w("low_confidence", "cv two")], 0.7)
    vlm = FakeVlm(
        VlmAnalysis(
            info=FakeVlm().info(),
            warnings=[_w("low_confidence", "vlm"), _w("vlm_refused", "r")],
        )
    )
    bp = analyze_reference(_red_lips_image(), deps=_deps(vlm, FakeCv(result)))
    codes = [w.code for w in bp.quality.warnings]
    assert codes == ["low_confidence", "vlm_refused"]
    assert bp.quality.warnings[0].message == "cv one"
    assert len(svc._dedupe([_w("low_confidence")] * 40)) == 1


@pytest.mark.parametrize(
    "vlm_warnings,cv_result,expected",
    [
        ([], CvAnalysis(_two(), [], 0.1), 0.7),
        ([_w("vlm_failed")], CvAnalysis(_two(), [], 0.1), 0.63),
        ([_w("vlm_refused")], CvAnalysis(_two(), [], 0.1), 0.63),
        ([], CvAnalysis(_two(), [_w("glasses_present")], 0.1), 0.63),
        ([], CvAnalysis([], [], 0.5), 0.5),
        ([_w("vlm_failed")], CvAnalysis([], [], 0.5), 0.45),
        ([], CvAnalysis([], [], None), None),
        ([_w("vlm_failed")], CvAnalysis([], [], None), None),
    ],
)
def test_overall_confidence(patch_profile, vlm_warnings, cv_result, expected):
    patch_profile(_ok_profile())
    vlm = FakeVlm(VlmAnalysis(info=FakeVlm().info(), warnings=vlm_warnings))
    bp = analyze_reference(_red_lips_image(), deps=_deps(vlm, FakeCv(cv_result)))
    assert bp.quality.overall_confidence == expected


def test_overall_confidence_rounds_to_4_decimals_and_penalty_needs_code():
    comps = [_comp("lipstick", "lip", "lip_full", 0.123456)]
    assert svc._overall_confidence(comps, None, []) == 0.1235
    assert svc._overall_confidence(comps, None, [_w("low_confidence")]) == 0.1235
    assert svc._overall_confidence(comps, None, [_w("vlm_failed")]) == round(0.123456 * 0.9, 4)


def test_overall_confidence_clamps_cv_fallback():
    assert svc._overall_confidence([], 1.7, []) == 1.0
    assert svc._overall_confidence([], -0.5, []) == 0.0


def test_warning_cap_keeps_first_in_order(monkeypatch):
    monkeypatch.setattr(svc, "MAX_WARNINGS", 2)
    out = svc._dedupe([_w("low_confidence"), _w("vlm_failed")], [_w("vlm_refused"), _w("glasses_present")])
    assert [w.code for w in out] == ["low_confidence", "vlm_failed"]


def test_output_has_no_heavy_fields(patch_profile):
    profile = _ok_profile()
    # Guard: the input profile must actually carry mask payloads, otherwise the check is vacuous.
    assert "png_base64" in profile.model_dump_json()
    patch_profile(profile)
    bp = analyze_reference(_red_lips_image(), deps=_deps())
    assert isinstance(bp, MakeupBlueprint)
    dumped = bp.model_dump_json()
    for word in ("mask", "png_base64", "landmarks", "label_map", "parsing"):
        assert word not in dumped, word
    # Parsing label payloads are dict/list-valued keys named "labels" outside style.
    # style.labels (vocabulary strings) is legitimate and is excluded explicitly.
    data = bp.model_dump(mode="json")
    data.pop("style", None)
    assert "labels" not in str(data)


@pytest.mark.parametrize(
    "exc,stage",
    [
        (CatalogDriftError(CANARY), "catalog"),
        (CvAnalysisError(CANARY), "cv"),
    ],
)
def test_cv_errors_mapped(patch_profile, exc, stage):
    patch_profile(_ok_profile())
    with pytest.raises(ReferenceAnalysisError) as ei:
        analyze_reference(_red_lips_image(), deps=_deps(cv_an=FakeCv(exc=exc)))
    assert ei.value.stage == stage
    assert CANARY not in str(ei.value)
    assert ei.value.__cause__ is None and ei.value.__suppress_context__ is True


def test_generic_cv_exception_mapped(patch_profile):
    patch_profile(_ok_profile())
    with pytest.raises(ReferenceAnalysisError) as ei:
        analyze_reference(
            _red_lips_image(), deps=_deps(cv_an=FakeCv(exc=RuntimeError(CANARY)))
        )
    assert ei.value.stage == "cv"
    assert CANARY not in str(ei.value)
    assert ei.value.__cause__ is None


def test_profile_error_mapped(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError(CANARY)

    monkeypatch.setattr(svc, "build_profile", boom)
    with pytest.raises(ReferenceAnalysisError) as ei:
        analyze_reference(_red_lips_image(), deps=_deps())
    assert ei.value.stage == "profile" and CANARY not in str(ei.value)
    assert ei.value.__cause__ is None and ei.value.__suppress_context__ is True


def test_busy_propagates(monkeypatch):
    def busy(*a, **k):
        raise FaceParserBusyError()

    monkeypatch.setattr(svc, "build_profile", busy)
    with pytest.raises(FaceParserBusyError):
        analyze_reference(_red_lips_image(), deps=_deps())


def test_parser_unavailable():
    with pytest.raises(ParserUnavailableError) as ei:
        analyze_reference(_red_lips_image(), deps=_deps(parsing=None))
    assert isinstance(ei.value, ReferenceAnalysisError)


def test_vlm_exception_degrades_to_cv_only(patch_profile):
    patch_profile(_ok_profile())
    vlm = FakeVlm(exc=RuntimeError(CANARY))
    bp = analyze_reference(_red_lips_image(), deps=_deps(vlm))
    assert [c.component.product_type for c in bp.components] == ["lipstick"]
    assert bp.style is None
    (w,) = [w for w in bp.quality.warnings if w.code == "vlm_failed"]
    assert CANARY not in (w.message or "")
    assert [p.kind for p in bp.providers] == ["cv", "vlm"]


def test_vlm_exception_and_info_exception_uses_stub_info(patch_profile):
    patch_profile(_ok_profile())
    vlm = FakeVlm(exc=RuntimeError("a"), info_exc=RuntimeError("b"))
    bp = analyze_reference(_red_lips_image(), deps=_deps(vlm))
    assert bp.providers[1].id == "vlm_stub"
    assert "vlm_failed" in [w.code for w in bp.quality.warnings]


def test_build_reference_deps_defaults():
    deps = build_reference_deps(Settings(_env_file=None), object(), None)
    assert isinstance(deps.cv_analyzer, CvReferenceAnalyzer)
    assert isinstance(deps.vlm_provider, StubVlmProvider)
    assert deps.parsing_provider is None
    assert type(deps.zone_provider).__name__ == "DefaultZoneProvider"
    with pytest.raises(Exception):
        deps.landmark_provider = None  # frozen


def test_no_model_construct_or_copy_in_service():
    text = SERVICE_SRC.read_text()
    assert "model_construct" not in text and "model_copy" not in text
    assert "logging" not in text and "print(" not in text


# --- integration: real local models, stub VLM, no network ---------------------------------
_SETTINGS = get_settings()
_LOCAL_MODELS_PRESENT = (
    _SETTINGS.resolved_model_path.exists() and _SETTINGS.resolved_face_parser_path.exists()
)


def _deny_network(*_a, **_k):
    raise AssertionError("network access attempted")


@pytest.mark.skipif(not _LOCAL_MODELS_PRESENT, reason="local model weights not downloaded")
def test_integration_portrait_stub_vlm(monkeypatch):
    from app.services.facial_analysis.landmarks import FaceLandmarkService
    from app.services.facial_profile.providers import (
        DefaultParsingProvider,
        DefaultZoneProvider,
        MediaPipeLandmarkProvider,
    )
    from app.services.segmentation.face_parser import OnnxFaceParser

    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _deny_network)
    monkeypatch.setattr(socket, "create_connection", _deny_network)
    bgr = cv2.imread(str(Path(__file__).parent / "fixtures" / "portrait.jpg"), cv2.IMREAD_COLOR)
    assert bgr is not None
    img = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
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
        deps = build_reference_deps(
            Settings(_env_file=None, makeup_vlm_enabled=False),
            MediaPipeLandmarkProvider(lm, _SETTINGS.resolved_model_path),
            DefaultParsingProvider(parser),
            DefaultZoneProvider(),
        )
        bp = analyze_reference(img, deps=deps)
    finally:
        lm.close() if hasattr(lm, "close") else None
    assert bp.status == "ok"
    MakeupBlueprint.model_validate(bp.model_dump())
    assert "vlm_disabled" in [w.code for w in bp.quality.warnings]
    assert [p.kind for p in bp.providers] == ["cv", "vlm"]
    assert bp.providers[1].enabled is False
    assert set(bp.timings_ms) == {"profile", "cv", "vlm", "total"}
