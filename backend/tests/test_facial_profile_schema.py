"""FacialProfile schema, versioning and compat."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas import facial_profile as fp
from app.schemas.face_measurements import BoundingBox, FaceMeasurements
from app.schemas.face_segmentation import (
    AlignmentCheck,
    AnatomicalLayer,
    ParseWindow,
    Point,
    RegionInfo,
    SegmentationWarning,
    Stage,
    Zone,
    ZoneBBox,
    ZoneMask,
)
from app.schemas.facial_landmarks import DetectionThresholds, ImageSize, Landmark
from app.services.facial_profile import compat
from app.services.facial_profile.versions import COMPONENT_VERSIONS, ZONES_ALGORITHM_VERSION


def _provider(kind):
    return fp.ProviderInfo(id=f"{kind}-test", kind=kind, algorithm_version="t-1")


def _meta(kind, status="ok"):
    return fp.ComponentMeta(
        component_version=COMPONENT_VERSIONS[kind if kind != "parsing" else "parsing"],
        status=status,
        confidence=0.9,
        provider=_provider("landmarks" if kind == "frame" else kind),
    )


def _components() -> fp.ComponentSet:
    box = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, width=10, height=10)
    meas = FaceMeasurements(
        face_bounding_box=box, interpupillary_distance_px=4, inner_eye_corner_distance_px=2,
        left_eye_width_px=1, right_eye_width_px=1, mouth_width_px=3, face_width_px=10, face_height_px=10,
    )
    lm = fp.LandmarksData(
        thresholds=DetectionThresholds(min_face_detection_confidence=0.5, min_face_presence_confidence=0.5),
        landmarks=[Landmark(index=0, x=0.5, y=0.5, z=0.0)],
        measurements=meas,
    )
    frame = fp.FaceFrameData(ipd_px=4.0, roll_deg=1.5, u=(1.0, 0.0), v=(0.0, 1.0), midline=[(5.0, 1.0), (5.0, 9.0)])
    layer = AnatomicalLayer(
        origin=Point(x=0, y=0), width=2, height=2, label_map_png_base64="AAAA",
        regions=[RegionInfo(id=1, name="skin", pixel_count=4, area_fraction=1.0, present=True)],
    )
    parsing = fp.ParsingData(
        parse_window=ParseWindow(x=0, y=0, size=2), includes_masks=True, regions=layer.regions, label_map=layer
    )
    zone = Zone(
        id="forehead", name="Forehead", group="face", side=None, sources=["segmentation"], status="ok",
        clipped_to=[], bbox=ZoneBBox(x=0, y=0, width=1, height=1), pixel_count=1,
        mask=ZoneMask(png_base64="AAAA"),
    )
    zones = fp.ZonesData(zones=[zone], stages=[Stage(number=1, key="base", name="Base", layers={"foundation": ["forehead"]})])
    return fp.ComponentSet(
        landmarks=fp.Component[fp.LandmarksData](meta=_meta("landmarks"), data=lm),
        frame=fp.Component[fp.FaceFrameData](meta=_meta("frame"), data=frame),
        parsing=fp.Component[fp.ParsingData](meta=_meta("parsing"), data=parsing),
        zones=fp.Component[fp.ZonesData](meta=_meta("zones"), data=zones),
    )


def _profile() -> fp.FacialProfile:
    return fp.FacialProfile(
        status="ok",
        image=ImageSize(width=100, height=100),
        components=_components(),
        quality=fp.ProfileQuality(
            alignment=AlignmentCheck(passed=True, fraction=1.0),
            warnings=[SegmentationWarning(code="head_tilt", message="tilted")],
            retake_suggested=["glasses"],
        ),
        timings_ms={"total": 1.0},
    )


def test_round_trip():
    profile = _profile()
    again = fp.FacialProfile.model_validate_json(profile.model_dump_json())
    assert again == profile
    assert again.schema_version == "1.0"


def test_data_status_invariant():
    with pytest.raises(ValidationError):
        fp.Component[fp.ZonesData](meta=_meta("zones", "unavailable"), data=fp.ZonesData(zones=[], stages=[]))
    with pytest.raises(ValidationError):
        fp.Component[fp.ZonesData](meta=_meta("zones", "ok"), data=None)
    fp.Component[fp.ZonesData](meta=_meta("zones", "unavailable"), data=None)
    fp.Component[fp.ZonesData](meta=_meta("zones", "degraded"), data=fp.ZonesData(zones=[], stages=[]))


def test_non_ok_status_has_no_components():
    ok = fp.FacialProfile(status="no_face_detected", image=ImageSize(width=1, height=1))
    assert ok.components is None
    with pytest.raises(ValidationError):
        fp.FacialProfile(status="no_face_detected", image=ImageSize(width=1, height=1), components=_components())


def test_confidence_range():
    with pytest.raises(ValidationError):
        fp.ComponentMeta(component_version=1, status="ok", confidence=1.5, provider=_provider("zones"))


def test_versions():
    assert ZONES_ALGORITHM_VERSION == "phase4-70zones-1"
    assert set(COMPONENT_VERSIONS) == {"landmarks", "frame", "parsing", "zones"}


@pytest.mark.parametrize("v", ["1.0", "1.7"])
def test_compat_accepts(v):
    assert compat.is_supported(v)
    compat.check({"schema_version": v})


@pytest.mark.parametrize("v", ["2.0", "0.9"])
def test_compat_rejects_unsupported(v):
    assert not compat.is_supported(v)
    with pytest.raises(compat.UnsupportedProfileVersion):
        compat.check({"schema_version": v})


@pytest.mark.parametrize("v", ["", "1", "1.", "a.b", "1.0.0", "-1.0", "1.0 "])
def test_compat_rejects_malformed(v):
    with pytest.raises(ValueError):
        compat.parse_version(v)
    with pytest.raises(compat.UnsupportedProfileVersion):
        compat.check({"schema_version": v})


def test_frozen_mappings():
    with pytest.raises(TypeError):
        COMPONENT_VERSIONS["zones"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        compat.SUPPORTED[2] = 0  # type: ignore[index]


@pytest.mark.parametrize("v", ["1" * 10000, "١.٠", "1" * 9 + ".0"])
def test_hostile_version_strings(v):
    with pytest.raises(ValueError):
        compat.parse_version(v)
    with pytest.raises(compat.UnsupportedProfileVersion):
        compat.check({"schema_version": v})


@pytest.mark.parametrize("v", [None, 1, 1.0])
def test_non_str_version(v):
    with pytest.raises(ValueError):
        compat.parse_version(v)  # type: ignore[arg-type]
    with pytest.raises(compat.UnsupportedProfileVersion):
        compat.check({"schema_version": v})


@pytest.mark.parametrize("p", [[], "1.0", None, 5])
def test_check_non_mapping(p):
    with pytest.raises(compat.UnsupportedProfileVersion):
        compat.check(p)  # type: ignore[arg-type]


def test_check_missing_version():
    with pytest.raises(compat.UnsupportedProfileVersion):
        compat.check({})


def _stage(n: int) -> Stage:
    return Stage(number=n, key=f"s{n}", name=f"Stage {n}", layers={"foundation": []})


def _zone(n: int) -> Zone:
    return Zone(
        id=f"z{n}", name="Z", group="face", side=None, sources=["segmentation"], status="ok",
        clipped_to=[], bbox=None, pixel_count=0, mask=None,
    )


def _length_error_locs(exc_info) -> list[tuple]:
    return [e["loc"] for e in exc_info.value.errors() if e["type"] == "too_long"]


def test_over_limit_lists_rejected():
    lm = Landmark(index=0, x=0, y=0, z=0)
    th = DetectionThresholds(min_face_detection_confidence=0.5, min_face_presence_confidence=0.5)
    with pytest.raises(ValidationError) as e:
        fp.LandmarksData(thresholds=th, landmarks=[lm] * 479)
    assert ("landmarks",) in _length_error_locs(e)
    with pytest.raises(ValidationError) as e:
        fp.FaceFrameData(ipd_px=1, roll_deg=0, u=(1, 0), v=(0, 1), midline=[(0.0, 0.0)] * 65)
    assert ("midline",) in _length_error_locs(e)
    with pytest.raises(ValidationError) as e:
        fp.ZonesData(zones=[], stages=[_stage(n) for n in range(1, 34)])
    assert _length_error_locs(e) == [("stages",)]
    with pytest.raises(ValidationError) as e:
        fp.ZonesData(zones=[_zone(n) for n in range(129)], stages=[])
    assert _length_error_locs(e) == [("zones",)]
    with pytest.raises(ValidationError) as e:
        fp.ProfileQuality(warnings=[SegmentationWarning(code="head_tilt", message="x")] * 33)
    assert _length_error_locs(e) == [("warnings",)]
    with pytest.raises(ValidationError) as e:
        fp.FacialProfile(
            status="no_face_detected", image=ImageSize(width=1, height=1),
            timings_ms={str(i): 1.0 for i in range(33)},
        )
    assert _length_error_locs(e) == [("timings_ms",)]


def test_at_limit_lists_accepted():
    lm = Landmark(index=0, x=0, y=0, z=0)
    th = DetectionThresholds(min_face_detection_confidence=0.5, min_face_presence_confidence=0.5)
    fp.LandmarksData(thresholds=th, landmarks=[lm] * 478)
    fp.FaceFrameData(ipd_px=1, roll_deg=0, u=(1, 0), v=(0, 1), midline=[(0.0, 0.0)] * 64)
    fp.ZonesData(zones=[_zone(n) for n in range(128)], stages=[_stage(n) for n in range(1, 33)])


def test_string_length_cap():
    with pytest.raises(ValidationError):
        fp.ProviderInfo(id="x" * 513, kind="zones", algorithm_version="1")
    fp.ProviderInfo(id="x" * 512, kind="zones", algorithm_version="1")
    with pytest.raises(ValidationError):
        fp.FacialProfile(status="no_face_detected", image=ImageSize(width=1, height=1), message="m" * 513)


def test_bad_sha256_rejected():
    with pytest.raises(ValidationError):
        fp.ProviderInfo(id="x", kind="zones", algorithm_version="1", weights_sha256="abc")
    fp.ProviderInfo(id="x", kind="zones", algorithm_version="1", weights_sha256="a" * 64)


def test_negative_confidence_rejected():
    with pytest.raises(ValidationError):
        fp.ComponentMeta(component_version=1, status="ok", confidence=-0.1, provider=_provider("zones"))


def test_ok_without_components_rejected():
    with pytest.raises(ValidationError):
        fp.FacialProfile(status="ok", image=ImageSize(width=1, height=1))


def test_retake_suggested_requires_ok():
    with pytest.raises(ValidationError):
        fp.FacialProfile(
            status="no_face_detected", image=ImageSize(width=1, height=1),
            quality=fp.ProfileQuality(retake_suggested=["glasses"]),
        )


_SLOT_KINDS = {"landmarks": "landmarks", "frame": "landmarks", "parsing": "parsing", "zones": "zones"}
_ALL_KINDS = ["landmarks", "parsing", "zones"]


@pytest.mark.parametrize("slot", list(_SLOT_KINDS))
def test_provider_kind_per_slot(slot):
    expected = _SLOT_KINDS[slot]
    good = _components().model_dump()
    good[slot]["meta"]["provider"]["kind"] = expected
    fp.ComponentSet.model_validate(good)
    for wrong in _ALL_KINDS:
        if wrong == expected:
            continue
        bad = _components().model_dump()
        bad[slot]["meta"]["provider"]["kind"] = wrong
        with pytest.raises(ValidationError):
            fp.ComponentSet.model_validate(bad)


@pytest.mark.parametrize(
    "value",
    ["A" * 64, "a" * 63, "a" * 65, "g" * 64, "a" * 63 + "-", "a" * 63 + "\n"],
)
def test_weights_sha256_rejected(value):
    with pytest.raises(ValidationError):
        fp.ProviderInfo(id="x", kind="zones", algorithm_version="1", weights_sha256=value)


@pytest.mark.parametrize("value", ["a" * 64, "0123456789abcdef" * 4])
def test_weights_sha256_accepted(value):
    assert fp.ProviderInfo(id="x", kind="zones", algorithm_version="1", weights_sha256=value).weights_sha256 == value


def _str_field_model(field: str, n: int):
    text = "x" * n
    if field in ("id", "algorithm_version", "license_warning"):
        kwargs = {"id": "x", "kind": "zones", "algorithm_version": "1"}
        kwargs[field] = text
        return fp.ProviderInfo(**kwargs)
    if field == "reason":
        return fp.ComponentMeta(component_version=1, status="unavailable", provider=_provider("zones"), reason=text)
    if field == "message":
        return fp.FacialProfile(status="no_face_detected", image=ImageSize(width=1, height=1), message=text)
    raise AssertionError(field)


@pytest.mark.parametrize("field", ["id", "algorithm_version", "license_warning", "reason", "message"])
def test_string_cap_boundary(field):
    _str_field_model(field, 512)
    with pytest.raises(ValidationError):
        _str_field_model(field, 513)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_confidence_bounds_accepted(value):
    assert fp.ComponentMeta(component_version=1, status="ok", confidence=value, provider=_provider("zones")).confidence == value


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_confidence_bounds_rejected(value):
    with pytest.raises(ValidationError):
        fp.ComponentMeta(component_version=1, status="ok", confidence=value, provider=_provider("zones"))


def test_check_ignores_unknown_fields():
    d = json.loads(_profile().model_dump_json())
    d["future_field"] = {"x": 1}
    compat.check(d)


def _field_names(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                yield from value
            yield from _field_names(value)
    elif isinstance(node, list):
        for item in node:
            yield from _field_names(item)


def test_schema_has_no_model_class_ids():
    names = list(_field_names(fp.FacialProfile.model_json_schema()))
    assert names
    assert not [n for n in names if "class_id" in n or "model_id_to" in n]
    source = Path(fp.__file__).read_text()
    assert "app.services.segmentation" not in source
