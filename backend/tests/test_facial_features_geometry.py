"""Geometric feature classifiers (heuristic, uncalibrated) and the 1.1 features schema."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from app.schemas import facial_profile as fp
from app.services.application_zones.frame import FaceFrame
from app.services.facial_features import brow, eye, face_shape, lip
from app.services.facial_features import thresholds as th
from app.services.facial_profile import compat
from app.services.facial_profile.versions import (
    COMPONENT_VERSIONS,
    FEATURES_ALGORITHM_VERSION,
    PROFILE_SCHEMA_VERSION,
)
from tests.synthetic import fixture_landmarks
from tests.test_facial_profile_schema import _components, _profile


def _mid(edges):
    return (edges[0] + edges[1]) / 2.0


MID_SPACING = _mid(th.EYE_SPACING_EDGES)
MID_TILT = _mid(th.EYE_TILT_EDGES_DEG)
MID_ARCH = _mid(th.BROW_ARCH_EDGES)


def make_face(
    spacing=MID_SPACING, tilts=(MID_TILT, MID_TILT), aperture=0.28, upper=0.11, lower=0.17, mouth=0.8,
    arch=MID_ARCH, thick=0.24,
):
    """Place eye, lip and brow points in frame coordinates on top of the real fixture landmarks."""
    base, _, _ = fixture_landmarks()
    pts = base.copy()
    frame = FaceFrame.from_landmarks(pts)
    ipd, u, v = frame.ipd, frame.u, frame.v
    mid = (pts[468] + pts[473]) / 2.0
    mu, mv = float(mid @ u), float(mid @ v)
    width = 0.4 * ipd

    def put(index, du, dv):
        pts[index] = du * u + dv * v

    for side, sign, tilt in (("right", -1.0, tilts[0]), ("left", 1.0, tilts[1])):
        inner_i, outer_i, top_i, bottom_i = frame.indices([133, 33, 159, 145], side)
        o_in = spacing * width / 2.0
        o_out = o_in + width * math.cos(math.radians(tilt))
        v_in = mv
        v_out = v_in - width * math.sin(math.radians(tilt))
        put(inner_i, mu + sign * o_in, v_in)
        put(outer_i, mu + sign * o_out, v_out)
        o_mid = (o_in + o_out) / 2.0
        v_mid = (v_in + v_out) / 2.0
        put(top_i, mu + sign * o_mid, v_mid - aperture * width / 2.0)
        put(bottom_i, mu + sign * o_mid, v_mid + aperture * width / 2.0)
        brow_v = mv - 0.35 * ipd
        for k, (up_i, lo_i) in enumerate(
            zip(frame.indices(brow.BROW_UPPER, side), frame.indices(brow.BROW_LOWER, side))
        ):
            frac = k / 4.0
            o = o_out + (o_in - o_out) * frac
            bump = (0.0, 0.7, 1.0, 0.7, 0.0)[k]
            put(up_i, mu + sign * o, brow_v - arch * width * bump)
            put(lo_i, mu + sign * o, brow_v - arch * width * bump + thick * width)
    mouth_v = mv + 1.0 * ipd
    put(0, mu, mouth_v)
    put(13, mu, mouth_v + upper * ipd)
    put(14, mu, mouth_v + upper * ipd + 0.01 * ipd)
    put(17, mu, mouth_v + (upper + lower) * ipd + 0.01 * ipd)
    put(61, mu - mouth * ipd / 2.0, mouth_v + 0.1 * ipd)
    put(291, mu + mouth * ipd / 2.0, mouth_v + 0.1 * ipd)
    return pts, FaceFrame.from_landmarks(pts)


def rotate(pts, degrees):
    a = math.radians(degrees)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = pts.mean(axis=0)
    out = (pts - c) @ r.T + c
    return out, FaceFrame.from_landmarks(out)


ALL = [
    eye.classify_eye_spacing, eye.classify_eye_tilt, eye.classify_eye_shape,
    lip.classify_lip_fullness, lip.classify_lip_balance, lip.classify_lip_width,
    brow.classify_brow_arch, brow.classify_brow_thickness,
]


@pytest.mark.parametrize(
    "spacing,label",
    [
        (th.EYE_SPACING_EDGES[0] - 0.1, "close"),
        (MID_SPACING, "average"),
        (th.EYE_SPACING_EDGES[1] + 0.1, "wide"),
    ],
)
def test_eye_spacing(spacing, label):
    assert eye.classify_eye_spacing(*make_face(spacing=spacing)).label == label


@pytest.mark.parametrize(
    "tilt,label",
    [
        (th.EYE_TILT_EDGES_DEG[1] + 6.0, "upturned"),
        (MID_TILT, "neutral"),
        (th.EYE_TILT_EDGES_DEG[0] - 5.0, "downturned"),
    ],
)
def test_eye_tilt_and_roll_invariance(tilt, label):
    pts, frame = make_face(tilts=(tilt, tilt))
    result = eye.classify_eye_tilt(pts, frame)
    assert result.label == label
    assert result.measurements["right_tilt_deg"] == pytest.approx(result.measurements["left_tilt_deg"], abs=0.1)
    rpts, rframe = rotate(pts, 15.0)
    rotated = eye.classify_eye_tilt(rpts, rframe)
    assert rotated.label == label
    assert rotated.measurements["tilt_deg"] == pytest.approx(result.measurements["tilt_deg"], abs=0.1)


def test_eye_tilt_disagreement_lowers_confidence():
    hi = th.EYE_TILT_EDGES_DEG[1] + 6.0
    same = eye.classify_eye_tilt(*make_face(tilts=(hi, hi)))
    differ = eye.classify_eye_tilt(*make_face(tilts=(hi + 4.0, hi - 4.0)))
    assert differ.confidence < same.confidence


@pytest.mark.parametrize("aperture,label", [(0.25, "almond"), (0.45, "round")])
def test_eye_shape(aperture, label):
    assert eye.classify_eye_shape(*make_face(aperture=aperture)).label == label


@pytest.mark.parametrize("upper,lower,label", [(0.08, 0.10, "thin"), (0.11, 0.17, "medium"), (0.17, 0.23, "full")])
def test_lip_fullness(upper, lower, label):
    pts, frame = make_face(upper=upper, lower=lower)
    assert lip.classify_lip_fullness(pts, frame).label == label


@pytest.mark.parametrize(
    "upper,lower,label", [(0.10, 0.20, "bottom_heavy"), (0.15, 0.15, "balanced"), (0.22, 0.12, "top_heavy")]
)
def test_lip_balance(upper, lower, label):
    assert lip.classify_lip_balance(*make_face(upper=upper, lower=lower)).label == label


@pytest.mark.parametrize("mouth,label", [(0.6, "narrow"), (0.8, "average"), (1.0, "wide")])
def test_lip_width(mouth, label):
    assert lip.classify_lip_width(*make_face(mouth=mouth)).label == label


@pytest.mark.parametrize(
    "arch,label",
    [
        (th.BROW_ARCH_EDGES[0] - 0.06, "straight"),
        (MID_ARCH, "soft_arched"),
        (th.BROW_ARCH_EDGES[1] + 0.06, "high_arched"),
    ],
)
def test_brow_arch(arch, label):
    assert brow.classify_brow_arch(*make_face(arch=arch)).label == label


@pytest.mark.parametrize("thick", [0.10, 0.24, 0.40])
def test_brow_thickness_is_not_measurable(thick):
    result = brow.classify_brow_thickness(*make_face(thick=thick))
    assert result.label == "unknown"
    assert result.confidence == 0.0
    assert result.reason == "not_measurable_from_landmarks"
    assert result.measurements["thickness_ratio"] > 0
    assert "right_thickness_ratio" in result.measurements and "left_thickness_ratio" in result.measurements


def test_recalibrated_edges_match_calibration_file():
    path = Path(__file__).resolve().parents[2] / "docs" / "evaluation" / "geometry_calibration_2026-11-12.json"
    if not path.exists():
        pytest.skip("calibration file missing")
    m = json.loads(path.read_text())["measurements"]
    cases = [
        (th.EYE_SPACING_EDGES, m["classify_eye_spacing"]["spacing_ratio"]),
        (th.EYE_TILT_EDGES_DEG, m["classify_eye_tilt"]["tilt_deg"]),
        (th.BROW_ARCH_EDGES, m["classify_brow_arch"]["arch_ratio"]),
    ]
    for edges, stats in cases:
        assert edges[0] == pytest.approx(stats["p20"], abs=0.01)
        assert edges[1] == pytest.approx(stats["p80"], abs=0.01)


def test_inside_margin_is_unknown_with_low_confidence():
    result = eye.classify_eye_spacing(*make_face(spacing=th.EYE_SPACING_EDGES[1] + 0.001))
    assert result.label == "unknown"
    assert result.confidence < 0.1
    assert result.reason == "within_ambiguity_margin"


@pytest.mark.parametrize("fn", ALL)
def test_nan_input_is_unknown_and_does_not_raise(fn):
    pts, frame = make_face()
    bad = np.full_like(pts, np.nan)
    result = fn(bad, frame)
    assert result.label == "unknown"
    assert result.confidence == 0.0
    assert result.reason


@pytest.mark.parametrize("fn", ALL)
def test_degenerate_input_is_unknown(fn):
    pts, frame = make_face()
    degenerate = np.tile(pts[468], (478, 1))
    result = fn(degenerate, frame)
    assert result.label == "unknown"
    assert result.confidence == 0.0
    assert result.reason
    assert fn(pts[:10], frame).label == "unknown"


@pytest.mark.parametrize("fn", ALL)
def test_every_label_has_measurements_and_valid_confidence(fn):
    result = fn(*make_face())
    assert result.measurements
    assert 0.0 <= result.confidence <= 1.0
    assert all(math.isfinite(x) for x in result.measurements.values())


def test_eyelid_type_is_always_unknown():
    e = fp.EyelidType()
    assert (e.value, e.source) == ("unknown", "unknown")


def _features_data() -> fp.FeaturesData:
    pts, frame = make_face()
    values = {
        "eye_spacing": eye.classify_eye_spacing(pts, frame),
        "eye_tilt": eye.classify_eye_tilt(pts, frame),
        "eye_shape": eye.classify_eye_shape(pts, frame),
        "lip_fullness": lip.classify_lip_fullness(pts, frame),
        "lip_balance": lip.classify_lip_balance(pts, frame),
        "lip_width": lip.classify_lip_width(pts, frame),
        "brow_arch": brow.classify_brow_arch(pts, frame),
        "brow_thickness": brow.classify_brow_thickness(pts, frame),
    }
    return fp.FeaturesData.model_validate({k: v.model_dump() for k, v in values.items()})


def _features_component(kind="features") -> fp.Component[fp.FeaturesData]:
    meta = fp.ComponentMeta(
        component_version=COMPONENT_VERSIONS["features"],
        status="ok",
        confidence=0.5,
        provider=fp.ProviderInfo(id="features-test", kind=kind, algorithm_version=FEATURES_ALGORITHM_VERSION),
    )
    return fp.Component[fp.FeaturesData](meta=meta, data=_features_data())


def test_compat_accepts_1_0_and_1_1_and_rejects_2_0():
    compat.check({"schema_version": "1.0"})
    compat.check({"schema_version": "1.1"})
    with pytest.raises(compat.UnsupportedProfileVersion):
        compat.check({"schema_version": "2.0"})
    assert PROFILE_SCHEMA_VERSION == "1.1"


def test_profile_without_features_validates():
    profile = _profile()
    assert profile.components is not None and profile.components.features is None
    payload = profile.model_dump(mode="json")
    payload["schema_version"] = "1.0"
    assert "features" not in payload["components"]  # absent features are omitted, never null
    again = fp.FacialProfile.model_validate(payload)
    assert again.schema_version == "1.0"
    assert again.components is not None and again.components.features is None


def test_profile_with_features_round_trips():
    profile = _profile()
    assert profile.components is not None
    profile.components.features = _features_component()
    again = fp.FacialProfile.model_validate_json(profile.model_dump_json())
    assert again == profile
    assert again.schema_version == "1.1"


SHAPES = ["oval", "round", "square", "heart", "oblong", "diamond"]


def make_face_shape(length, forehead, jaw, taper):
    """Symmetric face about the nose-tip u, built in frame coordinates on top of the fixture landmarks."""
    base, _, _ = fixture_landmarks()
    pts = base.copy()
    frame = FaceFrame.from_landmarks(pts)
    u, v = frame.u, frame.v
    mu = float(pts[1] @ u)
    width = abs(float(pts[454] @ u) - float(pts[234] @ u))
    v_jaw = float(pts[172] @ v)
    v_chin = float(pts[152] @ v)

    def put(index, du, dv):
        pts[index] = du * u + dv * v

    def pair(right, left, half, dv):
        put(right, mu - half, dv)
        put(left, mu + half, dv)

    put(152, mu, v_chin)
    put(10, mu, v_chin - length * width)
    pair(234, 454, width / 2.0, float(pts[234] @ v))
    pair(54, 284, forehead * width / 2.0, float(pts[54] @ v))
    jaw_half = jaw * width / 2.0
    pair(172, 397, jaw_half, v_jaw)
    pair(176, 400, taper * jaw_half, v_jaw + 0.05 * width)
    return pts, FaceFrame.from_landmarks(pts)


def _proto_face(name, other=None):
    vec = np.array(th.FACE_SHAPE_PROTOTYPES[name], dtype=float)
    if other is not None:
        vec = (vec + np.array(th.FACE_SHAPE_PROTOTYPES[other], dtype=float)) / 2.0
    else:
        vec = vec + 0.25 * th.FACE_SHAPE_MARGIN * np.array(th.FACE_SHAPE_DIM_SCALE) / 2.0
    return make_face_shape(*vec)


@pytest.mark.parametrize("name", SHAPES)
def test_face_shape_each_prototype(name):
    result = face_shape.classify_face_shape(*_proto_face(name))
    assert result.label == name
    assert 0.0 < result.confidence <= 0.8
    assert len(result.measurements) <= 16


def test_face_shape_midpoint_is_ambiguous():
    result = face_shape.classify_face_shape(*_proto_face("oval", "oblong"))
    assert result.label == "unknown" or result.confidence < 0.3
    assert "runner_up_index" in result.measurements
    assert "runner_up_distance" in result.measurements


def test_face_shape_roll_invariant():
    pts, frame = _proto_face("square")
    base = face_shape.classify_face_shape(pts, frame)
    rpts, rframe = rotate(pts, 15.0)
    rolled = face_shape.classify_face_shape(rpts, rframe)
    assert rolled.label == base.label
    assert rolled.confidence == pytest.approx(base.confidence, abs=1e-6)


def test_face_shape_yaw_lowers_confidence_then_unknown():
    pts, frame = _proto_face("oval")
    base = face_shape.classify_face_shape(pts, frame)
    width = base.measurements["cheek_width_px"]
    mild = pts.copy()
    mild[1] = mild[1] + 0.1 * width * frame.u
    assert face_shape.classify_face_shape(mild, frame).confidence < base.confidence
    strong = pts.copy()
    strong[1] = strong[1] + 0.25 * width * frame.u
    result = face_shape.classify_face_shape(strong, frame)
    assert result.label == "unknown"
    assert result.reason == "head_turned"
    assert result.measurements


def test_face_shape_degenerate_inputs_are_unknown():
    pts, frame = _proto_face("oval")
    nan = np.full_like(pts, np.nan)
    zero_width = pts.copy()
    zero_width[454] = zero_width[234]
    zero_length = pts.copy()
    zero_length[10] = zero_length[152]
    for bad in (nan, zero_width, zero_length):
        result = face_shape.classify_face_shape(bad, frame)
        assert result.label == "unknown"
        assert result.confidence == 0.0


@pytest.mark.parametrize("length", [0.8, 1.0, 1.2, 1.4])
@pytest.mark.parametrize("forehead", [0.6, 0.75])
@pytest.mark.parametrize("jaw", [0.6, 0.8])
@pytest.mark.parametrize("taper", [0.5, 0.7])
def test_face_shape_confidence_never_exceeds_cap(length, forehead, jaw, taper):
    result = face_shape.classify_face_shape(*make_face_shape(length, forehead, jaw, taper))
    assert 0.0 <= result.confidence <= 0.8


def _face_shape_data(label="round"):
    base = _features_data().model_dump()
    base["face_shape"] = {"label": label, "confidence": 0.5, "measurements": {"length_to_width": 1.0}}
    return base


def test_features_with_face_shape_round_trips():
    features = fp.FeaturesData.model_validate(_face_shape_data())
    again = fp.FeaturesData.model_validate_json(features.model_dump_json())
    assert again == features
    assert again.face_shape is not None and again.face_shape.label == "round"


def test_features_without_face_shape_omits_key():
    features = _features_data()
    assert features.face_shape is None
    assert "face_shape" not in features.model_dump()
    assert "face_shape" not in features.model_dump(mode="json")


def test_features_invalid_face_shape_label_rejected():
    with pytest.raises(ValidationError):
        fp.FeaturesData.model_validate(_face_shape_data("triangle"))


def test_oval_prototype_matches_calibration_means():
    path = Path(__file__).resolve().parents[2] / "docs" / "evaluation" / "face_shape_calibration_2026-11-12.json"
    if not path.exists():
        pytest.skip("calibration file missing")
    frontal = json.loads(path.read_text())["frontal"]
    keys = ["length_to_width", "forehead_54_to_cheek", "jaw_172_to_cheek", "taper_176_over_172"]
    means = []
    for key in keys:
        if key not in frontal:
            pytest.skip(f"calibration key {key} missing")
        means.append(frontal[key]["mean"])
    assert th.FACE_SHAPE_PROTOTYPES["oval"] == pytest.approx(tuple(means), abs=1e-3)


def test_fixture_face_is_inside_calibration_space():
    pts, _, _ = fixture_landmarks()
    m = face_shape.classify_face_shape(pts, FaceFrame.from_landmarks(pts)).measurements
    vec = [m["length_to_width"], m["forehead_to_cheek"], m["jaw_to_cheek"], m["chin_taper"]]
    z = [(x - mu) / s for x, mu, s in zip(vec, th.FACE_SHAPE_POP_MEAN, th.FACE_SHAPE_DIM_SCALE)]
    assert all(abs(v) <= 3.0 for v in z), z


def test_face_shape_on_fixture_landmarks():
    pts, _, _ = fixture_landmarks()
    result = face_shape.classify_face_shape(pts, FaceFrame.from_landmarks(pts))
    assert result.label in {*SHAPES, "unknown"}
    assert 0.0 <= result.confidence <= 0.8


def test_provider_kind_features_accepted_and_mismatch_rejected():
    fp.ProviderInfo(id="x", kind="features", algorithm_version="t")
    base = _components()
    with pytest.raises(ValidationError):
        fp.ComponentSet(
            landmarks=base.landmarks, frame=base.frame, parsing=base.parsing, zones=base.zones,
            features=_features_component(kind="zones"),
        )
