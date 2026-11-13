import math
from typing import get_args

import pytest

from app.schemas.adapted_makeup_plan import Adjustment, Direction
from app.schemas.facial_profile import (
    Component,
    ComponentMeta,
    ComponentSet,
    FaceFrameData,
    FacialProfile,
    ProviderInfo,
)
from app.services.makeup_adaptation import adapter
from app.services.makeup_adaptation.adapter import adapt
from app.services.makeup_adaptation.directions import DIRECTION_ELEVATION_DEG, attach_angles, usable_frame
from tests.test_makeup_adaptation_core import _blueprint, _lip, _profile


def _frame(roll_deg: float = 0.0, scale: float = 1.0) -> FaceFrameData:
    r = math.radians(roll_deg)
    u = (math.cos(r) * scale, math.sin(r) * scale)
    v = (-math.sin(r), math.cos(r))
    return FaceFrameData(ipd_px=100.0, roll_deg=roll_deg, u=u, v=v, midline=[])


def _adj(direction: str = "upward_outward", effect: str = "direction") -> Adjustment:
    return Adjustment(rule_id="r1", reason="lift_downturned_eyes", effect=effect, direction=direction)


def _profile_with_frame(frame: FaceFrameData | None) -> FacialProfile:
    base = _profile()
    comps = base.components
    assert comps is not None
    meta = ComponentMeta(
        component_version=1,
        status="ok" if frame is not None else "unavailable",
        provider=ProviderInfo(id="t", kind="landmarks", algorithm_version="1"),
    )
    new = ComponentSet(
        landmarks=comps.landmarks,
        frame=Component(meta=meta, data=frame),
        parsing=comps.parsing,
        zones=comps.zones,
    )
    return FacialProfile(status="ok", image=base.image, components=new)


def _angles(adj: Adjustment, frame: FaceFrameData):
    return {a.side: a for a in attach_angles([adj], frame)[0].direction_angles}


def test_table_covers_every_direction_label():
    assert set(DIRECTION_ELEVATION_DEG) == set(get_args(Direction))
    assert DIRECTION_ELEVATION_DEG == {"horizontal": 0.0, "outward": 10.0, "upward_outward": 25.0, "diagonal_up": 45.0}


@pytest.mark.parametrize("label,deg", list(DIRECTION_ELEVATION_DEG.items()))
def test_face_deg_same_both_sides_and_image_mirrors(label, deg):
    a = _angles(_adj(label), _frame())
    assert a["left"].face_deg == a["right"].face_deg == deg
    assert a["left"].image_deg == pytest.approx(deg)
    expected_right = 180.0 - deg
    assert a["right"].image_deg == pytest.approx(expected_right)


def test_roll_shifts_image_angle():
    a0 = _angles(_adj(), _frame(0.0))
    a1 = _angles(_adj(), _frame(10.0))
    assert a1["left"].face_deg == a0["left"].face_deg
    assert a1["left"].image_deg == pytest.approx(a0["left"].image_deg - 10.0)
    assert a1["right"].image_deg == pytest.approx(a0["right"].image_deg - 10.0)


def test_angles_in_range_and_only_direction_effects_touched():
    out = attach_angles([_adj(), _adj(effect="intensity")], _frame(179.0))
    assert len(out[0].direction_angles) == 2 and out[1].direction_angles == []
    for a in out[0].direction_angles:
        assert -180.0 < a.image_deg <= 180.0


def test_input_adjustments_not_mutated():
    src = _adj()
    attach_angles([src], _frame())
    assert src.direction_angles == []


@pytest.mark.parametrize(
    "frame",
    [
        None,
        FaceFrameData(ipd_px=1.0, roll_deg=0.0, u=(float("nan"), 0.0), v=(0.0, 1.0), midline=[]),
        FaceFrameData(ipd_px=1.0, roll_deg=float("inf"), u=(1.0, 0.0), v=(0.0, 1.0), midline=[]),
        _frame(scale=2.0),
        _frame(scale=0.0),
    ],
)
def test_unusable_frame_gives_no_angles(frame):
    assert usable_frame(_profile_with_frame(frame)) is None
    assert attach_angles([_adj()], None)[0].direction_angles == []


def _force_direction(monkeypatch):
    real = adapter.evaluate

    def fake(features, c, kept, available):
        evals, adjs, zones = real(features, c, kept, available)
        return evals, [*adjs, _adj()], zones

    monkeypatch.setattr(adapter, "evaluate", fake)


def test_missing_frame_adds_note_and_warning(monkeypatch):
    _force_direction(monkeypatch)
    plan = adapt(_profile_with_frame(None), _blueprint(_lip()))
    comp = plan.components[0]
    assert "direction_angles_unavailable" in comp.notes
    assert comp.adjustments[-1].direction == "upward_outward" and comp.adjustments[-1].direction_angles == []
    assert any(w.code == "direction_angles_unavailable" for w in plan.warnings)


def test_usable_frame_attaches_angles_without_note(monkeypatch):
    _force_direction(monkeypatch)
    plan = adapt(_profile_with_frame(_frame(5.0)), _blueprint(_lip()))
    comp = plan.components[0]
    assert "direction_angles_unavailable" not in comp.notes
    assert len(comp.adjustments[-1].direction_angles) == 2


def test_deterministic_and_colours_unchanged(monkeypatch):
    _force_direction(monkeypatch)
    profile, bp = _profile_with_frame(_frame(3.0)), _blueprint(_lip())
    a = adapt(profile, bp, clock=lambda: 0.0)
    b = adapt(profile, bp, clock=lambda: 0.0)
    assert a.model_dump() == b.model_dump()
    src = bp.components[0].component
    out = a.components[0].component
    assert out.color_hex == src.color_hex and out.intensity == src.intensity
    assert a.schema_version == "1.0"
