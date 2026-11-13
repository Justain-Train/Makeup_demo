import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from app.schemas.adapted_makeup_plan import Adjustment, ZoneSelection
from app.schemas.face_segmentation import Zone, ZoneBBox, ZoneMask
from app.schemas.facial_profile import (
    Classified,
    Component,
    ComponentMeta,
    ComponentSet,
    FacialProfile,
    FeaturesData,
    ImageSize,
    ParsingData,
    ProviderInfo,
    ZonesData,
)
from app.schemas.makeup_blueprint import BlueprintComponent, MakeupBlueprint
from app.schemas.makeup_component import MakeupComponent
from app.services.application_zones import catalog
from app.services.makeup_adaptation import rules
from app.services.makeup_adaptation.adapter import adapt
from app.services.makeup_adaptation.masks import union_zone_masks
from app.services.makeup_mapping import map_zones

W = H = 300
ALL_ZONES = set(catalog.ZONE_SPEC_BY_ID)

NEUTRAL = {
    "eye_spacing": "average", "eye_tilt": "neutral", "eye_shape": "almond", "lip_fullness": "medium",
    "lip_balance": "balanced", "lip_width": "average", "brow_arch": "soft_arched", "brow_thickness": "medium",
}


def _features(feature=None, label=None, conf=0.9, measurements=None):
    vals = {k: Classified(label=v, confidence=0.9) for k, v in NEUTRAL.items()}
    vals["face_shape"] = Classified(label="oval", confidence=0.9)
    if feature:
        vals[feature] = Classified(label=label, confidence=conf, measurements=measurements or {})
    return FeaturesData(**vals)


def _comp(pt, cat):
    return MakeupComponent(category=cat, product_type=pt, color_hex="#C0305A", intensity=0.6, finish="matte")


CASES = [
    ("eye_tilt_down_lift", "eye_tilt", "downturned", "eyeliner", "eye", 0.3, None),
    ("eye_tilt_up_balance", "eye_tilt", "upturned", "eyeliner", "eye", 0.3, None),
    ("eye_close_set_outer", "eye_spacing", "close", "eyeshadow", "eye", 0.4, None),
    ("eye_wide_set_inner", "eye_spacing", "wide", "eyeshadow", "eye", 0.4, None),
    ("eye_round_elongate", "eye_shape", "round", "eyeliner", "eye", 0.3, None),
    ("face_round_contour", "face_shape", "round", "contour_powder", "bronzer_contour", 0.4, None),
    ("face_round_contour", "face_shape", "round", "powder_blush", "blush", 0.4, None),
    ("face_oblong_horizontal", "face_shape", "oblong", "powder_blush", "blush", 0.4, None),
    ("face_oblong_horizontal", "face_shape", "oblong", "contour_cream", "bronzer_contour", 0.4, None),
    ("face_square_soften", "face_shape", "square", "contour_powder", "bronzer_contour", 0.4, None),
    ("face_heart_balance", "face_shape", "heart", "powder_blush", "blush", 0.4, None),
    ("face_heart_balance", "face_shape", "heart", "bronzer", "bronzer_contour", 0.4, None),
    ("lip_thin_full", "lip_fullness", "thin", "lip_liner", "lip", 0.5, None),
    ("lip_bottom_heavy_balance", "lip_balance", "bottom_heavy", "lipstick", "lip", 0.4, {"upper_to_lower": 0.5}),
    ("brow_straight_keep", "brow_arch", "straight", "brow_pencil", "eyebrows", 0.3, None),
]


def _outcome(res, rule_id):
    return next(e for e in res[0] if e.rule_id == rule_id).outcome


def _run(feats, pt, cat, avail=ALL_ZONES):
    c = _comp(pt, cat)
    kept = list(map_zones(pt, c.placement, c.target_zones).zone_ids)
    return rules.evaluate(feats, c, kept, avail), kept


@pytest.mark.parametrize("rid,feat,label,pt,cat,thr,meas", CASES)
def test_threshold_boundary(rid, feat, label, pt, cat, thr, meas):
    res, _ = _run(_features(feat, label, thr, meas), pt, cat)
    assert len(res[0]) == len(rules.RULES) == 12
    assert _outcome(res, rid) == "applied"
    res, _ = _run(_features(feat, label, thr - 0.01, meas), pt, cat)
    assert _outcome(res, rid) == "skipped_uncertain"


def test_unknown_label_uncertain():
    res, _ = _run(_features("eye_tilt", "unknown", 0.99), "eyeliner", "eye")
    assert _outcome(res, "eye_tilt_down_lift") == "skipped_uncertain"


def test_rule11_ratio_and_no_zone_change():
    f = lambda r: _features("lip_balance", "bottom_heavy", 0.9, {"upper_to_lower": r})  # noqa: E731
    res, _ = _run(f(0.55), "lipstick", "lip")
    assert _outcome(res, "lip_bottom_heavy_balance") == "applied" and res[2] is None
    assert any(a.effect == "emphasis" and "cupid_bow" in a.emphasis_zones for a in res[1])
    res, _ = _run(f(0.56), "lipstick", "lip")
    assert _outcome(res, "lip_bottom_heavy_balance") == "skipped_not_triggered"


def test_lipstick_rule10_not_applicable():
    res, _ = _run(_features("lip_fullness", "thin", 0.9), "lipstick", "lip")
    assert _outcome(res, "lip_thin_full") == "skipped_not_applicable"


def test_zone_unavailable_when_zone_missing_on_face():
    avail = ALL_ZONES - {"lip_outer_boundary"}
    res, _ = _run(_features("lip_fullness", "thin", 0.9), "lip_liner", "lip", avail)
    assert _outcome(res, "lip_thin_full") == "skipped_zone_unavailable" and res[2] is None


def test_zones_stay_in_layer():
    for rid, feat, label, pt, cat, thr, meas in CASES:
        res, _ = _run(_features(feat, label, 0.95, meas), pt, cat)
        if res[2] is not None:
            assert set(res[2]) <= set(map_zones(pt, None, ()).zone_ids)


def test_conflict_second_direction():
    res, _ = _run(_features("eye_tilt", "downturned", 0.9), "eyeliner", "eye")
    # rules 1 and 5 both direction on eyeliner; only one can fire
    f = _features("eye_tilt", "downturned", 0.9).model_copy(update={"eye_shape": Classified(label="round", confidence=0.9)})
    res, _ = _run(f, "eyeliner", "eye")
    assert _outcome(res, "eye_tilt_down_lift") == "applied"
    assert _outcome(res, "eye_round_elongate") == "skipped_conflict"
    assert len([a for a in res[1] if a.effect == "direction"]) == 1


def test_deterministic():
    f = _features("face_shape", "round", 0.9)
    assert _run(f, "contour_powder", "bronzer_contour")[0] == _run(f, "contour_powder", "bronzer_contour")[0]


def test_no_eyelid_type_in_service():
    root = Path(__file__).resolve().parents[1] / "app" / "services" / "makeup_adaptation"
    for p in root.glob("*.py"):
        assert "eyelid_type" not in p.read_text()


def test_reason_rejects_arbitrary_strings():
    with pytest.raises(ValidationError):
        Adjustment(rule_id="x", reason="free text", effect="zones")
    with pytest.raises(ValidationError):
        ZoneSelection(source="none", reason="free text")


def _png_b64(arr):
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def _zone(zid, rect, b64=None):
    x, y, w, h = rect
    arr = np.full((h, w), 255, dtype=np.uint8)
    return Zone(
        id=zid, name=zid, group="g", side=None, sources=["landmarks"], status="ok", confidence=0.9,
        clipped_to=[], bbox=ZoneBBox(x=x, y=y, width=w, height=h), pixel_count=w * h,
        mask=ZoneMask(png_base64=b64 or _png_b64(arr)),
    )


def _meta(kind, status="ok"):
    return ComponentMeta(component_version=1, status=status, provider=ProviderInfo(id="t", kind=kind, algorithm_version="1"))


def _profile(feats):
    zones = [_zone(z, (10 + (i % 20) * 10, 10 + (i // 20) * 10, 8, 8)) for i, z in enumerate(catalog.ZONE_SPEC_BY_ID)]
    comps = ComponentSet(
        landmarks=Component(meta=_meta("landmarks", "unavailable")),
        frame=Component(meta=_meta("landmarks", "unavailable")),
        parsing=Component(meta=_meta("parsing"), data=ParsingData(includes_masks=True, regions=[])),
        zones=Component(meta=_meta("zones"), data=ZonesData(zones=zones, stages=[])),
        features=None if feats is None else Component(meta=_meta("features"), data=feats),
    )
    return FacialProfile(status="ok", image=ImageSize(width=W, height=H), components=comps)


def _bp(*cs):
    from app.schemas.makeup_blueprint import ATTRIBUTABLE_FIELDS

    def _sources(c):
        out = {}
        for name in ATTRIBUTABLE_FIELDS:
            v = getattr(c, name)
            if (v if name == "target_zones" else v is not None):
                out[name] = "cv"
        return out

    comps = [BlueprintComponent(component=c, sources=_sources(c)) for c in cs]
    return MakeupBlueprint(status="ok", image=ImageSize(width=200, height=200), components=comps)


def test_adapt_applies_and_preserves_component():
    c = _comp("contour_powder", "bronzer_contour")
    fixed_clock = lambda: 0.0  # noqa: E731 - timings are wall-clock; determinism is about the plan content
    plan = adapt(_profile(_features("face_shape", "round", 0.9)), _bp(c), clock=fixed_clock)
    out = plan.components[0]
    assert out.component.color_hex == c.color_hex and out.component.finish == c.finish
    assert out.component.intensity == c.intensity
    assert out.zones and set(out.zones) <= set(map_zones("contour_powder", None, ()).zone_ids)
    assert out.zone_selection.source == "placement" and out.zone_selection.reason == "contour_round_face"
    assert len(out.rule_evaluations) == 12
    assert plan == adapt(_profile(_features("face_shape", "round", 0.9)), _bp(c), clock=fixed_clock)


def test_features_absent_warns():
    plan = adapt(_profile(None), _bp(_comp("lipstick", "lip")))
    assert any(w.code == "features_unavailable" for w in plan.warnings)
    assert plan.components[0].adjustments == [] and plan.components[0].rule_evaluations == []


def test_mask_corrupt_rgb_16bit_oversized():
    size = ImageSize(width=W, height=H)
    good = _png_b64(np.full((8, 8), 255, np.uint8))
    raw = bytearray(base64.b64decode(good))
    raw[40:44] = b"\x00\x00\x00\x00"
    corrupt = base64.b64encode(bytes(raw)).decode()
    rgb = _png_b64(np.full((8, 8, 3), 255, np.uint8))
    deep = _png_b64(np.full((8, 8), 60000, np.uint16))
    for b64 in (corrupt, rgb, deep):
        res = union_zone_masks([_zone("z", (0, 0, 8, 8), b64)], size)
        assert res is not None and res.invalid_ids == ("z",)
    big = _zone("z", (0, 0, 400, 8), _png_b64(np.full((8, 400), 255, np.uint8)))
    res = union_zone_masks([big], size)
    assert res is not None and res.invalid_ids == ("z",)
