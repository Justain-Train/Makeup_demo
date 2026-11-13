import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.schemas.adapted_makeup_plan import AdaptedMakeupPlan
from app.schemas.face_segmentation import Zone, ZoneBBox, ZoneMask
from app.schemas.facial_profile import (
    Component,
    ComponentMeta,
    ComponentSet,
    FacialProfile,
    ImageSize,
    ParsingData,
    ProviderInfo,
    ZonesData,
)
from app.schemas.makeup_blueprint import BlueprintComponent, MakeupBlueprint
from app.schemas.makeup_component import MakeupComponent
from app.services.application_zones import catalog
from app.services.makeup_adaptation import versions
from app.services.makeup_adaptation.adapter import adapt
from app.services.makeup_adaptation.masks import union_zone_masks
from app.services.makeup_mapping import PLACEMENT_ZONES, map_zones

W = H = 300


def _png_b64(arr):
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def _zone(zid, rect, status="ok", crop=None):
    x, y, w, h = rect
    arr = np.full((h, w), 255, dtype=np.uint8) if crop is None else crop
    return Zone(
        id=zid, name=zid, group="g", side=None, sources=["landmarks"], status=status,
        confidence=0.9, clipped_to=[], bbox=ZoneBBox(x=x, y=y, width=w, height=h),
        pixel_count=int((arr > 0).sum()), mask=ZoneMask(png_base64=_png_b64(arr)),
    )


def _meta(kind, status="ok"):
    return ComponentMeta(
        component_version=1, status=status,
        provider=ProviderInfo(id="t", kind=kind, algorithm_version="1"),
    )


def _profile(overrides=None, zones_unavailable=False, status="ok"):
    size = ImageSize(width=W, height=H)
    if status != "ok":
        return FacialProfile(status=status, image=size)
    overrides = overrides or {}
    zones = []
    for i, zid in enumerate(catalog.ZONE_SPEC_BY_ID):
        zones.append(overrides.get(zid) or _zone(zid, (10 + (i % 20) * 10, 10 + (i // 20) * 10, 8, 8)))
    zdata = None if zones_unavailable else ZonesData(zones=zones, stages=[])
    comps = ComponentSet(
        landmarks=Component(meta=_meta("landmarks", "unavailable")),
        frame=Component(meta=_meta("landmarks", "unavailable")),
        parsing=Component(meta=_meta("parsing"), data=ParsingData(includes_masks=True, regions=[])),
        zones=Component(meta=_meta("zones", "unavailable" if zones_unavailable else "ok"), data=zdata),
    )
    return FacialProfile(status="ok", image=size, components=comps)


def _bc(product_type, category, **kw):
    comp = MakeupComponent(category=category, product_type=product_type, **kw)
    sources = {"category": "cv", "product_type": "cv"}
    if comp.color_hex:
        sources["color_hex"] = "cv"
    if comp.intensity is not None:
        sources["intensity"] = "cv"
    if comp.finish:
        sources["finish"] = "vlm"
    if comp.placement:
        sources["placement"] = "vlm"
    if comp.target_zones:
        sources["target_zones"] = "cv"
    return BlueprintComponent(component=comp, sources=sources)


def _blueprint(*comps, status="ok"):
    size = ImageSize(width=200, height=200)
    if status != "ok":
        return MakeupBlueprint(status=status, image=size)
    return MakeupBlueprint(status="ok", image=size, components=list(comps))


def _lip(**kw):
    return _bc("lipstick", "lip", color_hex="#C0305A", intensity=0.7, finish="matte", **kw)


def test_ok_plan_schema_and_reference():
    plan = adapt(_profile(), _blueprint(_lip(target_zones=["lip_full"])))
    assert plan.status == "ok" and plan.schema_version == versions.PLAN_SCHEMA_VERSION == "1.0"
    assert plan.reference is not None and plan.reference.component_count == 1
    assert plan.providers and "adapt" in plan.timings_ms
    AdaptedMakeupPlan.model_validate(plan.model_dump())


def test_version_compat():
    assert versions.is_supported("1.0") and versions.is_supported("1.3")
    assert not versions.is_supported("2.0")
    with pytest.raises(versions.UnsupportedPlanVersion):
        versions.check({"schema_version": "9.0"})
    versions.check({"schema_version": "1.0", "extra": 1})


def test_reference_zones_carried_over():
    plan = adapt(_profile(), _blueprint(_lip(target_zones=["lip_full"])))
    c = plan.components[0]
    assert c.zones == ["lip_full"] and c.zone_selection.source == "reference"
    assert c.mask is not None and c.bbox is not None and c.pixel_count == 64
    assert c.sources["target_zones"] == "cv"


def _placement_case():
    for pt, cat in (("lipstick", "lip"), ("powder_blush", "blush"), ("eyeshadow", "eye")):
        for placement in PLACEMENT_ZONES:
            try:
                comp = MakeupComponent(category=cat, product_type=pt, placement=placement)
            except ValueError:
                continue
            if map_zones(pt, comp.placement).source == "placement":
                return pt, cat, comp.placement
    pytest.skip("no placement-narrowed case available")


def test_vlm_placement_without_zones():
    pt, cat, placement = _placement_case()
    plan = adapt(_profile(), _blueprint(_bc(pt, cat, placement=placement)))
    c = plan.components[0]
    assert c.zone_selection.source == "placement" and c.zones
    assert c.sources["placement"] == "vlm"


def test_out_of_layer_zone_falls_back_to_layer_default():
    plan = adapt(_profile(), _blueprint(_lip(target_zones=["cheek_apple_l"])))
    c = plan.components[0]
    assert "reference_zones_outside_layer" in c.notes
    assert c.zone_selection.source == "layer_default"
    assert "cheek_apple_l" not in c.zones
    assert any(w.code == "reference_zones_outside_layer" for w in plan.warnings)


def _rand(rng, h, w):
    return (rng.random((h, w)) > 0.5).astype(np.uint8) * 255


def test_union_pixel_count_overlap_bbox_centroid():
    rng = np.random.default_rng(0)
    rects = [(10, 10, 40, 40), (30, 30, 40, 40), (-5, -5, 20, 20)]
    crops = [_rand(rng, r[3], r[2]) for r in rects]
    zones = [_zone(f"z{i}", r, crop=c) for i, (r, c) in enumerate(zip(rects, crops))]
    res = union_zone_masks(zones, ImageSize(width=W, height=H))
    assert res is not None and not res.invalid_ids
    full = np.zeros((H, W), dtype=bool)
    separate = 0
    for (x, y, w, h), c in zip(rects, crops):
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        piece = c[y0 - y : y1 - y, x0 - x : x1 - x] > 0
        separate += int(piece.sum())
        full[y0:y1, x0:x1] |= piece
    assert res.pixel_count == int(full.sum()) and res.pixel_count < separate
    bx, by, bw, bh = res.bbox
    assert bx >= 0 and by >= 0 and bx + bw <= W and by + bh <= H
    cx, cy = res.centroid
    assert bx <= cx <= bx + bw and by <= cy <= by + bh
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(res.png_b64), np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (bh, bw) and int((decoded > 0).sum()) == res.pixel_count
    ys, xs = np.nonzero(full)
    assert abs(cx - xs.mean()) < 1e-9 and abs(cy - ys.mean()) < 1e-9


def test_union_skips_unavailable_and_maskless():
    zones = [_zone("a", (10, 10, 5, 5), status="unavailable"), _zone("b", (20, 20, 5, 5))]
    res = union_zone_masks(zones, ImageSize(width=W, height=H))
    assert res is not None and res.pixel_count == 25
    assert union_zone_masks([zones[0]], ImageSize(width=W, height=H)) is None


def test_unavailable_zone_skipped_and_noted():
    prof = _profile({"lip_full": _zone("lip_full", (100, 200, 20, 10), status="unavailable")})
    plan = adapt(prof, _blueprint(_lip(target_zones=["lip_full"])))
    c = plan.components[0]
    assert c.zones == [] and c.mask is None and c.bbox is None and c.pixel_count == 0
    assert "zone_unavailable_on_user" in c.notes and c.zone_selection.source == "none"


def test_zones_component_unavailable():
    plan = adapt(_profile(zones_unavailable=True), _blueprint(_lip(target_zones=["lip_full"])))
    assert plan.status == "ok"
    assert plan.components[0].mask is None
    assert "zone_unavailable_on_user" in plan.components[0].notes


def test_invalid_mask_shape_noted():
    bad = _zone("lip_full", (100, 200, 20, 10), crop=np.full((8, 8), 255, dtype=np.uint8))
    bad = Zone(**{**bad.model_dump(), "bbox": ZoneBBox(x=100, y=200, width=20, height=10)})
    plan = adapt(_profile({"lip_full": bad}), _blueprint(_lip(target_zones=["lip_full"])))
    c = plan.components[0]
    assert "zone_mask_invalid" in c.notes and c.mask is None


def test_stage_ordering_and_fields_unchanged():
    lip = _lip()
    blush = _bc("powder_blush", "blush", color_hex="#E0708A", intensity=0.4)
    contour = _bc("contour_powder", "bronzer_contour", color_hex="#8A6048", intensity=0.3)
    plan = adapt(_profile(), _blueprint(lip, blush, contour))
    numbers = [c.stage_number for c in plan.components]
    assert numbers == sorted(numbers) and len(numbers) == 3
    assert [c.index for c in plan.components] == [0, 1, 2]
    by_pt = {c.component.product_type: c for c in plan.components}
    for src in (lip, blush, contour):
        out = by_pt[src.component.product_type]
        assert out.component.color_hex == src.component.color_hex
        assert out.component.finish == src.component.finish
        assert out.component.intensity == src.component.intensity
        assert out.component.color_lab == src.component.color_lab
        assert out.adjustments == []


@pytest.mark.parametrize("status", ["no_face_detected", "multiple_faces_detected"])
def test_profile_status_precedence(status):
    plan = adapt(_profile(status=status), _blueprint(status="no_face_detected"))
    assert plan.status == status and plan.components == [] and plan.reference is None


def test_blueprint_statuses():
    plan = adapt(_profile(), _blueprint(status="no_face_detected"))
    assert plan.status == "reference_unavailable" and plan.reference is None
    plan = adapt(_profile(), _blueprint())
    assert plan.status == "no_makeup_detected" and plan.components == [] and plan.reference is not None


def test_deterministic():
    bp = _blueprint(_lip(target_zones=["lip_full"]), _bc("powder_blush", "blush"))
    prof = _profile()
    a = adapt(prof, bp).model_dump(exclude={"timings_ms"})
    b = adapt(prof, bp).model_dump(exclude={"timings_ms"})
    assert a == b


def test_injectable_clock():
    ticks = iter([1.0, 1.5])
    plan = adapt(_profile(), _blueprint(), clock=lambda: next(ticks))
    assert plan.timings_ms["adapt"] == pytest.approx(500.0)


def test_no_forbidden_constructs_in_package():
    root = Path(__file__).resolve().parents[1] / "app" / "services" / "makeup_adaptation"
    for path in root.glob("*.py"):
        text = path.read_text()
        for token in ("model_construct", "model_copy", "import logging", "logger"):
            assert token not in text, f"{token} in {path.name}"
