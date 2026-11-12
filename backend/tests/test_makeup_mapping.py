import pytest

from app.schemas.makeup_component import MakeupComponent
from app.services.application_zones import catalog
from app.services.makeup_mapping import (
    CatalogDriftError,
    PLACEMENT_ZONES,
    ZoneOutsideLayerError,
    map_component,
    map_zones,
)
from app.services.makeup_vocabulary import REGISTRY


def _layer(stage_key: str, layer_key: str) -> list[str]:
    return next(s for s in catalog.STAGES if s.key == stage_key).layers[layer_key]


def test_lifted_cheek_cream_blush():
    m = map_zones("cream_blush", "lifted_cheek")
    assert m.zone_ids == ("cheek_upper_r", "cheek_upper_l")
    assert m.source == "placement"
    assert m.reason == "placement_match"


def test_no_placement_full_layer():
    m = map_zones("cream_blush")
    assert m.zone_ids == tuple(_layer("blush", "blush"))
    assert m.source == "layer"
    assert m.reason == "layer_zones"


def test_empty_intersection_is_none():
    m = map_zones("lipstick", "outer_lip_edge")
    assert m.zone_ids == ()
    assert m.source == "none"
    assert m.reason == "placement_not_in_layer"


def test_bronzer_full_layer():
    m = map_zones("bronzer")
    assert m.zone_ids == (
        "temple_r", "temple_l", "forehead_perimeter", "cheek_outer_r", "cheek_outer_l",
        "jawline_outer_r", "jawline_outer_l")
    assert m.source == "layer"


@pytest.mark.parametrize("pt,pl,expected", [
    ("powder_highlighter", "forehead", ("forehead_center",)),
    ("bronzer", "temples", ("temple_r", "temple_l")),
    ("contour_powder", "jawline", ("jawline",)),
    ("cream_blush", "outer_cheek", ("cheek_outer_r", "cheek_outer_l")),
    ("powder_highlighter", "inner_corner", ("eye_corner_inner_r", "eye_corner_inner_l")),
    ("powder_highlighter", "cupid_bow", ("cupid_bow",)),
])
def test_new_placement_mappings(pt, pl, expected):
    assert map_zones(pt, pl).zone_ids == expected


def test_explicit_in_layer_ordered_deduped():
    layer = _layer("blush", "blush")
    picks = [layer[2], layer[0], layer[2]]
    m = map_zones("cream_blush", None, picks)
    assert m.zone_ids == (layer[0], layer[2])
    assert m.source == "explicit"
    assert m.reason == "explicit_zones"


def test_explicit_out_of_layer_raises_without_echo():
    with pytest.raises(ZoneOutsideLayerError) as e:
        map_zones("cream_blush", None, ["lip_full"])
    assert "lip_full" not in str(e.value)
    with pytest.raises(ZoneOutsideLayerError) as e:
        map_zones("cream_blush", None, ["not_a_zone_xyz"])
    assert "not_a_zone_xyz" not in str(e.value)


def test_special_effects():
    m = map_zones("glitter", "lid")
    assert m.zone_ids == () and m.source == "none"
    assert m.reason == "special_effect_note_only"
    ids = list(catalog.ZONE_SPEC_BY_ID)
    m = map_zones("glitter", None, [ids[3], ids[1], ids[3]])
    assert m.zone_ids == (ids[1], ids[3])
    assert m.source == "explicit"
    assert m.stage_key is None and m.layer_key is None
    with pytest.raises(ZoneOutsideLayerError) as e:
        map_zones("glitter", None, ["bogus_zone_abc"])
    assert "bogus_zone_abc" not in str(e.value)


def test_monkeypatched_stages_change_result(monkeypatch):
    base = map_zones("cream_blush")
    patched = [
        catalog.StageSpec(s.number, s.key, s.name, {k: v[:1] for k, v in s.layers.items()})
        for s in catalog.STAGES
    ]
    monkeypatch.setattr(catalog, "STAGES", patched)
    new = map_zones("cream_blush")
    assert new.zone_ids != base.zone_ids
    assert len(new.zone_ids) == 1


def test_table_drift():
    assert set(PLACEMENT_ZONES) == set(REGISTRY["placements"])
    for zones in PLACEMENT_ZONES.values():
        for z in zones:
            assert z in catalog.ZONE_SPEC_BY_ID
    with pytest.raises(TypeError):
        PLACEMENT_ZONES["x"] = ()  # type: ignore[index]


def _compatible_pairs() -> list[tuple[str, str]]:
    return [
        (pk, ptk)
        for pk, pl in REGISTRY["placements"].items()
        for ptk, pt in REGISTRY["product_types"].items()
        if pt.stage_key is not None and (not pl.categories or pt.category in pl.categories)
    ]


@pytest.mark.parametrize("pkey,ptkey", _compatible_pairs())
def test_sweep(pkey, ptkey):
    pt = REGISTRY["product_types"][ptkey]
    m = map_zones(ptkey, pkey)
    assert len(set(m.zone_ids)) == len(m.zone_ids)
    for z in m.zone_ids:
        assert z in catalog.ZONE_SPEC_BY_ID
    assert set(m.zone_ids) <= set(_layer(pt.stage_key, pt.layer_key))


def _expected_none() -> set[tuple[str, str, str]]:
    p = "placement_not_in_layer"
    out = {(pl, "bronzer", p) for pl in ("lifted_cheek", "hollow_of_cheek", "nose_bridge")}
    out |= {(pl, t, p) for pl in ("temples", "hairline", "outer_cheek")
            for t in ("contour_powder", "contour_cream")}
    out |= {("brow_bone", t, p) for t in ("brow_pencil", "brow_powder", "brow_gel")}
    out |= {("under_eye", t, p) for t in
            ("setting_spray", "eyeshadow", "eyeliner", "mascara", "eye_primer")}
    out |= {("full_lip", "lip_liner", p)}
    out |= {("cupid_bow", "lip_gloss", p)}
    out |= {("outer_lip_edge", t, p) for t in ("lipstick", "lip_tint", "lip_gloss")}
    return out


def test_none_combos_snapshot():
    # A change to this set means the catalog or the placement table changed and needs review.
    actual = set()
    for pk, ptk in _compatible_pairs():
        m = map_zones(ptk, pk)
        if m.source == "none":
            actual.add((pk, ptk, m.reason))
    assert actual == _expected_none()
    assert len(actual) == 22


def _without_stage(monkeypatch, drop_layer: bool):
    stages = []
    for s in catalog.STAGES:
        if s.key == "blush":
            if not drop_layer:
                continue
            s = catalog.StageSpec(s.number, s.key, s.name,
                                  {k: v for k, v in s.layers.items() if k != "blush"})
        stages.append(s)
    monkeypatch.setattr(catalog, "STAGES", stages)


@pytest.mark.parametrize("drop_layer", [False, True])
def test_catalog_drift_fails_closed(monkeypatch, drop_layer):
    _without_stage(monkeypatch, drop_layer)
    with pytest.raises(CatalogDriftError):
        map_zones("cream_blush")
    with pytest.raises(CatalogDriftError):
        map_zones("cream_blush", "lifted_cheek")


@pytest.mark.parametrize("bad", [None, (z for z in ["cheek_r_secret"]), {"cheek_r_secret"}, "cheek_r_secret"])
def test_target_zones_type_rejected(bad):
    with pytest.raises(ValueError) as e:
        map_zones("cream_blush", None, bad)
    assert "cheek_r_secret" not in str(e.value)


@pytest.mark.parametrize("pt,pl", [
    ("brow_pencil", "brow_bone"),
    ("eyeshadow", "under_eye"),
    ("lip_liner", "full_lip"),
    ("lipstick", "outer_lip_edge"),
])
def test_known_empty_combos(pt, pl):
    m = map_zones(pt, pl)
    assert m.zone_ids == ()
    assert m.source == "none"


def test_map_component():
    c = MakeupComponent(category="blush", product_type="cream_blush", placement="lifted_cheek")
    assert map_component(c).zone_ids == ("cheek_upper_r", "cheek_upper_l")


def test_caps():
    with pytest.raises(ValueError):
        map_zones("cream_blush", None, ["cheek_r"] * 129)
    with pytest.raises(ValueError):
        map_zones("cream_blush", None, ["x" * 65])
    with pytest.raises(ValueError):
        map_zones("cream_blush", None, [1])  # type: ignore[list-item]
