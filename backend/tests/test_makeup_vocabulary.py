import dataclasses
import types

import pytest

from app.services.application_zones.catalog import STAGES, ZONE_SPEC_BY_ID
from app.services.makeup_vocabulary import (
    CATEGORIES,
    KINDS,
    REGISTRY,
    VOCABULARY_VERSION,
    InvalidVocabularyVersion,
    UnknownTermError,
    normalize,
    parse_vocabulary_version,
    product_types_for_category,
    require_term,
    resolve_term,
    stage_layer_for,
    validate_registry,
)

PTS = REGISTRY["product_types"]
STAGE_BY_KEY = {s.key: s for s in STAGES}


def test_twelve_categories():
    assert len(CATEGORIES) == 12


def test_five_kinds():
    assert set(REGISTRY) == set(KINDS)


@pytest.mark.parametrize("key", [k for k, t in PTS.items() if t.category != "special_effects"])
def test_stage_and_layer_exist(key):
    pt = PTS[key]
    assert pt.stage_key in STAGE_BY_KEY
    assert pt.layer_key in STAGE_BY_KEY[pt.stage_key].layers


def test_special_effects_all_none():
    se = [t for t in PTS.values() if t.category == "special_effects"]
    assert {t.key for t in se} == {"face_gems", "glitter", "face_paint", "freckles"}
    assert all(t.stage_key is None and t.layer_key is None for t in se)


def test_only_special_effects_none():
    assert all(t.stage_key is not None for t in PTS.values() if t.category != "special_effects")


@pytest.mark.parametrize("cat", CATEGORIES)
def test_category_has_product_type(cat):
    assert product_types_for_category(cat)


def test_validate_registry_passes():
    validate_registry()


def test_validate_registry_duplicate_alias():
    term = REGISTRY["finishes"]["matte"]
    bad = dict(REGISTRY)
    bad["finishes"] = {
        "matte": term,
        "satin": dataclasses.replace(REGISTRY["finishes"]["satin"], aliases=("Matte",)),
    }
    with pytest.raises(ValueError):
        validate_registry(bad)


def test_validate_registry_canonical_alias_collision():
    bad = dict(REGISTRY)
    bad["directions"] = {
        "upward": dataclasses.replace(REGISTRY["directions"]["upward"], aliases=("inward",)),
        "inward": REGISTRY["directions"]["inward"],
    }
    with pytest.raises(ValueError):
        validate_registry(bad)


def test_validate_registry_unknown_category():
    bad = dict(REGISTRY)
    bad["product_types"] = {"primer": dataclasses.replace(PTS["primer"], category="nope")}
    with pytest.raises(ValueError):
        validate_registry(bad)


def test_validate_registry_none_stage_outside_special_effects():
    bad = dict(REGISTRY)
    bad["product_types"] = {"primer": dataclasses.replace(PTS["primer"], stage_key=None, layer_key=None)}
    with pytest.raises(ValueError):
        validate_registry(bad)


def test_validate_registry_unknown_placement_category():
    bad = dict(REGISTRY)
    bad["placements"] = {"lid": dataclasses.replace(REGISTRY["placements"]["lid"], categories=("nope",))}
    with pytest.raises(ValueError):
        validate_registry(bad)


@pytest.mark.parametrize("text", ["Eye Shadow", "eye-shadow", "EYE_SHADOW", "  eye   shadow  ", "eyeshadow"])
def test_alias_variants(text):
    assert resolve_term("product_types", text) == "eyeshadow"


@pytest.mark.parametrize("text,key", [
    ("Center of Forehead", "forehead"), ("temple", "temples"), ("Jaw Line", "jawline"),
])
def test_new_placement_aliases(text, key):
    assert resolve_term("placements", text) == key


def test_canonical_resolves_to_itself():
    assert resolve_term("finishes", "matte") == "matte"


def test_unknown_returns_none():
    assert resolve_term("finishes", "zzz") is None


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        resolve_term("nope", "matte")


def test_require_term_raises():
    with pytest.raises(UnknownTermError):
        require_term("finishes", "zzz")


def test_require_term_ok():
    assert require_term("directions", "Toward Temple") == "outward"


def test_overlong_rejected():
    with pytest.raises(ValueError):
        normalize("a" * 65)


def test_non_str_rejected():
    with pytest.raises(ValueError):
        resolve_term("finishes", 5)  # type: ignore[arg-type]


def test_exactly_64_chars_accepted():
    assert normalize("a" * 64) == "a" * 64


@pytest.mark.parametrize("text", ["ﷺ" * 4, "㌀" * 22])
def test_nfkc_expansion_over_limit_rejected(text):
    assert len(text) <= 64
    with pytest.raises(ValueError):
        normalize(text)


@pytest.mark.parametrize("kind", [5, None, ["finishes"], {"a": 1}, b"finishes"])
def test_non_str_or_unhashable_kind_raises(kind):
    with pytest.raises(ValueError):
        resolve_term(kind, "matte")  # type: ignore[arg-type]


def test_unknown_kind_message_does_not_echo():
    with pytest.raises(ValueError) as exc:
        resolve_term("secret_kind_xyz", "matte")
    assert "secret_kind_xyz" not in str(exc.value)


def test_validate_registry_still_raises_on_bad_data():
    bad = dict(REGISTRY)
    bad["product_types"] = {"primer": dataclasses.replace(PTS["primer"], category="nope")}
    with pytest.raises(ValueError):
        validate_registry(bad)


def test_stage_layer_for():
    assert stage_layer_for("contour") == ("bronzer_contour", "contour")
    assert stage_layer_for("lip gloss") == ("lips", "gloss")


def test_no_crease_placement():
    assert "crease" not in REGISTRY["placements"]
    assert resolve_term("placements", "crease") is None


def test_registry_containers_immutable():
    assert isinstance(REGISTRY, types.MappingProxyType)
    with pytest.raises(TypeError):
        REGISTRY["x"] = {}  # type: ignore[index]
    for kind in KINDS:
        assert isinstance(REGISTRY[kind], types.MappingProxyType)
        with pytest.raises(TypeError):
            REGISTRY[kind]["x"] = None  # type: ignore[index]


def test_term_fields_immutable_tuples():
    for kind in KINDS:
        for term in REGISTRY[kind].values():
            assert isinstance(term.aliases, tuple)
            with pytest.raises(dataclasses.FrozenInstanceError):
                term.key = "x"  # type: ignore[misc]


def test_no_term_equals_zone_id():
    for kind in KINDS:
        for term in REGISTRY[kind].values():
            names = term.aliases if kind == "placements" else (term.key, *term.aliases)
            for name in names:
                assert normalize(name) not in ZONE_SPEC_BY_ID


@pytest.mark.parametrize(
    "text,expected",
    [("Cupid's Bow", "cupid_bow"), ("Cupid’s Bow", "cupid_bow"), ("cupid bow", "cupid_bow"),
     ("nose-bridge", "nose_bridge")],
)
def test_placement_names_matching_zones(text, expected):
    assert resolve_term("placements", text) == expected


@pytest.mark.parametrize("v", ["1.0", "0.1", "12.34", "9999.9999"])
def test_good_versions(v):
    assert parse_vocabulary_version(v) == tuple(int(p) for p in v.split("."))


@pytest.mark.parametrize("v", ["1", "1.0.0", "a.b", "1.", "-1.0", "12345.0", "1.0\n", "１.０", "1" * 17 + ".0", ""])
def test_bad_versions(v):
    with pytest.raises(InvalidVocabularyVersion):
        parse_vocabulary_version(v)


def test_non_str_version():
    with pytest.raises(InvalidVocabularyVersion):
        parse_vocabulary_version(1.0)  # type: ignore[arg-type]


def test_current_version_parses():
    assert parse_vocabulary_version(VOCABULARY_VERSION) == (1, 1)
