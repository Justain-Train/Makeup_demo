import math

import pytest
from pydantic import ValidationError

from app.schemas.makeup_component import MakeupComponent
from app.services.application_zones.catalog import ZONE_SPEC_BY_ID
from app.services.makeup_vocabulary import (
    UnsupportedComponentVersion,
    check,
    hex_to_lab,
    is_supported,
    normalize_hex,
    parse_hex,
)

ZONE_IDS = list(ZONE_SPEC_BY_ID)


def make(**kw):
    base = {"category": "blush", "product_type": "cream_blush"}
    base.update(kw)
    return MakeupComponent(**base)


@pytest.mark.parametrize(
    "hex_,expected,tol",
    [
        ("#FFFFFF", (100, 0, 0), 0.05),
        ("#000000", (0, 0, 0), 0.05),
        ("#FF0000", (53.24, 80.09, 67.20), 0.05),
        ("#00FF00", (87.73, -86.18, 83.18), 0.05),
        ("#0000FF", (32.30, 79.19, -107.86), 0.05),
    ],
)
def test_hex_to_lab(hex_, expected, tol):
    got = hex_to_lab(hex_)
    assert all(abs(g - e) <= tol for g, e in zip(got, expected))


def test_hex_to_lab_gray():
    l_, a, b = hex_to_lab("#808080")
    assert abs(l_ - 53.59) <= 0.05
    assert abs(a) <= 0.02 and abs(b) <= 0.02


def test_hex_to_lab_black_exact():
    assert hex_to_lab("#000000") == (0, 0, 0)


def test_short_hex_expands():
    assert parse_hex("#abc") == (0xAA, 0xBB, 0xCC)
    assert normalize_hex("#abc") == normalize_hex("#AABBCC") == "#AABBCC"


def test_hex_uppercase_storage():
    assert make(color_hex="#ff8800").color_hex == "#FF8800"


@pytest.mark.parametrize(
    "bad",
    ["FF0000", "#FF0000FF", "#FFFF", "#FF00", " #FF0000", "#FF0000 ", "#FF 000", "#GG0000", "#ＦＦ0000",
     "#FF0000\n", "", "#", "#" + "F" * 100, 5, None, b"#FF0000", ["#FF0000"]],
)
def test_parse_hex_rejects(bad):
    with pytest.raises(ValueError):
        parse_hex(bad)  # type: ignore[arg-type]


def test_parse_hex_error_does_not_echo():
    with pytest.raises(ValueError) as exc:
        parse_hex("secretXYZ")
    assert "secretXYZ" not in str(exc.value)


@pytest.mark.parametrize("bad", ["FF0000", "#FF0000FF", " #FFF", "#GGG"])
def test_component_rejects_bad_hex(bad):
    with pytest.raises(ValidationError):
        make(color_hex=bad)


def test_color_lab_computed():
    c = make(color_hex="#FF0000")
    assert c.color_lab == hex_to_lab("#FF0000")


def test_alias_product_type_canonicalised():
    c = MakeupComponent(category="foundation", product_type="Tinted Moisturizer")
    assert c.product_type == "skin_tint"


def test_alias_placement_canonicalised():
    c = MakeupComponent(category="lip", product_type="lipstick", placement="Cupid's Bow")
    assert c.placement == "cupid_bow"


def test_category_normalized():
    assert MakeupComponent(category="Skin Prep", product_type="primer").category == "skin_prep"


def test_finish_and_direction_canonicalised():
    c = make(finish="Glowy", direction="Toward Temple")
    assert (c.finish, c.direction) == ("dewy", "outward")


def test_technique_canonicalised():
    c = MakeupComponent(category="eye", product_type="eyeshadow", technique="Smokey")
    assert c.technique == "smoky_blending"


@pytest.mark.parametrize(
    "kw",
    [
        {"category": "lip", "product_type": "cream_blush"},
        {"category": "eye", "product_type": "lipstick"},
        {"category": "nope", "product_type": "cream_blush"},
        {"category": 5, "product_type": "cream_blush"},
        {"product_type": "nope"},
        {"product_type": 5},
    ],
)
def test_category_and_product_type_rejected(kw):
    with pytest.raises(ValidationError):
        make(**kw)


def test_technique_category_mismatch():
    with pytest.raises(ValidationError):
        make(technique="winged")


def test_placement_category_mismatch():
    with pytest.raises(ValidationError):
        make(placement="lid")


def test_empty_categories_technique_allowed():
    assert make(technique="blending").technique == "blending"


def test_minimal_component():
    c = MakeupComponent(category="skin_prep", product_type="primer")
    assert c.color_hex is None and c.color_lab is None
    assert c.target_zones == []
    assert c.schema_version == "1.0"


def test_missing_required():
    with pytest.raises(ValidationError):
        MakeupComponent(category="blush")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        MakeupComponent(product_type="primer")  # type: ignore[call-arg]


@pytest.mark.parametrize("field", ["intensity", "confidence"])
@pytest.mark.parametrize("val", [0.0, 0.5, 1.0])
def test_unit_range_ok(field, val):
    assert getattr(make(**{field: val}), field) == val


@pytest.mark.parametrize("field", ["intensity", "confidence"])
@pytest.mark.parametrize("val", [-0.01, 1.01, math.nan, math.inf, -math.inf, "x"])
def test_unit_range_rejected(field, val):
    with pytest.raises(ValidationError):
        make(**{field: val})


@pytest.mark.parametrize("ut", ["warm", "cool", "neutral", "olive"])
def test_undertone_ok(ut):
    assert make(undertone=ut).undertone == ut


def test_undertone_bad():
    with pytest.raises(ValidationError):
        make(undertone="purple")


def test_shade_name_limits():
    assert make(shade_name="a" * 64).shade_name == "a" * 64
    with pytest.raises(ValidationError):
        make(shade_name="a" * 65)


def test_placement_note_limits():
    assert make(placement_note="a" * 256).placement_note == "a" * 256
    with pytest.raises(ValidationError):
        make(placement_note="a" * 257)


def test_special_effects_empty_zones():
    c = MakeupComponent(category="special_effects", product_type="glitter", target_zones=[])
    assert c.target_zones == []


def test_special_effects_real_zones():
    c = MakeupComponent(category="special_effects", product_type="freckles", target_zones=ZONE_IDS[:3])
    assert c.target_zones == ZONE_IDS[:3]


def test_unknown_zone_rejected():
    with pytest.raises(ValidationError):
        make(target_zones=["not_a_zone"])


def test_duplicate_zone_rejected():
    with pytest.raises(ValidationError):
        make(target_zones=[ZONE_IDS[0], ZONE_IDS[0]])


def test_overlong_zone_id_rejected():
    with pytest.raises(ValidationError):
        make(target_zones=["a" * 65])


def test_non_str_zone_rejected():
    with pytest.raises(ValidationError):
        make(target_zones=[5])


def test_zone_cap_rejected():
    with pytest.raises(ValidationError):
        make(target_zones=ZONE_IDS[:5] * 26)
    with pytest.raises(ValidationError):
        make(target_zones=[f"z{i}" for i in range(129)])


def test_caller_color_lab_ignored():
    c = make(color_hex="#FF0000", color_lab=(1.0, 2.0, 3.0))
    assert c.color_lab == hex_to_lab("#FF0000")


def test_caller_color_lab_ignored_without_color():
    assert make(color_lab=(1.0, 2.0, 3.0)).color_lab is None
    assert make(color_lab="garbage").color_lab is None


def test_extra_fields_ignored():
    c = make(future_field=1)
    assert not hasattr(c, "future_field")


@pytest.mark.parametrize("v", ["1.0", "1.9"])
def test_compat_pass(v):
    assert is_supported(v)
    check({"schema_version": v, "extra": 1})


@pytest.mark.parametrize(
    "v", ["2.0", "0.9", "abc", "1" * 17 + ".0", "１.０", "1.0\n", "1", 1.0, None, b"1.0"]
)
def test_compat_fail(v):
    with pytest.raises(UnsupportedComponentVersion):
        check({"schema_version": v})


def test_compat_missing_version():
    with pytest.raises(UnsupportedComponentVersion):
        check({})


@pytest.mark.parametrize("m", [None, "1.0", 5, ["1.0"]])
def test_compat_non_mapping(m):
    with pytest.raises(UnsupportedComponentVersion):
        check(m)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kw",
    [
        {"shade_name": "SECRETMARKER" + "a" * 70},
        {"color_hex": "#GGGGGG"},
        {"category": "SECRETMARKERcat"},
    ],
)
def test_errors_do_not_echo_input(kw):
    with pytest.raises(ValidationError) as exc:
        make(**kw)
    marker = "SECRETMARKER"
    for text in (str(exc.value), repr(exc.value), repr(exc.value.errors(include_input=False))):
        assert marker not in text
        assert "#GGGGGG" not in text


def test_component_rejects_unsupported_version():
    with pytest.raises(ValidationError):
        make(schema_version="2.0")
    assert make(schema_version="1.5").schema_version == "1.5"
