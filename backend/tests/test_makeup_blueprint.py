import pytest
from pydantic import ValidationError

from app.schemas.makeup_blueprint import (
    BlueprintComponent,
    BlueprintQuality,
    BlueprintStyle,
    BlueprintWarning,
    MakeupBlueprint,
    ReferenceProviderInfo,
)
from app.services.application_zones.catalog import ZONE_SPEC_BY_ID
from app.services.makeup_blueprint import (
    SUPPORTED_BLUEPRINT_VERSIONS,
    UnsupportedBlueprintVersion,
    check,
    is_supported,
)

ZONE_IDS = list(ZONE_SPEC_BY_ID)
SENTINEL = "SENTINEL_SECRET_42"
IMAGE = {"width": 100, "height": 100}


def comp(**over):
    base = {"category": "blush", "product_type": "cream_blush"}
    base.update(over)
    return base


def bcomp(component=None, sources=None):
    component = component if component is not None else comp()
    if sources is None:
        sources = {"category": "cv", "product_type": "cv"}
    return {"component": component, "sources": sources}


def provider(kind="cv", **over):
    d = {"id": f"p-{kind}", "kind": kind, "algorithm_version": "1", "enabled": True}
    d.update(over)
    return d


def bp(**over):
    d = {"status": "ok", "image": IMAGE}
    d.update(over)
    return d


def test_full_roundtrip():
    data = bp(
        message="m",
        components=[
            bcomp(
                comp(color_hex="#ff0000", intensity=0.5, finish="Glowy", confidence=0.7, target_zones=ZONE_IDS[:2]),
                {
                    "category": "cv",
                    "product_type": "vlm",
                    "color_hex": "cv",
                    "intensity": "cv",
                    "finish": "vlm",
                    "target_zones": "cv",
                },
            )
        ],
        style={"labels": ["Glass Skin", "soft"], "confidence": 0.8},
        quality={"warnings": [{"code": "low_confidence", "message": "x"}], "overall_confidence": 0.6},
        providers=[provider("cv"), provider("vlm", model="m")],
        timings_ms={"cv": 1.5},
    )
    m = MakeupBlueprint.model_validate(data)
    dumped = m.model_dump(mode="json")
    again = MakeupBlueprint.model_validate(dumped)
    assert again == m
    assert again.style.labels == ["glass_skin", "soft"]


@pytest.mark.parametrize("v", ["1.0", "1.9"])
def test_version_accepted(v):
    assert is_supported(v)
    assert MakeupBlueprint.model_validate(bp(schema_version=v)).schema_version == v


@pytest.mark.parametrize("v", ["2.0", "0.9", "abc", "1", "1.0.0", "", "1." + "0" * 30])
def test_version_rejected(v):
    with pytest.raises(ValueError):
        MakeupBlueprint.model_validate(bp(schema_version=v))


def test_supported_mapping_is_readonly():
    with pytest.raises(TypeError):
        SUPPORTED_BLUEPRINT_VERSIONS[2] = 0  # type: ignore[index]


def test_check_ok_and_ignores_unknown():
    check({"schema_version": "1.3", "extra": 1})


@pytest.mark.parametrize("payload", [{"schema_version": "2.0"}, {"schema_version": 1}, {}, {"schema_version": "x"}, []])
def test_check_raises(payload):
    with pytest.raises(UnsupportedBlueprintVersion):
        check(payload)


def test_check_does_not_echo_input():
    for v in (SENTINEL, "9.9", SENTINEL * 20):
        with pytest.raises(UnsupportedBlueprintVersion) as ei:
            check({"schema_version": v})
        assert SENTINEL not in str(ei.value)
        assert "9.9" not in str(ei.value)


@pytest.mark.parametrize("status", ["no_face_detected", "multiple_faces_detected"])
@pytest.mark.parametrize(
    "extra",
    [
        {"components": [bcomp()]},
        {"style": {"labels": ["soft"]}},
        {"providers": [provider()]},
    ],
)
def test_non_ok_status_rejects_content(status, extra):
    with pytest.raises(ValidationError):
        MakeupBlueprint.model_validate(bp(status=status, **extra))


def test_non_ok_status_empty_valid():
    assert MakeupBlueprint.model_validate(bp(status="no_face_detected")).components == []


def test_ok_with_empty_components_valid():
    assert MakeupBlueprint.model_validate(bp()).components == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("color_hex", "#FF0000"),
        ("intensity", 0.5),
        ("target_zones", ZONE_IDS[:1]),
        ("undertone", "warm"),
    ],
)
def test_vlm_source_rejected_for_cv_fields(field, value):
    sources = {"category": "cv", "product_type": "cv", field: "vlm"}
    with pytest.raises(ValidationError):
        BlueprintComponent.model_validate(bcomp(comp(**{field: value}), sources))
    sources[field] = "cv"
    BlueprintComponent.model_validate(bcomp(comp(**{field: value}), sources))


def test_missing_source_rejected():
    with pytest.raises(ValidationError):
        BlueprintComponent.model_validate(bcomp(comp(color_hex="#FF0000"), {"category": "cv", "product_type": "cv"}))


def test_missing_category_source_rejected():
    with pytest.raises(ValidationError):
        BlueprintComponent.model_validate(bcomp(sources={"product_type": "cv"}))


def test_source_for_empty_field_rejected():
    with pytest.raises(ValidationError):
        BlueprintComponent.model_validate(bcomp(sources={"category": "cv", "product_type": "cv", "color_hex": "cv"}))


def test_source_for_empty_zones_rejected():
    with pytest.raises(ValidationError):
        BlueprintComponent.model_validate(bcomp(sources={"category": "cv", "product_type": "cv", "target_zones": "cv"}))


def test_unknown_source_key_rejected():
    with pytest.raises(ValidationError):
        BlueprintComponent.model_validate(bcomp(sources={"category": "cv", "product_type": "cv", "confidence": "cv"}))


def test_invalid_source_value_rejected():
    with pytest.raises(ValidationError):
        BlueprintComponent.model_validate(bcomp(sources={"category": "x", "product_type": "cv"}))


def test_sources_cap():
    keys = [f"k{i}" for i in range(17)]
    with pytest.raises(ValidationError):
        BlueprintComponent.model_validate(bcomp(sources={k: "cv" for k in keys}))
    # exact cap passes the length check (fails only on unknown keys, not on size)
    try:
        BlueprintComponent.model_validate(bcomp(sources={k: "cv" for k in keys[:16]}))
    except ValidationError as exc:
        assert all("at most" not in e["msg"] for e in exc.errors(include_input=False))


def test_components_cap():
    assert len(MakeupBlueprint.model_validate(bp(components=[bcomp()] * 32)).components) == 32
    with pytest.raises(ValidationError):
        MakeupBlueprint.model_validate(bp(components=[bcomp()] * 33))


def test_labels_cap():
    labels = sorted(
        ["natural", "glam", "editorial", "bold", "soft", "dramatic", "minimal", "glass_skin", "smoky", "vintage"]
    )
    assert len(BlueprintStyle.model_validate({"labels": labels[:8]}).labels) == 8
    with pytest.raises(ValidationError):
        BlueprintStyle.model_validate({"labels": labels[:9]})


def test_warnings_cap():
    w = {"code": "vlm_failed"}
    assert len(BlueprintQuality.model_validate({"warnings": [w] * 32}).warnings) == 32
    with pytest.raises(ValidationError):
        BlueprintQuality.model_validate({"warnings": [w] * 33})


def test_providers_cap():
    # kinds are unique so only 2 are possible; check the cap on the list itself
    with pytest.raises(ValidationError):
        MakeupBlueprint.model_validate(bp(providers=[provider("cv")] * 9))


def test_message_cap():
    assert MakeupBlueprint.model_validate(bp(message="a" * 512)).message == "a" * 512
    with pytest.raises(ValidationError):
        MakeupBlueprint.model_validate(bp(message="a" * 513))


def test_warning_message_cap():
    assert BlueprintWarning(code="vlm_failed", message="a" * 256)
    with pytest.raises(ValidationError):
        BlueprintWarning(code="vlm_failed", message="a" * 257)


def test_provider_text_cap():
    assert ReferenceProviderInfo.model_validate(provider(id="a" * 64))
    with pytest.raises(ValidationError):
        ReferenceProviderInfo.model_validate(provider(id="a" * 65))


@pytest.mark.parametrize("bad", ["Bad Id!", "a" * 65, ""])
def test_provider_id_rejected(bad):
    with pytest.raises(ValidationError):
        ReferenceProviderInfo.model_validate(provider(id=bad))


def test_provider_model_pattern():
    assert ReferenceProviderInfo.model_validate(provider("vlm", model="claude-opus-5-5")).model == "claude-opus-5-5"
    with pytest.raises(ValidationError):
        ReferenceProviderInfo.model_validate(provider("vlm", model="claude opus"))


@pytest.mark.parametrize(
    "key,value",
    [
        ("cv_ms", float("nan")),
        ("cv_ms", float("inf")),
        ("cv_ms", -1.0),
        ("cv_ms", 1e8),
        ("CV", 1.0),
        ("cv ms", 1.0),
        ("a" * 65, 1.0),
    ],
)
def test_timings_rejected(key, value):
    with pytest.raises(ValidationError):
        MakeupBlueprint.model_validate(bp(timings_ms={key: value}))


def test_timings_valid():
    assert MakeupBlueprint.model_validate(bp(timings_ms={"cv_ms": 12.5})).timings_ms == {"cv_ms": 12.5}


def test_timings_cap():
    assert MakeupBlueprint.model_validate(bp(timings_ms={str(i): 1.0 for i in range(32)}))
    with pytest.raises(ValidationError):
        MakeupBlueprint.model_validate(bp(timings_ms={str(i): 1.0 for i in range(33)}))


def test_duplicate_provider_kinds_rejected():
    with pytest.raises(ValidationError):
        MakeupBlueprint.model_validate(bp(providers=[provider("cv", id="a"), provider("cv", id="b")]))


def test_unknown_warning_code_rejected():
    with pytest.raises(ValidationError):
        BlueprintWarning(code="nope")


def test_unknown_style_label_rejected():
    with pytest.raises(ValidationError):
        BlueprintStyle.model_validate({"labels": ["grunge"]})


def test_style_label_normalization():
    s = BlueprintStyle.model_validate({"labels": ["Glass Skin", "SMOKY", " glass-skin2"[:0] + "Soft "]})
    assert s.labels == ["glass_skin", "smoky", "soft"]


def test_style_hyphen_normalizes():
    assert BlueprintStyle.model_validate({"labels": ["Glass-Skin"]}).labels == ["glass_skin"]


def test_duplicate_labels_rejected():
    with pytest.raises(ValidationError):
        BlueprintStyle.model_validate({"labels": ["Glass Skin", "glass_skin"]})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 1.5, -0.1])
def test_style_confidence_rejected(bad):
    with pytest.raises(ValidationError):
        BlueprintStyle.model_validate({"labels": [], "confidence": bad})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_overall_confidence_rejected(bad):
    with pytest.raises(ValidationError):
        BlueprintQuality.model_validate({"overall_confidence": bad})


def test_style_source_only_vlm():
    assert BlueprintStyle().source == "vlm"
    with pytest.raises(ValidationError):
        BlueprintStyle.model_validate({"source": "cv"})


def test_unknown_top_level_field_ignored():
    m = MakeupBlueprint.model_validate(bp(future_field=1))
    assert not hasattr(m, "future_field")


def test_invalid_nested_component_rejected():
    with pytest.raises(ValidationError):
        MakeupBlueprint.model_validate(
            bp(components=[bcomp(comp(product_type="lipstick"), {"category": "cv", "product_type": "cv"})])
        )


def test_nested_color_lab_derived():
    m = MakeupBlueprint.model_validate(
        bp(
            components=[
                bcomp(
                    comp(color_hex="#ff0000", color_lab=[1, 2, 3]),
                    {"category": "cv", "product_type": "cv", "color_hex": "cv"},
                )
            ]
        )
    )
    lab = m.components[0].component.color_lab
    assert lab is not None and tuple(lab) != (1, 2, 3)


def test_sentinel_not_in_error_text():
    with pytest.raises(ValidationError) as ei:
        MakeupBlueprint.model_validate(
            bp(
                message=SENTINEL * 100,
                schema_version=SENTINEL,
                style={"labels": [SENTINEL]},
                components=[bcomp(comp(category=SENTINEL), {SENTINEL: "cv"})],
            )
        )
    assert SENTINEL not in str(ei.value)
    assert SENTINEL not in str(ei.value.errors(include_input=False))
