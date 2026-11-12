"""Tests for the makeup VLM provider layer. Never touches the network."""

import base64
import builtins
import logging
from pathlib import Path
from types import SimpleNamespace

import anthropic
import cv2
import httpx2
import numpy as np
import pytest

from app.config import Settings
from app.schemas.makeup_blueprint import BlueprintComponent
from app.schemas.makeup_component import MakeupComponent
from app.services.makeup_blueprint import vlm as vlm_mod
from app.services.makeup_blueprint.claude_vlm import ClaudeVlmProvider
from app.services.makeup_blueprint.vlm import (
    StubVlmProvider,
    VlmAnalysis,
    VlmAnnotationResult,
    attach_annotations,
    build_vlm_provider,
)
from app.services.makeup_blueprint.vlm_prompt import (
    VlmAnnotation,
    VlmProposal,
    VlmResponse,
    VlmResponseRaw,
    encode_image,
)

CANARY = "CANARY-SECRET-xyz123"
APP_DIR = Path(__file__).resolve().parent.parent / "app" / "services" / "makeup_blueprint"


def _img(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _cv(product_type: str = "lipstick", category: str = "lip", **extra: str) -> BlueprintComponent:
    comp = MakeupComponent.model_validate(
        {"category": category, "product_type": product_type, "color_hex": "#aa3344", **extra}
    )
    sources = {"category": "cv", "product_type": "cv", "color_hex": "cv"}
    sources.update({k: "cv" for k in extra})
    return BlueprintComponent.model_validate({"component": comp, "sources": sources})


class FakeMessages:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.kwargs = result, exc, None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if self.exc is not None:
            raise self.exc
        return self.result


class FakeClient:
    def __init__(self, result=None, exc=None):
        self.messages = FakeMessages(result, exc)
        self.beta = SimpleNamespace(messages=self.messages)


def _ok(resp: VlmResponse, stop: str = "end_turn"):
    return SimpleNamespace(stop_reason=stop, parsed_output=resp)


def _provider(client) -> ClaudeVlmProvider:
    return ClaudeVlmProvider(client, "claude-opus-5-5", "medium", 16000)


def _req() -> httpx2.Request:
    return httpx2.Request("POST", "https://example.invalid/v1/messages")


def _status_exc(cls, status: int):
    resp = httpx2.Response(status, request=_req())
    return cls(CANARY, response=resp, body=None)


def _codes(a: VlmAnalysis) -> list[str]:
    return [w.code for w in a.warnings]


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.makeup_vlm_enabled is False
    assert s.makeup_vlm_model == "claude-opus-5-5"
    assert not any("api_key" in name for name in Settings.model_fields)


def test_disabled_returns_stub_without_anthropic(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("anthropic must not be constructed")

    monkeypatch.setattr(anthropic, "Anthropic", boom)
    real_import = builtins.__import__

    def guard(name, *a, **k):
        if name.endswith("claude_vlm"):
            raise AssertionError("claude_vlm must not be imported")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guard)
    p = build_vlm_provider(Settings(_env_file=None, makeup_vlm_enabled=False))
    assert isinstance(p, StubVlmProvider)


def test_enabled_builds_with_injected_client():
    p = build_vlm_provider(Settings(_env_file=None, makeup_vlm_enabled=True), client=FakeClient())
    assert isinstance(p, ClaudeVlmProvider)
    assert p.info().model == "claude-opus-5-5" and p.info().enabled


def test_stub_deterministic():
    a = StubVlmProvider().analyze(_img(), [])
    b = StubVlmProvider().analyze(_img(8, 8), [_cv()])
    assert a == b
    assert _codes(a) == ["vlm_disabled"]
    assert a.annotations == [] and a.style is None
    assert a.info.id == "vlm_stub" and a.info.enabled is False and a.info.model is None


def test_call_kwargs_and_messages():
    client = FakeClient(_ok(VlmResponse()))
    img = _img()
    _provider(client).analyze(img, [_cv(), _cv("eyeliner", "eye")])
    kw = client.messages.kwargs
    assert set(kw) == {
        "model", "max_tokens", "thinking", "output_config", "output_format",
        "betas", "fallbacks", "system", "messages",
    }
    assert kw["model"] == "claude-opus-5-5" and kw["max_tokens"] == 16000
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"] == {"effort": "medium"}
    assert kw["output_format"] is VlmResponseRaw
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["fallbacks"] == "default"
    assert "temperature" not in kw
    content = kw["messages"][0]["content"]
    assert content[0]["type"] == "image"
    raw = base64.b64decode(content[0]["source"]["data"])
    assert raw[:3] == b"\xff\xd8\xff"
    assert raw != img.tobytes()
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert "0: lipstick/lip" in content[1]["text"]
    assert "1: eyeliner/eye" in content[1]["text"]


def test_valid_response():
    resp = VlmResponse(
        annotations=[VlmAnnotation(index=0, finish="matte")],
        proposals=[VlmProposal(category="eyebrows", product_type="brow_pencil", technique="defining")],
        style_labels=["natural", "soft"],
        style_confidence=0.8,
    )
    a = _provider(FakeClient(_ok(resp))).analyze(_img(), [_cv()])
    assert a.warnings == []
    assert a.annotations == [VlmAnnotationResult(index=0, finish="matte")]
    assert a.style is not None and a.style.labels == ["natural", "soft"] and a.style.confidence == 0.8
    (p,) = a.proposals
    assert isinstance(p, BlueprintComponent)
    assert p.component.target_zones == [] and p.component.color_hex is None
    assert set(p.sources.values()) == {"vlm"}


def test_duplicate_cv_product_type_proposal_dropped():
    resp = VlmResponse(
        proposals=[
            VlmProposal(category="eye", product_type="eyeliner"),
            VlmProposal(category="eye", product_type="mascara"),
            VlmProposal(category="eye", product_type="mascara"),
        ]
    )
    a = _provider(FakeClient(_ok(resp))).analyze(_img(), [_cv("eyeliner", "eye")])
    assert [p.component.product_type for p in a.proposals] == ["mascara"]


def test_out_of_vocab_proposal_rejected_by_schema():
    with pytest.raises(Exception):
        VlmProposal(category="lip", product_type="lipstick")
    with pytest.raises(Exception):
        VlmResponse(proposals=[VlmProposal(category="eye", product_type="mascara")] * 5)


def test_refusal_and_failures():
    assert _codes(_provider(FakeClient(_ok(VlmResponse(), "refusal"))).analyze(_img(), [])) == ["vlm_refused"]
    assert _codes(_provider(FakeClient(_ok(VlmResponse(), "max_tokens"))).analyze(_img(), [])) == ["vlm_failed"]
    none = SimpleNamespace(stop_reason="end_turn", parsed_output=None)
    assert _codes(_provider(FakeClient(none)).analyze(_img(), [])) == ["vlm_failed"]
    wrong = SimpleNamespace(stop_reason="end_turn", parsed_output={"annotations": []})
    assert _codes(_provider(FakeClient(wrong)).analyze(_img(), [])) == ["vlm_failed"]
    # category-invalid semantic combo fails re-validation of the proposal
    bad = VlmResponse(
        proposals=[VlmProposal(category="eyebrows", product_type="brow_gel", technique="winged")]
    )
    assert _codes(_provider(FakeClient(_ok(bad))).analyze(_img(), [])) == ["vlm_failed"]
    # invalid image
    bad_img = np.zeros((4, 4), dtype=np.uint8)
    assert _codes(_provider(FakeClient(_ok(VlmResponse()))).analyze(bad_img, [])) == ["vlm_failed"]


@pytest.mark.parametrize(
    "exc",
    [
        _status_exc(anthropic.AuthenticationError, 401),
        _status_exc(anthropic.PermissionDeniedError, 403),
        _status_exc(anthropic.NotFoundError, 404),
        _status_exc(anthropic.RateLimitError, 429),
        _status_exc(anthropic.InternalServerError, 500),
        _status_exc(anthropic.BadRequestError, 400),
        anthropic.APITimeoutError(_req()),
        anthropic.APIConnectionError(message=CANARY, request=_req()),
        anthropic.AnthropicError(CANARY),
        RuntimeError(CANARY),
    ],
)
def test_sdk_errors_become_warnings(exc):
    a = _provider(FakeClient(exc=exc)).analyze(_img(), [])
    assert _codes(a) == ["vlm_failed"]
    assert a.annotations == [] and a.proposals == [] and a.style is None
    assert CANARY not in (a.warnings[0].message or "")


def test_attach_never_overrides_cv_and_skips_invalid():
    cv = [_cv("lipstick", "lip", finish="matte"), _cv("eyeliner", "eye")]
    analysis = VlmAnalysis(
        info=StubVlmProvider().info(),
        annotations=[
            VlmAnnotationResult(0, finish="dewy", direction="upward"),
            VlmAnnotationResult(1, technique="winged", placement="lash_line"),
            VlmAnnotationResult(5, finish="matte"),
            VlmAnnotationResult(0, technique="winged"),
        ],
    )
    out = attach_annotations(cv, analysis)
    assert len(out) == 2
    assert out[0].component.finish == "matte" and out[0].sources["finish"] == "cv"
    assert out[0].component.direction == "upward" and out[0].sources["direction"] == "vlm"
    assert out[0].component.technique is None
    assert out[0].component.color_hex == "#AA3344" and out[0].sources["color_hex"] == "cv"
    assert out[1].component.technique == "winged" and out[1].sources["technique"] == "vlm"
    for key in ("color_hex", "undertone", "intensity", "target_zones"):
        assert out[1].sources.get(key) in (None, "cv")
    assert cv[0].component.direction is None


def test_attach_appends_proposals():
    prop = _provider(
        FakeClient(_ok(VlmResponse(proposals=[VlmProposal(category="eye", product_type="mascara")])))
    ).analyze(_img(), []).proposals
    out = attach_annotations([_cv()], VlmAnalysis(info=StubVlmProvider().info(), proposals=prop))
    assert [c.component.product_type for c in out] == ["lipstick", "mascara"]


def test_no_model_construct_or_copy():
    for name in ("vlm.py", "vlm_prompt.py", "claude_vlm.py"):
        text = (APP_DIR / name).read_text()
        assert "model_construct" not in text and "model_copy" not in text


def test_no_sensitive_logging(caplog):
    caplog.set_level(logging.DEBUG)
    img = _img()
    _provider(FakeClient(exc=anthropic.AnthropicError(CANARY))).analyze(img, [])
    client = FakeClient(_ok(VlmResponse()))
    _provider(client).analyze(img, [])
    b64 = client.messages.kwargs["messages"][0]["content"][0]["source"]["data"]
    assert CANARY not in caplog.text
    assert b64[:40] not in caplog.text


def test_max_retries_setting_default_and_bounds():
    assert Settings(_env_file=None).makeup_vlm_max_retries == 1
    assert Settings(_env_file=None, makeup_vlm_max_retries=0).makeup_vlm_max_retries == 0
    assert Settings(_env_file=None, makeup_vlm_max_retries=3).makeup_vlm_max_retries == 3
    for bad in (-1, 4):
        with pytest.raises(Exception):
            Settings(_env_file=None, makeup_vlm_max_retries=bad)


@pytest.mark.parametrize("retries", [0, 1, 3])
def test_anthropic_constructed_with_retries(monkeypatch, retries):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(anthropic, "Anthropic", fake)
    s = Settings(_env_file=None, makeup_vlm_enabled=True, makeup_vlm_max_retries=retries)
    p = build_vlm_provider(s)
    assert isinstance(p, ClaudeVlmProvider)
    assert seen == {"timeout": s.makeup_vlm_timeout_s, "max_retries": retries}


def test_claude_vlm_import_failure_returns_failed_provider(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "app.services.makeup_blueprint.claude_vlm", None)
    p = build_vlm_provider(Settings(_env_file=None, makeup_vlm_enabled=True))
    a = p.analyze(_img(), [])
    assert _codes(a) == ["vlm_failed"]


def test_constructor_failure_returns_failed_provider(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError(CANARY)

    monkeypatch.setattr(anthropic, "Anthropic", boom)
    p = build_vlm_provider(Settings(_env_file=None, makeup_vlm_enabled=True))
    a = p.analyze(_img(), [])
    assert _codes(a) == ["vlm_failed"]
    assert CANARY not in (a.warnings[0].message or "")
    assert p.info().kind == "vlm"


@pytest.mark.parametrize(
    "exc",
    [
        _status_exc(anthropic.AuthenticationError, 401),
        anthropic.CredentialsError(CANARY),
        TypeError(CANARY),
    ],
)
def test_request_time_credential_errors(exc):
    a = _provider(FakeClient(exc=exc)).analyze(_img(), [])
    assert _codes(a) == ["vlm_failed"]
    assert CANARY not in (a.warnings[0].message or "")


def test_overflow_is_truncated_and_accepted():
    raw = VlmResponseRaw(
        annotations=[VlmAnnotation(index=i % 32, finish="matte") for i in range(33)],
        proposals=[VlmProposal(category="eye", product_type="mascara")] * 5,
        style_labels=["natural", "glam", "editorial", "bold", "soft", "dramatic", "minimal", "smoky", "vintage"],
    )
    assert len(raw.annotations) == 33 and len(raw.proposals) == 5 and len(raw.style_labels) == 9
    a = _provider(FakeClient(_ok(raw))).analyze(_img(), [])
    assert a.warnings == []
    assert len(a.annotations) == 32
    assert len(a.proposals) == 1  # duplicates of one product_type collapse
    assert a.style is not None and len(a.style.labels) == 8


def test_overflow_four_distinct_proposals_cap():
    raw = VlmResponseRaw(
        proposals=[
            VlmProposal(category="eyebrows", product_type="brow_pencil"),
            VlmProposal(category="eyebrows", product_type="brow_powder"),
            VlmProposal(category="eyebrows", product_type="brow_gel"),
            VlmProposal(category="eye", product_type="eyeliner"),
            VlmProposal(category="eye", product_type="mascara"),
        ]
    )
    a = _provider(FakeClient(_ok(raw))).analyze(_img(), [])
    assert [p.component.product_type for p in a.proposals] == [
        "brow_pencil", "brow_powder", "brow_gel", "eyeliner",
    ]


def test_large_image_downscaled():
    big = np.zeros((2000, 3000, 3), dtype=np.uint8)
    b64, media = encode_image(big)
    assert media == "image/jpeg"
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) <= 1568
    with pytest.raises(ValueError):
        encode_image(np.zeros((4, 4), dtype=np.float32))
