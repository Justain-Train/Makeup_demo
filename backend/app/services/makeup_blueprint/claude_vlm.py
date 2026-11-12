"""Claude-backed VLM provider. Semantic attributes only; never raises, never logs."""

from typing import Any

import anthropic
import numpy as np
from pydantic import ValidationError

from app.schemas.makeup_blueprint import (
    BlueprintComponent,
    BlueprintStyle,
    BlueprintWarning,
    ReferenceProviderInfo,
)
from app.schemas.makeup_component import MakeupComponent
from app.services.makeup_blueprint.vlm import VlmAnalysis, VlmAnnotationResult
from app.services.makeup_blueprint.vlm_prompt import (
    MAX_ANNOTATIONS,
    MAX_PROPOSALS,
    MAX_STYLE_LABELS,
    PROPOSAL_CATEGORIES,
    PROPOSAL_PRODUCT_TYPES,
    SYSTEM_PROMPT,
    VlmResponse,
    VlmResponseRaw,
    build_messages,
    encode_image,
)

_BETAS = ["server-side-fallback-2026-07-01"]

_MSG_FAILED = "Semantic analysis failed."
_MSG_REFUSED = "Semantic analysis was refused."
_MSG_AUTH = "Semantic analysis is not authorized."
_MSG_NOT_FOUND = "Semantic analysis model was not found."
_MSG_RATE = "Semantic analysis is rate limited."
_MSG_STATUS = "Semantic analysis service returned an error."
_MSG_TIMEOUT = "Semantic analysis timed out."
_MSG_CONNECTION = "Semantic analysis service is unreachable."
_MSG_BAD_IMAGE = "Reference image could not be prepared for analysis."
_MSG_BAD_RESPONSE = "Semantic analysis returned an unusable response."


def _error_message(exc: BaseException) -> str:
    if isinstance(
        exc,
        (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.CredentialsError),
    ):
        return _MSG_AUTH
    if isinstance(exc, anthropic.NotFoundError):
        return _MSG_NOT_FOUND
    if isinstance(exc, anthropic.RateLimitError):
        return _MSG_RATE
    if isinstance(exc, anthropic.APIStatusError):
        return _MSG_STATUS
    if isinstance(exc, anthropic.APITimeoutError):
        return _MSG_TIMEOUT
    if isinstance(exc, anthropic.APIConnectionError):
        return _MSG_CONNECTION
    return _MSG_FAILED


class ClaudeVlmProvider:
    def __init__(self, client: Any, model: str, effort: str, max_tokens: int) -> None:
        self._client = client
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens

    def info(self) -> ReferenceProviderInfo:
        return ReferenceProviderInfo(
            id="claude_vlm", kind="vlm", algorithm_version="1.0", model=self._model, enabled=True
        )

    def _fail(self, code: str, message: str) -> VlmAnalysis:
        return VlmAnalysis(
            info=self.info(),
            warnings=[BlueprintWarning(code=code, message=message)],  # type: ignore[arg-type]
        )

    def analyze(self, image: np.ndarray, cv: list[BlueprintComponent]) -> VlmAnalysis:
        try:
            image_b64, media_type = encode_image(image)
        except Exception:
            return self._fail("vlm_failed", _MSG_BAD_IMAGE)
        messages = build_messages(image_b64, media_type, cv)
        try:
            parsed = self._client.beta.messages.parse(
                model=self._model,
                max_tokens=self._max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self._effort},
                output_format=VlmResponseRaw,
                betas=_BETAS,
                fallbacks="default",
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except anthropic.AnthropicError as exc:
            return self._fail("vlm_failed", _error_message(exc))
        except Exception:
            return self._fail("vlm_failed", _MSG_FAILED)
        try:
            return self._build(parsed, cv)
        except Exception:
            return self._fail("vlm_failed", _MSG_BAD_RESPONSE)

    def _build(self, parsed: Any, cv: list[BlueprintComponent]) -> VlmAnalysis:
        stop = getattr(parsed, "stop_reason", None)
        if stop == "refusal":
            return self._fail("vlm_refused", _MSG_REFUSED)
        if stop == "max_tokens":
            return self._fail("vlm_failed", _MSG_BAD_RESPONSE)
        out = getattr(parsed, "parsed_output", None)
        if not isinstance(out, (VlmResponse, VlmResponseRaw)):
            return self._fail("vlm_failed", _MSG_BAD_RESPONSE)
        try:
            data = out.model_dump()
            data["proposals"] = data["proposals"][:MAX_PROPOSALS]
            data["annotations"] = data["annotations"][:MAX_ANNOTATIONS]
            data["style_labels"] = data["style_labels"][:MAX_STYLE_LABELS]
            resp = VlmResponse.model_validate(data)
            annotations = [
                VlmAnnotationResult(
                    index=a.index,
                    finish=a.finish,
                    technique=a.technique,
                    placement=a.placement,
                    direction=a.direction,
                )
                for a in resp.annotations
            ]
            seen = {c.component.product_type for c in cv}
            proposals: list[BlueprintComponent] = []
            for p in resp.proposals:
                if p.category not in PROPOSAL_CATEGORIES or p.product_type not in PROPOSAL_PRODUCT_TYPES:
                    continue
                if p.product_type in seen or len(proposals) >= MAX_PROPOSALS:
                    continue
                fields = {
                    k: v
                    for k, v in (
                        ("finish", p.finish),
                        ("technique", p.technique),
                        ("placement", p.placement),
                        ("direction", p.direction),
                    )
                    if v is not None
                }
                component = MakeupComponent.model_validate(
                    {"category": p.category, "product_type": p.product_type, **fields}
                )
                sources = {"category": "vlm", "product_type": "vlm", **{k: "vlm" for k in fields}}
                proposals.append(
                    BlueprintComponent.model_validate({"component": component, "sources": sources})
                )
                seen.add(p.product_type)
            style = None
            if resp.style_labels or resp.style_confidence is not None:
                style = BlueprintStyle.model_validate(
                    {"labels": list(resp.style_labels), "confidence": resp.style_confidence}
                )
        except ValidationError:
            return self._fail("vlm_failed", _MSG_BAD_RESPONSE)
        return VlmAnalysis(
            info=self.info(), annotations=annotations, proposals=proposals, style=style
        )
