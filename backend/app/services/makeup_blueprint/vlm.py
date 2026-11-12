"""VLM provider protocol, result types, disabled stub, factory and annotation merge."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from pydantic import ValidationError

from app.config import Settings
from app.schemas.makeup_blueprint import (
    BlueprintComponent,
    BlueprintStyle,
    BlueprintWarning,
    ReferenceProviderInfo,
)
from app.schemas.makeup_component import MakeupComponent

_SEMANTIC_FIELDS = ("finish", "technique", "placement", "direction")


@dataclass(frozen=True)
class VlmAnnotationResult:
    index: int
    finish: str | None = None
    technique: str | None = None
    placement: str | None = None
    direction: str | None = None


@dataclass(frozen=True)
class VlmAnalysis:
    info: ReferenceProviderInfo
    annotations: list[VlmAnnotationResult] = field(default_factory=list)
    proposals: list[BlueprintComponent] = field(default_factory=list)
    style: BlueprintStyle | None = None
    warnings: list[BlueprintWarning] = field(default_factory=list)


class VlmProvider(Protocol):
    def info(self) -> ReferenceProviderInfo: ...

    def analyze(self, image: np.ndarray, cv: list[BlueprintComponent]) -> VlmAnalysis: ...


class StubVlmProvider:
    """Used when the VLM is disabled. Deterministic; ignores the image."""

    def info(self) -> ReferenceProviderInfo:
        return ReferenceProviderInfo(
            id="vlm_stub", kind="vlm", algorithm_version="1.0", model=None, enabled=False
        )

    def analyze(self, image: np.ndarray, cv: list[BlueprintComponent]) -> VlmAnalysis:
        return VlmAnalysis(
            info=self.info(),
            warnings=[
                BlueprintWarning(code="vlm_disabled", message="Semantic analysis is disabled.")
            ],
        )


class FailedVlmProvider:
    """Used when the Claude client cannot be constructed. Keeps no exception text."""

    def __init__(self, model: str) -> None:
        self._model = model

    def info(self) -> ReferenceProviderInfo:
        return ReferenceProviderInfo(
            id="claude_vlm", kind="vlm", algorithm_version="1.0", model=self._model, enabled=True
        )

    def analyze(self, image: np.ndarray, cv: list[BlueprintComponent]) -> VlmAnalysis:
        return VlmAnalysis(
            info=self.info(),
            warnings=[BlueprintWarning(code="vlm_failed", message="Semantic analysis failed.")],
        )


def attach_annotations(cv: list[BlueprintComponent], analysis: VlmAnalysis) -> list[BlueprintComponent]:
    """Merge VLM semantic fields into CV components without overriding CV values."""
    merged: list[BlueprintComponent] = list(cv)
    for ann in analysis.annotations:
        if not 0 <= ann.index < len(cv):
            continue
        base = merged[ann.index]
        updates: dict[str, Any] = {}
        for name in _SEMANTIC_FIELDS:
            value = getattr(ann, name)
            if value is not None and getattr(base.component, name) is None:
                updates[name] = value
        if not updates:
            continue
        try:
            component = MakeupComponent.model_validate({**base.component.model_dump(), **updates})
            sources = {**base.sources, **{k: "vlm" for k in updates}}
            merged[ann.index] = BlueprintComponent.model_validate(
                {"component": component, "sources": sources}
            )
        except ValidationError:
            continue
    return merged + list(analysis.proposals)


def build_vlm_provider(settings: Settings, client: Any = None) -> VlmProvider:
    if not settings.makeup_vlm_enabled:
        return StubVlmProvider()
    try:
        from app.services.makeup_blueprint.claude_vlm import ClaudeVlmProvider

        if client is None:
            import anthropic

            client = anthropic.Anthropic(
                timeout=settings.makeup_vlm_timeout_s,
                max_retries=settings.makeup_vlm_max_retries,
            )
    except Exception:
        return FailedVlmProvider(settings.makeup_vlm_model)
    return ClaudeVlmProvider(
        client=client,
        model=settings.makeup_vlm_model,
        effort=settings.makeup_vlm_effort,
        max_tokens=settings.makeup_vlm_max_tokens,
    )
