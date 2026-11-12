"""Swappable providers for the FacialProfile builder, with default adapters over the existing services."""

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Protocol

import numpy as np

from app.schemas.face_segmentation import LICENSE_WARNING
from app.schemas.facial_landmarks import FacialLandmarkResult
from app.schemas.facial_profile import ProviderInfo
from app.services.application_zones.derive import DerivedZone, derive_zones
from app.services.application_zones.frame import FaceFrame
from app.services.facial_analysis.landmarks import FaceLandmarkService
from app.services.facial_profile.versions import FRAME_ALGORITHM_VERSION, ZONES_ALGORITHM_VERSION
from app.services.segmentation.face_parser import FaceParser, ParseResult


class LandmarkProvider(Protocol):
    def info(self) -> ProviderInfo: ...

    def detect(self, image: np.ndarray) -> FacialLandmarkResult: ...


class ParsingProvider(Protocol):
    def info(self) -> ProviderInfo: ...

    def parse(self, image: np.ndarray, bbox: tuple[float, float, float, float]) -> ParseResult: ...


class ZoneProvider(Protocol):
    def info(self) -> ProviderInfo: ...

    def derive(
        self,
        frame: FaceFrame,
        labels: np.ndarray,
        confidence: np.ndarray,
        origin: tuple[int, int],
    ) -> list[DerivedZone]: ...


def _file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _mediapipe_version() -> str:
    try:
        return f"mediapipe-{importlib.metadata.version('mediapipe')}"
    except Exception:  # noqa: BLE001 - metadata lookup must never fail the request
        return "mediapipe-unknown"


class MediaPipeLandmarkProvider:
    def __init__(self, service: FaceLandmarkService, model_path: Path | None = None) -> None:
        self._service = service
        self._info = ProviderInfo(
            id="mediapipe-face-landmarker",
            kind="landmarks",
            weights_sha256=_file_sha256(model_path),
            algorithm_version=_mediapipe_version(),
        )

    def info(self) -> ProviderInfo:
        return self._info

    def detect(self, image: np.ndarray) -> FacialLandmarkResult:
        return self._service.detect_landmarks(image)


class DefaultParsingProvider:
    def __init__(self, parser: FaceParser) -> None:
        self._parser = parser

    def info(self) -> ProviderInfo:
        i = self._parser.info
        return ProviderInfo(
            id=i.id,
            kind="parsing",
            weights_sha256=i.weights_sha256,
            algorithm_version=f"{i.architecture}; input={i.input_size}",
            license_warning=LICENSE_WARNING,
        )

    def parse(self, image: np.ndarray, bbox: tuple[float, float, float, float]) -> ParseResult:
        return self._parser.parse(image, bbox)


class DefaultZoneProvider:
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            id="application-zones",
            kind="zones",
            algorithm_version=ZONES_ALGORITHM_VERSION,
            license_warning=LICENSE_WARNING,
        )

    def derive(
        self,
        frame: FaceFrame,
        labels: np.ndarray,
        confidence: np.ndarray,
        origin: tuple[int, int],
    ) -> list[DerivedZone]:
        return derive_zones(frame, labels, confidence, origin)


def frame_provider_info() -> ProviderInfo:
    return ProviderInfo(id="face-frame-ipd", kind="landmarks", algorithm_version=FRAME_ALGORITHM_VERSION)
