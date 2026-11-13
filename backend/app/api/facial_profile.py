"""HTTP layer for the FacialProfile: validate, delegate, map errors to status codes."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status

from app.api.face_segmentation import get_face_parser
from app.api.facial_analysis import get_landmark_service
from app.api.rate_limit import rate_limit
from app.api.uploads import read_validated_image
from app.config import Settings, get_settings
from app.schemas.facial_profile import FacialProfile
from app.services.facial_analysis.landmarks import FaceLandmarkService
from app.services.facial_profile.builder import build_profile
from app.services.facial_profile.providers import (
    DefaultFeatureProvider,
    DefaultParsingProvider,
    DefaultZoneProvider,
    FeatureProvider,
    LandmarkProvider,
    MediaPipeLandmarkProvider,
    ParsingProvider,
    ZoneProvider,
)
from app.services.segmentation.face_parser import FaceParser, FaceParserBusyError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/facial-analysis", tags=["facial-profile"])


def get_landmark_provider(
    request: Request,
    service: FaceLandmarkService = Depends(get_landmark_service),
    settings: Settings = Depends(get_settings),
) -> LandmarkProvider:
    """Built once per service instance and kept on app.state, so the weights are hashed once."""
    cached = getattr(request.app.state, "profile_landmark_provider", None)
    if cached is not None and cached[0] is service:
        return cached[1]
    provider = MediaPipeLandmarkProvider(service, settings.resolved_model_path)
    request.app.state.profile_landmark_provider = (service, provider)
    return provider


def get_parsing_provider(parser: FaceParser | None = Depends(get_face_parser)) -> ParsingProvider | None:
    return None if parser is None else DefaultParsingProvider(parser)


def get_zone_provider() -> ZoneProvider:
    return DefaultZoneProvider()


def get_feature_provider() -> FeatureProvider:
    return DefaultFeatureProvider()


@router.post("/profile", response_model=FacialProfile, dependencies=[Depends(rate_limit("profile"))])
def facial_profile_route(
    file: UploadFile = File(..., description="Selfie image (JPEG, PNG or WebP)"),
    include_masks: bool = Query(True, description="Include the label map and zone masks"),
    landmark_provider: LandmarkProvider = Depends(get_landmark_provider),
    parsing_provider: ParsingProvider | None = Depends(get_parsing_provider),
    zone_provider: ZoneProvider = Depends(get_zone_provider),
    feature_provider: FeatureProvider = Depends(get_feature_provider),
    settings: Settings = Depends(get_settings),
) -> FacialProfile:
    image = read_validated_image(file, settings)
    if parsing_provider is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Face parser is not available. Run `python scripts/download_face_parser.py` "
            "from backend/ and restart the server. Landmark detection still works.",
        )
    try:
        return build_profile(
            image,
            landmark_provider,
            parsing_provider,
            zone_provider,
            include_masks,
            feature_provider=feature_provider,
        )
    except FaceParserBusyError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The server is busy. Please retry shortly.",
            headers={"Retry-After": "2"},
        ) from exc
    except Exception as exc:  # noqa: BLE001 - never leak details of image-derived failures
        logger.error("Facial profile failed: %s", type(exc).__name__)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Facial profile failed.") from None
