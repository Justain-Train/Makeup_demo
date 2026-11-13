"""HTTP layer for the makeup preview: validate both uploads, delegate, map errors to status codes."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status

from app.api.facial_profile import (
    get_feature_provider,
    get_landmark_provider,
    get_parsing_provider,
    get_zone_provider,
)
from app.api.makeup_reference import get_reference_deps
from app.api.rate_limit import rate_limit
from app.api.uploads import read_validated_image
from app.config import Settings, get_settings
from app.schemas.makeup_preview import PreviewResponse
from app.services.facial_profile.providers import FeatureProvider, LandmarkProvider, ParsingProvider, ZoneProvider
from app.services.makeup_adaptation.pipeline import adapt_from_images
from app.services.makeup_blueprint.service import ParserUnavailableError, ReferenceAnalysisError, ReferenceDeps
from app.services.makeup_mapping import CatalogDriftError
from app.services.makeup_preview import build_preview
from app.services.segmentation.face_parser import FaceParserBusyError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/makeup", tags=["makeup-preview"])

_DETAIL_FAILED = "Makeup preview failed."
_DETAIL_BUSY = "The server is busy. Please retry shortly."
_DETAIL_PARSER = (
    "Face parser is not available. Run `python scripts/download_face_parser.py` "
    "from backend/ and restart the server. Landmark detection still works."
)


@router.post("/preview", response_model=PreviewResponse, dependencies=[Depends(rate_limit("preview"))])
def preview_route(
    response: Response,
    selfie: UploadFile = File(..., description="Selfie image (JPEG, PNG or WebP)"),
    reference: UploadFile = File(..., description="Reference makeup image (JPEG, PNG or WebP)"),
    landmark_provider: LandmarkProvider = Depends(get_landmark_provider),
    parsing_provider: ParsingProvider | None = Depends(get_parsing_provider),
    zone_provider: ZoneProvider = Depends(get_zone_provider),
    feature_provider: FeatureProvider = Depends(get_feature_provider),
    reference_deps: ReferenceDeps = Depends(get_reference_deps),
    settings: Settings = Depends(get_settings),
) -> PreviewResponse:
    selfie_image = read_validated_image(selfie, settings)
    reference_image = read_validated_image(reference, settings)
    if parsing_provider is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, _DETAIL_PARSER)
    response.headers["Cache-Control"] = "no-store"
    try:
        plan = adapt_from_images(
            selfie_image,
            reference_image,
            (landmark_provider, parsing_provider, zone_provider, feature_provider),
            reference_deps,
            True,
        )
        return build_preview(selfie_image, plan)
    except FaceParserBusyError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, _DETAIL_BUSY, headers={"Retry-After": "2"}
        ) from None
    except ParserUnavailableError:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, _DETAIL_PARSER) from None
    except ReferenceAnalysisError as exc:
        logger.error("Makeup preview failed: %s (stage=%s)", type(exc).__name__, exc.stage)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, _DETAIL_FAILED) from None
    except CatalogDriftError as exc:
        logger.error("Makeup preview failed: %s (stage=adapt)", type(exc).__name__)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, _DETAIL_FAILED) from None
    except Exception as exc:  # noqa: BLE001 - never leak details of image-derived failures
        logger.error("Makeup preview failed: %s (stage=pipeline)", type(exc).__name__)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, _DETAIL_FAILED) from None
