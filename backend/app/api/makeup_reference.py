"""HTTP layer for reference analysis: validate, delegate, map errors to status codes."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile, status

from app.api.facial_profile import get_landmark_provider, get_parsing_provider, get_zone_provider
from app.api.rate_limit import rate_limit
from app.api.uploads import read_validated_image
from app.config import Settings, get_settings
from app.schemas.makeup_blueprint import MakeupBlueprint
from app.services.facial_profile.providers import LandmarkProvider, ParsingProvider, ZoneProvider
from app.services.makeup_blueprint.cv_analysis import CvReferenceAnalyzer
from app.services.makeup_blueprint.service import (
    ParserUnavailableError,
    ReferenceAnalysisError,
    ReferenceDeps,
    analyze_reference,
)
from app.services.makeup_blueprint.vlm import VlmProvider, build_vlm_provider
from app.services.segmentation.face_parser import FaceParserBusyError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/makeup", tags=["makeup-reference"])

_DETAIL_FAILED = "Reference analysis failed."
_DETAIL_BUSY = "The server is busy. Please retry shortly."
_DETAIL_PARSER = (
    "Face parser is not available. Run `python scripts/download_face_parser.py` "
    "from backend/ and restart the server. Landmark detection still works."
)


def get_vlm_provider(request: Request, settings: Settings = Depends(get_settings)) -> VlmProvider:
    """Built once per Settings instance (get_settings is lru_cached) and kept on app.state."""
    cached = getattr(request.app.state, "makeup_vlm_provider", None)
    if cached is not None and cached[0] is settings:
        return cached[1]
    provider = build_vlm_provider(settings)
    request.app.state.makeup_vlm_provider = (settings, provider)
    return provider


def get_reference_deps(
    landmark_provider: LandmarkProvider = Depends(get_landmark_provider),
    parsing_provider: ParsingProvider | None = Depends(get_parsing_provider),
    zone_provider: ZoneProvider = Depends(get_zone_provider),
    vlm_provider: VlmProvider = Depends(get_vlm_provider),
) -> ReferenceDeps:
    return ReferenceDeps(
        landmark_provider=landmark_provider,
        parsing_provider=parsing_provider,
        zone_provider=zone_provider,
        cv_analyzer=CvReferenceAnalyzer(),
        vlm_provider=vlm_provider,
    )


@router.post(
    "/reference-analysis", response_model=MakeupBlueprint, dependencies=[Depends(rate_limit("reference"))]
)
def reference_analysis_route(
    response: Response,
    file: UploadFile = File(..., description="Reference makeup image (JPEG, PNG or WebP)"),
    deps: ReferenceDeps = Depends(get_reference_deps),
    settings: Settings = Depends(get_settings),
) -> MakeupBlueprint:
    image = read_validated_image(file, settings)
    response.headers["Cache-Control"] = "no-store"
    try:
        return analyze_reference(image, deps=deps)
    except FaceParserBusyError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, _DETAIL_BUSY, headers={"Retry-After": "2"}
        ) from None
    except ParserUnavailableError:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, _DETAIL_PARSER) from None
    except ReferenceAnalysisError as exc:
        logger.error("Reference analysis failed: %s (stage=%s)", type(exc).__name__, exc.stage)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, _DETAIL_FAILED) from None
    except Exception as exc:  # noqa: BLE001 - never leak details of image-derived failures
        logger.error("Reference analysis failed: %s", type(exc).__name__)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, _DETAIL_FAILED) from None
