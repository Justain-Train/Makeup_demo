"""HTTP layer for face segmentation: validate, delegate, map errors to status codes."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.api.facial_analysis import get_landmark_service
from app.api.uploads import read_validated_image
from app.config import Settings, get_settings
from app.schemas.face_segmentation import FaceSegmentationResponse
from app.services.face_regions import segment_face
from app.services.facial_analysis.landmarks import FaceLandmarkService
from app.services.segmentation.face_parser import FaceParser, FaceParserBusyError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/facial-analysis", tags=["face-segmentation"])


def get_face_parser(request: Request) -> FaceParser | None:
    """None when the weights are missing/corrupt (the landmark endpoints keep working)."""
    return getattr(request.app.state, "face_parser", None)


@router.post("/segmentation", response_model=FaceSegmentationResponse)
def segment_face_route(
    file: UploadFile = File(..., description="Selfie image (JPEG, PNG or WebP)"),
    landmark_service: FaceLandmarkService = Depends(get_landmark_service),
    parser: FaceParser | None = Depends(get_face_parser),
    settings: Settings = Depends(get_settings),
) -> FaceSegmentationResponse:
    image = read_validated_image(file, settings)
    if parser is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Face parser is not available. Run `python scripts/download_face_parser.py` "
            "from backend/ and restart the server. Landmark detection still works.",
        )
    try:
        return segment_face(image, landmark_service, parser)
    except FaceParserBusyError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The server is busy. Please retry shortly.",
            headers={"Retry-After": "2"},
        ) from exc
    except Exception as exc:  # noqa: BLE001 - never leak details of image-derived failures
        # Log only the exception type: messages/tracebacks could contain image data.
        logger.error("Segmentation failed: %s", type(exc).__name__)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Face segmentation failed.") from None
