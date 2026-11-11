"""HTTP layer only: validate the upload, call services, shape the response."""

from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.api.uploads import read_validated_image
from app.config import Settings, get_settings
from app.schemas.face_measurements import FacialLandmarkResponse
from app.schemas.facial_landmarks import FaceMeshTopology
from app.services.facial_analysis.geometry import compute_basic_measurements
from app.services.facial_analysis.landmarks import FaceLandmarkService, get_face_mesh_topology

router = APIRouter(prefix="/api/facial-analysis", tags=["facial-analysis"])


def get_landmark_service(request: Request) -> FaceLandmarkService:
    """The service is created once at startup (see main.py). It is injected here so tests can override it."""
    return request.app.state.landmark_service


# A sync `def` route: FastAPI runs it in a worker thread, so the CPU-bound
# MediaPipe call doesn't block the event loop.
@router.post("/landmarks", response_model=FacialLandmarkResponse)
def detect_landmarks(
    file: UploadFile = File(..., description="Selfie image (JPEG, PNG or WebP)"),
    service: FaceLandmarkService = Depends(get_landmark_service),
    settings: Settings = Depends(get_settings),
) -> FacialLandmarkResponse:
    image = read_validated_image(file, settings)

    result = service.detect_landmarks(image)
    # No face / multiple faces are not HTTP errors: the request was valid, so
    # we return 200 with a status and let the client decide what to do.
    return FacialLandmarkResponse(
        **result.model_dump(),
        measurements=compute_basic_measurements(result),
    )


@router.get("/topology", response_model=FaceMeshTopology)
def face_mesh_topology() -> FaceMeshTopology:
    """Static mesh connections for drawing; fetch once on the client."""
    return get_face_mesh_topology()
