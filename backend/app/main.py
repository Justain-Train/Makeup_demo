"""FastAPI entrypoint. Run from backend/:  uvicorn app.main:app --reload"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import face_segmentation, facial_analysis, facial_profile
from app.config import get_settings
from app.services.facial_analysis.landmarks import FaceLandmarkService
from app.services.segmentation.face_parser import FaceParserError, OnnxFaceParser

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Load the model once at startup. If the model file is missing, startup
    # fails immediately with an error message saying how to download it.
    service = FaceLandmarkService(
        model_path=settings.resolved_model_path,
        min_face_detection_confidence=settings.min_face_detection_confidence,
        min_face_presence_confidence=settings.min_face_presence_confidence,
        secondary_face_area_ratio=settings.secondary_face_area_ratio,
    )
    # From here on the landmarker exists: close it on ANY exit, including a
    # failure while the rest of startup runs (an unclosed MediaPipe landmarker
    # can hang the process when it is garbage-collected).
    try:
        app.state.landmark_service = service
        # The face parser is optional at startup: if its weights are missing,
        # unreadable or fail verification, landmarks keep working and
        # /segmentation answers 503.
        app.state.face_parser = None
        try:
            app.state.face_parser = OnnxFaceParser(
                model_path=settings.resolved_face_parser_path,
                expected_sha256=settings.face_parser_sha256,
                crop_scale=settings.face_parser_crop_scale,
                max_concurrency=settings.face_parser_max_concurrency,
                queue_timeout_s=settings.face_parser_queue_timeout_s,
                intra_op_threads=settings.face_parser_threads,
            )
        except FaceParserError as exc:
            logger.error("Face parser unavailable, /segmentation will return 503: %s", exc)
        except Exception as exc:  # noqa: BLE001 - never let the optional parser stop the API
            logger.error("Face parser failed to start (%s), /segmentation will return 503", type(exc).__name__)
        yield
    finally:
        service.close()


def create_app(use_lifespan: bool = True) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="AI Makeup Assistant API", lifespan=lifespan if use_lifespan else None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(facial_analysis.router)
    app.include_router(face_segmentation.router)
    app.include_router(facial_profile.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
