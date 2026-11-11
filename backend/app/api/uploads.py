"""Upload validation shared by the facial-analysis routes (HTTP concerns only)."""

import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.config import Settings
from app.services.image_loading import (
    ALLOWED_CONTENT_TYPES,
    ImageValidationError,
    load_rgb_image,
)


def read_validated_image(file: UploadFile, settings: Settings) -> np.ndarray:
    """Validate type, size and content of an upload and decode it to RGB.

    415 wrong declared type, 413 too many bytes, 400 undecodable / wrong real
    format / too many pixels.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type '{file.content_type}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}.",
        )

    data = file.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"Image exceeds {settings.max_upload_mb:g} MB limit.",
        )

    try:
        return load_rgb_image(data, max_pixels=settings.max_image_pixels)
    except ImageValidationError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
