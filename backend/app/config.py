"""Application settings, loaded from environment variables / backend/.env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    face_landmarker_model_path: Path = Path("models/face_landmarker.task")
    cors_origins: str = "http://localhost:3000"
    max_upload_mb: float = 10
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    # A face is "significant" if its bbox area >= this fraction of the largest face's.
    secondary_face_area_ratio: float = Field(default=0.20, gt=0, le=1)
    # Decoded-pixel cap (25 MP). Checked from the header before decoding.
    max_image_pixels: int = Field(default=25_000_000, gt=0)

    # Face parsing (anatomical segmentation). CelebAMask-HQ weights: NON-COMMERCIAL.
    face_parser_model_path: Path = Path("models/face_parser_resnet18.onnx")
    face_parser_sha256: str = "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f"
    face_parser_crop_scale: float = Field(default=2.0, ge=1.0, le=4.0)
    face_parser_max_concurrency: int = Field(default=2, ge=1, le=16)
    face_parser_queue_timeout_s: float = Field(default=5.0, gt=0)
    face_parser_threads: int = Field(default=0, ge=0, le=64)  # 0 = onnxruntime default

    # Makeup VLM (semantic annotations only). Off by default. API key comes from the SDK's env.
    makeup_vlm_enabled: bool = False
    makeup_vlm_model: str = Field(default="claude-opus-5-5", pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    makeup_vlm_effort: Literal["low", "medium", "high"] = "medium"
    makeup_vlm_timeout_s: float = Field(default=30, gt=0, le=300)
    makeup_vlm_max_tokens: int = Field(default=16000, ge=1024, le=64000)
    makeup_vlm_max_retries: int = Field(default=1, ge=0, le=3)

    # Per-client token bucket on the upload routes (in-process; per worker).
    rate_limit_enabled: bool = True
    rate_limit_per_min: float = Field(default=10, gt=0)
    rate_limit_burst: int = Field(default=3, ge=1)
    rate_limit_max_keys: int = Field(default=10_000, ge=1)

    @property
    def resolved_model_path(self) -> Path:
        """Relative model paths are resolved against backend/, not the CWD."""
        path = self.face_landmarker_model_path
        return path if path.is_absolute() else BACKEND_DIR / path

    @property
    def resolved_face_parser_path(self) -> Path:
        path = self.face_parser_model_path
        return path if path.is_absolute() else BACKEND_DIR / path

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
