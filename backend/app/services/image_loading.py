"""Decode uploaded bytes into an RGB numpy array. No MediaPipe here."""

import io

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
# Decoder allow-list. The declared content type is attacker-controlled, so
# Pillow is told which real formats it may decode (no TIFF/BMP/GIF/...).
ALLOWED_FORMATS = ["JPEG", "PNG", "WEBP"]
DEFAULT_MAX_IMAGE_PIXELS = 25_000_000


MAX_ICC_BYTES = 4 * 1024 * 1024


class ImageValidationError(ValueError):
    """The upload is not a usable image."""


def _to_srgb(img: Image.Image) -> Image.Image:
    """RGB image; converted from the embedded ICC profile to sRGB when one is usable.

    Any problem with the profile falls back silently to a plain RGB conversion.
    """
    icc = img.info.get("icc_profile")
    if not icc or not isinstance(icc, bytes) or len(icc) > MAX_ICC_BYTES:
        return img.convert("RGB")
    try:
        work = img
        if work.mode in ("RGBA", "P", "PA", "RGBa"):
            work = work.convert("RGB")
        elif work.mode in ("LA", "La"):
            work = work.convert("L")
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        dst = ImageCms.createProfile("sRGB")
        out = ImageCms.profileToProfile(work, src, dst, outputMode="RGB")
        if out is None:
            return img.convert("RGB")
        return out
    except Exception:  # noqa: BLE001 - any CMS failure falls back to the plain decode
        return img.convert("RGB")


def load_rgb_image(data: bytes, max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS) -> np.ndarray:
    """Return an (H, W, 3) uint8 RGB array.

    EXIF orientation is applied first. Phone selfies are often stored sideways
    with an EXIF "rotate" flag. Browsers honor that flag when displaying the
    image, so we must too, or the landmarks would not line up with what the
    frontend shows.

    An embedded ICC profile is converted to sRGB (default perceptual intent).
    Browsers colour-manage, so without this the pixels look less saturated
    than the image the user sees.

    The pixel count is checked from the header *before* decoding, so a small
    file that expands to a huge bitmap (decompression bomb) is rejected cheaply.
    """
    if not data:
        raise ImageValidationError("Uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(data), formats=ALLOWED_FORMATS) as img:
            if img.width * img.height > max_pixels:
                raise ImageValidationError(
                    f"Image is too large ({img.width}x{img.height}); "
                    f"the limit is {max_pixels / 1e6:g} megapixels."
                )
            img.load()  # force full decode so truncated/corrupt files fail here
            img = ImageOps.exif_transpose(img)
            rgb = _to_srgb(img)
    except ImageValidationError:
        raise
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,  # some Pillow plugins signal corrupt data this way
        ValueError,
        EOFError,
    ) as exc:
        raise ImageValidationError("File could not be decoded as an image.") from exc
    return np.asarray(rgb, dtype=np.uint8)
