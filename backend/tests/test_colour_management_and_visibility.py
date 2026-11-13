"""Phase 9a follow-up: ICC colour management in load_rgb_image and renderer opacity/stage visibility.

All images are synthetic and generated in-test. No user photos.
"""

import io
import logging
import struct

import numpy as np
import pytest
from PIL import Image, ImageCms, ImageOps

from app.schemas.adapted_makeup_plan import Adjustment, AdaptedComponent, AdaptedMakeupPlan, ZoneSelection
from app.schemas.face_segmentation import ZoneBBox, ZoneMask
from app.schemas.facial_profile import ImageSize
from app.schemas.makeup_component import MakeupComponent
from app.services import image_loading
from app.services.image_loading import ImageValidationError, load_rgb_image
from app.services.makeup_preview.color import srgb_to_lab
from app.services.makeup_preview.renderer import render_layers

import base64
import cv2

H, W = 48, 64


# --- helpers -------------------------------------------------------------------------

def _rgb(seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).integers(40, 220, size=(H, W, 3), dtype=np.uint8)


def _srgb_profile() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _gamma_profile(gamma: float = 1.8) -> bytes:
    """Valid RGB profile that is NOT sRGB: sRGB with the TRC parametric gamma patched in place."""
    data = bytearray(_srgb_profile())
    count = struct.unpack(">I", data[128:132])[0]
    for i in range(count):
        base = 132 + 12 * i
        sig = bytes(data[base : base + 4])
        off = struct.unpack(">I", data[base + 4 : base + 8])[0]
        if sig in (b"rTRC", b"gTRC", b"bTRC"):
            assert bytes(data[off : off + 4]) == b"para"
            assert struct.unpack(">H", data[off + 8 : off + 10])[0] == 3  # 5-param curve, g first
            data[off + 12 : off + 16] = struct.pack(">i", int(round(gamma * 65536)))
    out = bytes(data)
    ImageCms.ImageCmsProfile(io.BytesIO(out))  # must still be a loadable profile
    return out


def _png(img: Image.Image, icc: bytes | None = None) -> bytes:
    buf = io.BytesIO()
    kwargs = {"icc_profile": icc} if icc else {}
    img.save(buf, format="PNG", **kwargs)
    return buf.getvalue()


def _naive(data: bytes) -> Image.Image:
    """The plain decode the production code must reproduce when no usable profile exists."""
    with Image.open(io.BytesIO(data), formats=["JPEG", "PNG", "WEBP"]) as img:
        img.load()
        return ImageOps.exif_transpose(img).convert("RGB")


def _naive_arr(data: bytes) -> np.ndarray:
    return np.asarray(_naive(data), dtype=np.uint8)


# --- Criterion 1: no ICC -> byte-identical to plain decode ---------------------------

def test_png_without_icc_matches_plain_decode():
    data = _png(Image.fromarray(_rgb()))
    out = load_rgb_image(data)
    assert out.dtype == np.uint8 and out.shape == (H, W, 3)
    assert np.array_equal(out, _naive_arr(data))


def test_jpeg_without_icc_matches_plain_decode_after_exif_transpose():
    img = Image.fromarray(_rgb())
    exif = img.getexif()
    exif[0x0112] = 6  # orientation: rotate 90 -> dimensions swap
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90, exif=exif.tobytes())
    data = buf.getvalue()
    out = load_rgb_image(data)
    assert out.shape == (W, H, 3)
    assert np.array_equal(out, _naive_arr(data))


# --- Criterion 2: embedded sRGB profile -> within +/-1 of plain decode ---------------

def test_srgb_profile_png_within_one_of_plain_decode():
    data = _png(Image.fromarray(_rgb()), icc=_srgb_profile())
    out = load_rgb_image(data).astype(np.int16)
    assert out.shape == (H, W, 3)
    assert np.abs(out - _naive_arr(data).astype(np.int16)).max() <= 1


# --- Criterion 3: non-sRGB profile is actually converted -----------------------------

def test_non_srgb_profile_is_converted_and_matches_profile_to_profile():
    icc = _gamma_profile(1.8)
    data = _png(Image.fromarray(_rgb()), icc=icc)
    out = load_rgb_image(data)

    naive = _naive_arr(data).astype(np.int16)
    assert np.abs(out.astype(np.int16) - naive).max() >= 3, "profile was not applied"

    src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    dst = ImageCms.createProfile("sRGB")
    expected = np.asarray(ImageCms.profileToProfile(_naive(data), src, dst, outputMode="RGB"), dtype=np.uint8)
    assert np.array_equal(out, expected)


# --- Criterion 4: malformed / oversize ICC -> silent plain-decode fallback -----------

def _app_log_records(caplog) -> list:
    """Records from anything except Pillow's own PNG-parser debug output (not production logging)."""
    return [r for r in caplog.records if not r.name.startswith("PIL")]


def test_malformed_icc_falls_back_silently(caplog):
    junk = np.random.default_rng(9).bytes(512)
    data = _png(Image.fromarray(_rgb()), icc=junk)
    with caplog.at_level(logging.DEBUG):
        out = load_rgb_image(data)
    assert np.array_equal(out, _naive_arr(data))
    assert _app_log_records(caplog) == []


def test_oversize_icc_falls_back_silently(monkeypatch, caplog):
    icc = _gamma_profile(1.8)
    data = _png(Image.fromarray(_rgb()), icc=icc)
    # Sanity: without the cap this profile does get applied, so the fallback is due to the cap.
    converted = load_rgb_image(data)
    assert np.abs(converted.astype(np.int16) - _naive_arr(data).astype(np.int16)).max() >= 3

    monkeypatch.setattr(image_loading, "MAX_ICC_BYTES", 64)
    with caplog.at_level(logging.DEBUG):
        out = load_rgb_image(data)
    assert np.array_equal(out, _naive_arr(data))
    assert _app_log_records(caplog) == []


def test_empty_and_garbage_bytes_raise_validation_error():
    with pytest.raises(ImageValidationError):
        load_rgb_image(b"")
    with pytest.raises(ImageValidationError):
        load_rgb_image(b"not an image at all" * 20)


# --- Criterion 5: RGBA / P / L modes with and without profile -> HxWx3 uint8 ---------

def test_rgba_png_with_srgb_profile_decodes_to_rgb():
    rgba = np.dstack([_rgb(), np.full((H, W), 255, np.uint8)])
    data = _png(Image.fromarray(rgba, mode="RGBA"), icc=_srgb_profile())
    out = load_rgb_image(data)
    assert out.dtype == np.uint8 and out.shape == (H, W, 3)


def test_p_mode_png_with_srgb_profile_decodes_to_rgb():
    pal = Image.fromarray(_rgb()).quantize(colors=32)
    assert pal.mode == "P"
    data = _png(pal, icc=_srgb_profile())
    out = load_rgb_image(data)
    assert out.dtype == np.uint8 and out.shape == (H, W, 3)


def test_l_mode_png_with_srgb_profile_decodes_to_rgb():
    gray = Image.fromarray(_rgb()[..., 0], mode="L")
    data = _png(gray, icc=_srgb_profile())
    out = load_rgb_image(data)
    assert out.dtype == np.uint8 and out.shape == (H, W, 3)


def test_l_mode_png_without_profile_decodes_to_rgb():
    gray = Image.fromarray(_rgb()[..., 0], mode="L")
    out = load_rgb_image(_png(gray))
    assert out.dtype == np.uint8 and out.shape == (H, W, 3)


# --- Renderer fixtures ---------------------------------------------------------------

RED = "#e02020"
DARK = "#202020"


def _ellipse(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    inside = ((xx - cx) / (w / 2.0)) ** 2 + ((yy - cy) / (h / 2.0)) ** 2 <= 1.0
    return inside.astype(np.uint8) * 255


def _comp(
    index: int,
    *,
    rect: tuple[int, int, int, int] = (20, 16, 30, 24),
    color: str | None = RED,
    intensity: float | None = 1.0,
    finish: str | None = "matte",
    stage_key: str | None = None,
    multipliers: tuple[float, ...] = (),
) -> AdaptedComponent:
    x, y, w, h = rect
    arr = _ellipse(h, w)
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    component = MakeupComponent(
        category="eye", product_type="eyeshadow", color_hex=color, finish=finish, intensity=intensity
    )
    adjustments = [
        Adjustment(
            rule_id="lift_downturned_eyes",
            reason="lift_downturned_eyes",
            effect="intensity",
            intensity_multiplier=m,
            direction_angles=[],
        )
        for m in multipliers
    ]
    return AdaptedComponent(
        index=index,
        component=component,
        stage_key=stage_key,
        zone_selection=ZoneSelection(source="placement"),
        mask=ZoneMask(png_base64=base64.b64encode(buf.tobytes()).decode()),
        bbox=ZoneBBox(x=x, y=y, width=w, height=h),
        centroid=(x + w / 2.0, y + h / 2.0),
        pixel_count=int((arr > 0).sum()),
        adjustments=adjustments,
    )


def _plan(*comps: AdaptedComponent) -> AdaptedMakeupPlan:
    return AdaptedMakeupPlan(status="ok", image=ImageSize(width=W, height=H), components=list(comps))


def _report(comp: AdaptedComponent):
    res = render_layers(_rgb(), _plan(comp), seed=0)
    assert len(res.reports) == 1
    return res.reports[0]


# --- Criterion 6: opacity mapping ----------------------------------------------------

def test_opacity_intensity_0_2_is_0_52():
    r = _report(_comp(0, intensity=0.2))
    assert r.status == "rendered"
    assert r.opacity == pytest.approx(0.52)


def test_opacity_none_intensity_is_0_7():
    r = _report(_comp(0, intensity=None))
    assert r.status == "rendered"
    assert r.opacity == pytest.approx(0.7)


def test_intensity_zero_is_zero_opacity_skip():
    r = _report(_comp(0, intensity=0.0))
    assert r.status == "skipped"
    assert r.reason == "zero_opacity"
    assert r.opacity == 0.0
    assert r.pixels_touched == 0


def test_multiplier_zero_is_zero_opacity_skip():
    r = _report(_comp(0, intensity=1.0, multipliers=(0.0,)))
    assert r.status == "skipped"
    assert r.reason == "zero_opacity"
    assert r.opacity == 0.0


def test_multiplier_half_at_full_intensity_is_0_5():
    r = _report(_comp(0, intensity=1.0, multipliers=(0.5,)))
    assert r.status == "rendered"
    assert r.opacity == pytest.approx(0.5)


# --- Criterion 7: lips shift L more than blush for identical geometry/target/opacity -

def _mean_abs_dL(stage_key: str) -> float:
    img = _rgb()
    comp = _comp(0, color=DARK, intensity=1.0, stage_key=stage_key)
    res = render_layers(img, _plan(comp), seed=0)
    inside = res.alpha_union > 0.5
    assert inside.any()
    dL = srgb_to_lab(res.image)[..., 0] - srgb_to_lab(img)[..., 0]
    return float(np.abs(dL[inside]).mean())


def test_lips_shift_l_more_than_blush():
    lips = _mean_abs_dL("lips")
    blush = _mean_abs_dL("blush")
    assert blush > 0.0
    assert lips > 1.5 * blush, f"lips={lips:.3f} blush={blush:.3f}"


def test_eyebrow_eye_stage_shifts_l_more_than_default():
    assert _mean_abs_dL("eyebrow_eye") > 1.2 * _mean_abs_dL("blush")


# --- Criterion 8: invariants (alpha==0 untouched, deterministic) ---------------------

def test_alpha_zero_pixels_are_byte_identical_and_output_deterministic():
    img = _rgb()
    plan = _plan(
        _comp(0, stage_key="lips", color=DARK, intensity=0.9, finish="satin"),
        _comp(1, rect=(44, 30, 14, 12), stage_key="blush", intensity=None, multipliers=(0.5,)),
    )
    a = render_layers(img, plan, seed=5)
    b = render_layers(img, plan, seed=5)
    assert np.array_equal(a.image, b.image)
    assert a.reports == b.reports
    zero = a.alpha_union == 0
    assert zero.any() and not zero.all()
    assert np.array_equal(a.image[zero], img[zero])
