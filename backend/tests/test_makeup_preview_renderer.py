"""Phase 9a-1 renderer: compositing, masks, skip reasons, determinism, reports and input validation.

Plans are built only through the Pydantic schemas (no model_construct / model_copy).
"""

import base64

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from app.schemas.adapted_makeup_plan import (
    Adjustment,
    AdaptedComponent,
    AdaptedMakeupPlan,
    DirectionAngle,
    ZoneSelection,
)
from app.schemas.face_segmentation import ZoneBBox, ZoneMask
from app.schemas.facial_profile import ImageSize
from app.schemas.makeup_component import MakeupComponent
from app.schemas.makeup_preview import ComponentRenderReport, PreviewResponse
from app.services.makeup_preview.color import srgb_to_lab
from app.services.makeup_preview.metrics import mean_delta_e_inside, outside_changed_count
from app.services.makeup_preview.renderer import render_layers

W, H = 96, 80
RED, BLUE, GREEN = "#e02020", "#2040e0", "#20c040"


def _png_bytes(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return buf.tobytes()


def _png_b64(arr: np.ndarray) -> str:
    return base64.b64encode(_png_bytes(arr)).decode()


def _ellipse(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    inside = ((xx - cx) / max(w / 2.0, 1)) ** 2 + ((yy - cy) / max(h / 2.0, 1)) ** 2 <= 1.0
    return inside.astype(np.uint8) * 255


def _image(seed: int = 7) -> np.ndarray:
    """Textured mid-range RGB image (mid-range keeps Lab->sRGB clipping out of the way)."""
    return np.random.default_rng(seed).integers(80, 180, size=(H, W, 3), dtype=np.uint8)


def _comp(
    index: int,
    rect: tuple[int, int, int, int],
    *,
    color: str | None = RED,
    finish: str | None = "matte",
    intensity: float | None = 0.8,
    stage: int | None = None,
    adjustments: tuple[Adjustment, ...] = (),
    mask: np.ndarray | None = None,
    category: str = "eye",
    product_type: str = "eyeshadow",
) -> AdaptedComponent:
    x, y, w, h = rect
    arr = _ellipse(h, w) if mask is None else mask
    component = MakeupComponent(
        category=category, product_type=product_type, color_hex=color, finish=finish, intensity=intensity
    )
    return AdaptedComponent(
        index=index,
        component=component,
        stage_number=stage,
        zone_selection=ZoneSelection(source="placement"),
        mask=ZoneMask(png_base64=_png_b64(arr)),
        bbox=ZoneBBox(x=x, y=y, width=w, height=h),
        centroid=(x + w / 2.0, y + h / 2.0),
        pixel_count=int((arr > 0).sum()),
        adjustments=list(adjustments),
    )


def _plan(*components: AdaptedComponent, w: int = W, h: int = H) -> AdaptedMakeupPlan:
    return AdaptedMakeupPlan(status="ok", image=ImageSize(width=w, height=h), components=list(components))


def _adj(multiplier: float | None = None, angles: tuple[DirectionAngle, ...] = ()) -> Adjustment:
    return Adjustment(
        rule_id="lift_downturned_eyes",
        reason="lift_downturned_eyes",
        effect="intensity",
        intensity_multiplier=multiplier,
        direction_angles=list(angles),
    )


# --- 1. Outside-mask pixels unchanged -------------------------------------------------

def test_pixels_outside_masks_are_unchanged():
    img = _image()
    plan = _plan(_comp(0, (20, 20, 30, 24), finish="glitter", intensity=1.0))
    res = render_layers(img, plan, seed=3)
    assert outside_changed_count(img, res.image, res.alpha_union) == 0
    zero = res.alpha_union == 0
    assert zero.any() and not zero.all()
    assert np.array_equal(res.image[zero], img[zero])


def test_inside_mask_pixels_do_change():
    img = _image()
    res = render_layers(img, _plan(_comp(0, (20, 20, 30, 24), intensity=1.0)), seed=0)
    assert np.count_nonzero(res.alpha_union > 0.5) > 0
    assert not np.array_equal(res.image, img)


# --- 2. Determinism and seed behaviour ------------------------------------------------

def test_same_seed_gives_identical_output():
    img = _image()
    plan = _plan(_comp(0, (20, 20, 30, 24), finish="shimmer"), _comp(1, (40, 30, 30, 24), finish="glitter"))
    a = render_layers(img, plan, seed=11)
    b = render_layers(img, plan, seed=11)
    assert np.array_equal(a.image, b.image)
    assert a.reports == b.reports


@pytest.mark.parametrize("finish", ["shimmer", "metallic", "glitter"])
def test_different_seed_changes_noise_finishes(finish):
    img = _image()
    plan = _plan(_comp(0, (20, 20, 30, 24), finish=finish, intensity=1.0))
    a = render_layers(img, plan, seed=1)
    b = render_layers(img, plan, seed=2)
    assert not np.array_equal(a.image, b.image)


@pytest.mark.parametrize("finish", ["matte", "satin", "natural", "velvet", "dewy", "glossy"])
def test_different_seed_does_not_change_deterministic_finishes(finish):
    img = _image()
    plan = _plan(_comp(0, (20, 20, 30, 24), finish=finish, intensity=1.0))
    a = render_layers(img, plan, seed=1)
    b = render_layers(img, plan, seed=2)
    assert np.array_equal(a.image, b.image)


# --- 3. Skip reasons ------------------------------------------------------------------

def _invalid_mask_components() -> dict[str, AdaptedComponent]:
    good = _ellipse(10, 12)
    return {
        "bad_base64": AdaptedComponent(
            index=0, component=MakeupComponent(category="eye", product_type="eyeshadow", color_hex=RED),
            zone_selection=ZoneSelection(source="placement"), mask=ZoneMask(png_base64="%%%not-base64%%%"),
            bbox=ZoneBBox(x=10, y=10, width=12, height=10), centroid=(16.0, 15.0), pixel_count=0,
        ),
        "dims_mismatch": _comp(0, (10, 10, 12, 10), mask=np.full((11, 12), 255, np.uint8)),
        "non_png": AdaptedComponent(
            index=0, component=MakeupComponent(category="eye", product_type="eyeshadow", color_hex=RED),
            zone_selection=ZoneSelection(source="placement"),
            mask=ZoneMask(png_base64=base64.b64encode(b"GIF89a plain text, not a png").decode()),
            bbox=ZoneBBox(x=10, y=10, width=12, height=10), centroid=(16.0, 15.0), pixel_count=0,
        ),
        "oversize": AdaptedComponent(
            index=0, component=MakeupComponent(category="eye", product_type="eyeshadow", color_hex=RED),
            zone_selection=ZoneSelection(source="placement"), mask=ZoneMask(png_base64="A" * 4000),
            bbox=ZoneBBox(x=10, y=10, width=12, height=10), centroid=(16.0, 15.0), pixel_count=0,
        ),
        "truncated_png": AdaptedComponent(
            index=0, component=MakeupComponent(category="eye", product_type="eyeshadow", color_hex=RED),
            zone_selection=ZoneSelection(source="placement"),
            mask=ZoneMask(png_base64=base64.b64encode(_png_bytes(good)[:30]).decode()),
            bbox=ZoneBBox(x=10, y=10, width=12, height=10), centroid=(16.0, 15.0), pixel_count=0,
        ),
        "zero_width_bbox": AdaptedComponent(
            index=0, component=MakeupComponent(category="eye", product_type="eyeshadow", color_hex=RED),
            zone_selection=ZoneSelection(source="placement"), mask=ZoneMask(png_base64="AAAA"),
            bbox=ZoneBBox(x=10, y=10, width=0, height=10), centroid=(10.0, 15.0), pixel_count=0,
        ),
    }


@pytest.mark.parametrize("case", sorted(_invalid_mask_components()))
def test_invalid_mask_is_skipped_and_others_render(case):
    img = _image()
    bad = _invalid_mask_components()[case]
    good = _comp(1, (40, 30, 30, 24), finish="matte", intensity=1.0)
    res = render_layers(img, _plan(bad, good), seed=0)
    by_index = {r.index: r for r in res.reports}
    assert by_index[0].status == "skipped" and by_index[0].reason == "mask_invalid", case
    assert by_index[0].pixels_touched == 0
    assert by_index[1].status == "rendered" and by_index[1].pixels_touched > 0
    assert outside_changed_count(img, res.image, res.alpha_union) == 0


def test_missing_mask_is_skipped_as_no_mask():
    img = _image()
    missing = AdaptedComponent(
        index=0, component=MakeupComponent(category="eye", product_type="eyeshadow", color_hex=RED),
        zone_selection=ZoneSelection(source="none"),
    )
    good = _comp(1, (40, 30, 30, 24), intensity=1.0)
    res = render_layers(img, _plan(missing, good), seed=0)
    by_index = {r.index: r for r in res.reports}
    assert by_index[0].status == "skipped" and by_index[0].reason == "no_mask"
    assert by_index[1].status == "rendered" and by_index[1].pixels_touched > 0


def test_missing_colour_is_skipped_and_others_render():
    img = _image()
    no_colour = _comp(0, (10, 10, 20, 16), color=None, intensity=1.0)
    good = _comp(1, (40, 30, 30, 24), intensity=1.0)
    res = render_layers(img, _plan(no_colour, good), seed=0)
    by_index = {r.index: r for r in res.reports}
    assert by_index[0].status == "skipped" and by_index[0].reason == "no_colour"
    assert by_index[1].status == "rendered" and by_index[1].pixels_touched > 0
    assert outside_changed_count(img, res.image, res.alpha_union) == 0


# --- 4. Opacity ----------------------------------------------------------------------

def test_zero_intensity_gives_identity_with_zero_opacity_reason():
    img = _image()
    res = render_layers(img, _plan(_comp(0, (20, 20, 30, 24), intensity=0.0)), seed=0)
    assert np.array_equal(res.image, img)
    assert np.count_nonzero(res.alpha_union) == 0
    (rep,) = res.reports
    assert rep.status == "skipped" and rep.reason == "zero_opacity" and rep.opacity == 0.0


def test_zero_multiplier_gives_identity_with_zero_opacity_reason():
    img = _image()
    comp = _comp(0, (20, 20, 30, 24), intensity=0.8, adjustments=(_adj(multiplier=0.0),))
    res = render_layers(img, _plan(comp), seed=0)
    assert np.array_equal(res.image, img)
    (rep,) = res.reports
    assert rep.reason == "zero_opacity" and rep.status == "skipped"


def test_larger_multiplier_gives_larger_mean_delta_e_inside_mask():
    img = _image()
    rect = (20, 20, 30, 24)
    mask = _ellipse(24, 30)
    full = np.zeros((H, W), np.uint8)
    full[20:44, 20:50] = mask
    deltas = []
    for mult in (0.2, 0.9):
        comp = _comp(0, rect, intensity=0.5, finish="matte", adjustments=(_adj(multiplier=mult),))
        res = render_layers(img, _plan(comp), seed=0)
        deltas.append(mean_delta_e_inside(img, res.image, full))
    assert deltas[0] > 0.0
    assert deltas[1] > deltas[0], deltas


# --- 5. Stage compositing -------------------------------------------------------------

def test_swapping_stage_numbers_changes_overlap_result():
    img = _image()
    rect = (20, 20, 30, 24)
    a_first = _plan(_comp(0, rect, color=RED, intensity=1.0, stage=1),
                    _comp(1, rect, color=BLUE, intensity=1.0, stage=2))
    a_swapped = _plan(_comp(0, rect, color=RED, intensity=1.0, stage=2),
                      _comp(1, rect, color=BLUE, intensity=1.0, stage=1))
    r1 = render_layers(img, a_first, seed=0)
    r2 = render_layers(img, a_swapped, seed=0)
    assert not np.array_equal(r1.image, r2.image)


def test_stage_none_renders_last():
    img = _image()
    rect = (20, 20, 30, 24)
    none_stage = _plan(_comp(0, rect, color=RED, intensity=1.0, stage=None),
                       _comp(1, rect, color=BLUE, intensity=1.0, stage=0))
    explicit_last = _plan(_comp(0, rect, color=RED, intensity=1.0, stage=5),
                          _comp(1, rect, color=BLUE, intensity=1.0, stage=0))
    r_none = render_layers(img, none_stage, seed=0)
    r_last = render_layers(img, explicit_last, seed=0)
    assert [r.index for r in r_none.reports] == [1, 0]
    assert np.array_equal(r_none.image, r_last.image)


# --- 6. Matte lightness ---------------------------------------------------------------

def test_matte_std_of_l_inside_mask_scales_as_expected():
    img = _image(seed=21)
    rect = (20, 20, 40, 30)
    full = np.full((30, 40), 255, np.uint8)
    opacity = 0.5
    res = render_layers(img, _plan(_comp(0, rect, finish="matte", intensity=opacity, mask=full)), seed=0)
    sel = np.zeros((H, W), bool)
    sel[20:50, 20:60] = True
    l_orig = srgb_to_lab(img[sel].astype(np.float64) / 255.0)[:, 0]
    l_new = srgb_to_lab(res.image[sel].astype(np.float64) / 255.0)[:, 0]
    ratio = float(np.std(l_new) / np.std(l_orig))
    expected = 1.0 - 0.3 * opacity
    assert abs(ratio - expected) < 0.05, (ratio, expected)


# --- 7. Clipping and input validation -------------------------------------------------

def test_bbox_partly_outside_image_is_clipped_without_error():
    img = _image()
    comp = _comp(0, (-5, -3, 20, 15), intensity=1.0)
    res = render_layers(img, _plan(comp), seed=0)
    assert res.image.shape == (H, W, 3)
    assert res.reports[0].status == "rendered" and res.reports[0].pixels_touched > 0
    assert outside_changed_count(img, res.image, res.alpha_union) == 0


@pytest.mark.parametrize(
    "bad_image",
    [
        pytest.param(np.zeros((W, H, 3), np.uint8), id="transposed_shape"),
        pytest.param(np.zeros((H + 1, W, 3), np.uint8), id="wrong_height"),
        pytest.param(np.zeros((H, W, 3), np.float32), id="float32"),
        pytest.param(np.zeros((H, W, 3), np.int16), id="int16"),
        pytest.param(np.zeros((H, W, 4), np.uint8), id="rgba"),
        pytest.param(np.zeros((H, W, 1), np.uint8), id="single_channel_3d"),
        pytest.param(np.zeros((H, W), np.uint8), id="2d_grayscale"),
        pytest.param([[[0, 0, 0]]], id="list_input"),
    ],
)
def test_invalid_image_inputs_raise_value_error(bad_image):
    plan = _plan(_comp(0, (20, 20, 30, 24)))
    with pytest.raises(ValueError):
        render_layers(bad_image, plan, seed=0)


def test_empty_plan_returns_copy_and_no_reports():
    img = _image()
    res = render_layers(img, _plan(), seed=0)
    assert res.reports == ()
    assert np.array_equal(res.image, img)
    assert res.image is not img


# --- 9. Reports -----------------------------------------------------------------------

def test_one_report_per_component_including_skipped():
    img = _image()
    plan = _plan(
        _comp(0, (10, 10, 20, 16), color=None),
        _comp(1, (30, 30, 20, 16)),
        _comp(2, (50, 40, 20, 16), intensity=0.0),
    )
    res = render_layers(img, plan, seed=0)
    assert sorted(r.index for r in res.reports) == [0, 1, 2]


def test_directional_stretch_deferred_when_direction_angles_present():
    angle = DirectionAngle(side="left", face_deg=30.0, image_deg=-20.0)
    img = _image()
    res = render_layers(img, _plan(_comp(0, (20, 20, 30, 24), adjustments=(_adj(angles=(angle,)),))), seed=0)
    assert res.reports[0].directional_stretch == "deferred"


def test_directional_stretch_not_applicable_without_direction_angles():
    img = _image()
    res = render_layers(img, _plan(_comp(0, (20, 20, 30, 24), adjustments=(_adj(),))), seed=0)
    assert res.reports[0].directional_stretch == "not_applicable"


def test_directional_stretch_never_applied():
    angle = DirectionAngle(side="right", face_deg=-45.0, image_deg=45.0)
    img = _image()
    plan = _plan(
        _comp(0, (10, 10, 20, 16), adjustments=(_adj(angles=(angle,)),)),
        _comp(1, (40, 40, 20, 16)),
        _comp(2, (10, 40, 20, 16), intensity=0.0, adjustments=(_adj(angles=(angle,)),)),
    )
    res = render_layers(img, plan, seed=0)
    assert all(r.directional_stretch != "applied" for r in res.reports)


def test_rendered_components_report_pixels_touched_positive():
    img = _image()
    plan = _plan(_comp(0, (20, 20, 30, 24), intensity=0.6), _comp(1, (50, 40, 24, 20), finish="glitter"))
    res = render_layers(img, plan, seed=0)
    rendered = [r for r in res.reports if r.status == "rendered"]
    assert len(rendered) == 2
    assert all(r.pixels_touched > 0 for r in rendered)


def test_component_fully_off_image_is_not_reported_as_rendered_with_zero_pixels():
    img = _image()
    res = render_layers(img, _plan(_comp(0, (W + 60, 20, 30, 24), intensity=1.0)), seed=0)
    (rep,) = res.reports
    assert rep.status == "skipped" or rep.pixels_touched > 0, rep


def test_empty_mask_is_not_reported_as_rendered_with_zero_pixels():
    img = _image()
    empty = np.zeros((24, 30), np.uint8)
    res = render_layers(img, _plan(_comp(0, (20, 20, 30, 24), mask=empty, intensity=1.0)), seed=0)
    (rep,) = res.reports
    assert rep.status == "skipped" or rep.pixels_touched > 0, rep


def test_finish_none_defaults_to_natural_and_reports_finish():
    img = _image()
    res = render_layers(img, _plan(_comp(0, (20, 20, 30, 24), finish=None)), seed=0)
    assert res.reports[0].finish_applied == "natural"


def test_preview_report_schema_rejects_out_of_range_values():
    with pytest.raises(ValidationError):
        ComponentRenderReport(
            index=32, status="rendered", opacity=0.5, finish_applied="matte",
            directional_stretch="not_applicable", pixels_touched=1,
        )
    with pytest.raises(ValidationError):
        ComponentRenderReport(
            index=0, status="rendered", opacity=1.5, finish_applied="matte",
            directional_stretch="not_applicable", pixels_touched=1,
        )
    with pytest.raises(ValidationError):
        PreviewResponse(image_png_base64="", width=0, height=10, plan_status="ok")
