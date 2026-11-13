import math

import cv2
import numpy as np

from app.schemas.face_segmentation import Zone, ZoneBBox, ZoneMask
from app.schemas.facial_profile import Classified, FeaturesData, SkinToneData
from app.services.facial_features import thresholds as t
from app.services.facial_features.skin_tone import classify_skin_tone
from app.services.makeup_blueprint import cv_analysis as cv
from tests.test_makeup_cv_analysis import RECTS, H, W, _image, _profile, _zone

LABELS = ("dark", "brown", "tan", "intermediate", "light", "very_light")
GRAY = (128, 128, 128)


def _lab_rgb(l_, a, b):
    out = cv2.cvtColor(np.array([[[l_, a, b]]], dtype=np.float32), cv2.COLOR_LAB2RGB)[0, 0]
    return tuple(int(round(float(np.clip(v, 0.0, 1.0)) * 255)) for v in out)


def _zones(rects=None):
    return _profile(rects).components.zones.data.zones


def _patch(l_, a, b, bg=GRAY):
    img = np.empty((H, W, 3), dtype=np.uint8)
    img[:] = bg
    x, y, w, h = RECTS["face_skin"]
    img[y : y + h, x : x + w] = _lab_rgb(l_, a, b)
    return img


def _ita_patch(ita, b=12.0, a=8.0, bg=GRAY):
    return _patch(50.0 + b * math.tan(math.radians(ita)), a, b, bg)


def _hue_patch(hue, chroma=20.0, l_=65.0, bg=GRAY):
    return _patch(l_, chroma * math.cos(math.radians(hue)), chroma * math.sin(math.radians(hue)), bg)


def _classify(img, glasses=False, rects=None):
    return classify_skin_tone(img, _zones(rects), glasses)


def test_every_depth_band_reachable():
    seen = {_classify(_ita_patch(i)).depth.label for i in range(-50, 71, 5)}
    assert set(LABELS) <= seen


def test_monotonic_in_l_star():
    idx = []
    for l_ in range(30, 86, 5):
        label = _classify(_patch(float(l_), 8.0, 12.0)).depth.label
        if label != "unknown":
            idx.append(LABELS.index(label))
    assert idx == sorted(idx) and len(set(idx)) > 3


def test_edge_within_margin_is_unknown():
    for edge in t.SKIN_ITA_EDGES[:-1]:
        res = _classify(_ita_patch(edge, b=25.0, a=8.0))
        measured = res.depth.measurements["ita_deg"]
        assert abs(measured - edge) < 3.0
        if abs(measured - edge) < t.SKIN_ITA_MARGIN:
            assert res.depth.label == "unknown"
            assert res.depth.reason == "within_ambiguity_margin"
        else:
            assert res.depth.label in LABELS
    res = _classify(_ita_patch(41.0, b=25.0, a=8.0))
    assert res.depth.label == "unknown" or abs(res.depth.measurements["ita_deg"] - 41.0) >= t.SKIN_ITA_MARGIN


def test_undertone_per_hue_band():
    cool = _classify(_hue_patch(t.SKIN_HUE_COOL_BELOW - 8)).undertone
    neutral = _classify(_hue_patch((t.SKIN_HUE_COOL_BELOW + t.SKIN_HUE_WARM_ABOVE) / 2)).undertone
    warm = _classify(_hue_patch((t.SKIN_HUE_WARM_ABOVE + t.SKIN_HUE_OLIVE_ABOVE) / 2)).undertone
    olive = _classify(_hue_patch(t.SKIN_HUE_OLIVE_ABOVE + 8)).undertone
    assert (cool.label, neutral.label, warm.label, olive.label) == ("cool", "neutral", "warm", "olive")


def test_undertone_confidence_below_depth_confidence():
    res = _classify(_hue_patch(20.0))
    assert res.depth.label != "unknown" and res.undertone.label == "cool"
    assert 0.0 < res.undertone.confidence < res.depth.confidence <= t.SKIN_DEPTH_CONF_CAP
    assert res.undertone.confidence <= t.SKIN_UNDERTONE_CONF_CAP


def test_color_cast_poor_makes_undertone_unknown():
    bg = _lab_rgb(70.0, 0.0, t.SKIN_CAST_CHROMA_POOR + 15.0)
    res = _classify(_hue_patch(20.0, bg=bg))
    assert "color_cast" in res.lighting_reasons
    assert res.lighting_quality == "poor"
    assert res.undertone.label == "unknown"


def test_color_cast_caution_lowers_confidence():
    base = _classify(_hue_patch(20.0))
    bg = _lab_rgb(70.0, 0.0, (t.SKIN_CAST_CHROMA_CAUTION + t.SKIN_CAST_CHROMA_POOR) / 2)
    res = _classify(_hue_patch(20.0, bg=bg))
    assert "color_cast" in res.lighting_reasons
    assert res.lighting_quality == "caution"
    assert res.undertone.confidence < base.undertone.confidence
    assert res.depth.confidence < base.depth.confidence


def test_glasses_lowers_confidence():
    base = _classify(_hue_patch(20.0))
    res = _classify(_hue_patch(20.0), glasses=True)
    assert "glasses" in res.lighting_reasons
    assert res.depth.confidence < base.depth.confidence


def _clipped_patch(l_, a, b, value, frac=0.3):
    img = _patch(l_, a, b)
    x, y, w, h = RECTS["face_skin"]
    n = int(w * frac)
    img[y : y + h, x : x + n] = value
    return img


def test_deep_skin_well_exposed_is_dark_not_unknown():
    res = _classify(_patch(30.0, 8.0, 12.0))
    assert res.depth.label == "dark"
    assert "too_dark" not in res.lighting_reasons and "overexposed" not in res.lighting_reasons
    assert res.depth.measurements["crushed_frac"] == 0.0 and res.depth.measurements["blown_frac"] == 0.0


def test_crushed_and_blown_clipping():
    dark = _classify(_clipped_patch(30.0, 8.0, 12.0, (0, 0, 0)))
    assert "too_dark" in dark.lighting_reasons
    assert dark.depth.label == "unknown" and dark.lighting_quality == "poor"
    assert dark.depth.measurements["crushed_frac"] > t.SKIN_CLIP_POOR
    bright = _classify(_clipped_patch(70.0, 8.0, 12.0, (255, 255, 255)))
    assert "overexposed" in bright.lighting_reasons
    assert bright.depth.label == "unknown" and bright.lighting_quality == "poor"
    assert bright.depth.measurements["blown_frac"] > t.SKIN_CLIP_POOR


def test_mild_clipping_lowers_confidence_only():
    base = _classify(_patch(30.0, 8.0, 12.0))
    frac = (t.SKIN_CLIP_CAUTION + t.SKIN_CLIP_POOR) / 2
    res = _classify(_clipped_patch(30.0, 8.0, 12.0, (0, 0, 0), frac))
    assert "too_dark" in res.lighting_reasons
    assert res.depth.label == "dark"
    assert res.depth.confidence < base.depth.confidence


def test_low_chroma_olive_uses_olive_hue_constant():
    hue = t.SKIN_OLIVE_LOW_CHROMA_HUE + 4.0
    res = _classify(_hue_patch(hue, chroma=t.SKIN_OLIVE_LOW_CHROMA - 3.0))
    assert res.undertone.label == "olive"
    low = _classify(_hue_patch(t.SKIN_OLIVE_LOW_CHROMA_HUE - 6.0, chroma=t.SKIN_OLIVE_LOW_CHROMA - 3.0))
    assert low.undertone.label != "olive"


def test_classify_logs_nothing(caplog):
    with caplog.at_level("DEBUG"):
        _classify(_patch(30.0, 8.0, 12.0))
        _classify(_clipped_patch(30.0, 8.0, 12.0, (0, 0, 0)))
    assert not [r for r in caplog.records if r.name.startswith("app.services.facial_features")]


def test_insufficient_inputs_do_not_raise():
    img = _ita_patch(40.0)
    tiny = {"face_skin": (20, 20, 30, 30)}
    for res in (
        _classify(img, rects=tiny),
        _classify(img, rects={"lip_full": RECTS["lip_full"]}),
        classify_skin_tone(img, []),
    ):
        assert res.depth.label == "unknown" and res.undertone.label == "unknown"
        assert res.depth.reason == "insufficient_skin"
        assert res.lighting_reasons == ["insufficient_skin"]


def test_corrupt_mask_and_bad_image_do_not_raise():
    bad = Zone(
        id="face_skin", name="face_skin", group="g", side=None, sources=["landmarks"], status="ok",
        confidence=0.9, clipped_to=[], bbox=ZoneBBox(x=20, y=20, width=260, height=260),
        pixel_count=1, mask=ZoneMask(png_base64="AAAA"),
    )
    res = classify_skin_tone(_ita_patch(40.0), [bad])
    assert res.depth.label == "unknown" and res.undertone.label == "unknown"
    res = classify_skin_tone(np.zeros((10, 10), dtype=np.float32), _zones())
    assert res.depth.label == "unknown" and res.depth.reason == "degenerate_input"


def test_all_values_finite():
    for i in range(-50, 71, 10):
        res = _classify(_ita_patch(i))
        for c in (res.depth, res.undertone):
            assert math.isfinite(c.confidence) and 0.0 <= c.confidence <= 1.0
            assert all(math.isfinite(v) for v in c.measurements.values())
    keys = set(_classify(_ita_patch(40.0)).depth.measurements)
    assert {"l_star", "a_star", "b_star", "ita_deg", "hue_deg", "chroma", "cast_chroma", "skin_px"} <= keys


def _features(skin=None):
    def c(label):
        return Classified(label=label, confidence=0.5)

    return FeaturesData(
        eye_spacing=c("close"), eye_tilt=c("neutral"), eye_shape=c("almond"), lip_fullness=c("thin"),
        lip_balance=c("balanced"), lip_width=c("average"), brow_arch=c("straight"),
        brow_thickness=c("thin"), skin_tone=skin,
    )


def test_schema_round_trip_and_omission():
    res = _classify(_hue_patch(20.0))
    assert SkinToneData.model_validate(res.model_dump()) == res
    assert "skin_tone" not in _features().model_dump()
    assert "skin_tone" in _features(res).model_dump()


def test_wrapper_aliases_are_private_functions():
    assert cv.decode_zone_mask is cv._crop
    assert cv.trimmed_median_hex is cv._trimmed_hex
    assert cv.full_zone_mask is cv._full


def test_bare_skin_reference_matches_phase7_logic():
    img = _image()
    zones = _zones()
    by_id = {z.id: z for z in zones}

    def crop_of(zid):
        return cv._crop(img.shape, by_id[zid]) if zid in by_id else None

    mask, fallback = cv.bare_skin_reference(img, crop_of)
    expected = np.zeros(img.shape[:2], dtype=bool)
    x, y, w, h = RECTS["face_skin"]
    expected[y : y + h, x : x + w] = True
    for zid in {z for r in cv.TARGETS for z in r.zones} | set(cv.REFERENCE_EXCLUDED_ZONES):
        if zid in RECTS:
            zx, zy, zw, zh = RECTS[zid]
            expected[zy : zy + zh, zx : zx + zw] = False
    assert not fallback and np.array_equal(mask, expected)

    tiny = {"face_skin": _zone("face_skin", (20, 20, 30, 30))}
    mask, fallback = cv.bare_skin_reference(img, lambda zid: cv._crop(img.shape, tiny[zid]) if zid in tiny else None)
    assert fallback and int(mask.sum()) == 900
    assert cv.bare_skin_reference(img, lambda zid: None) is None
