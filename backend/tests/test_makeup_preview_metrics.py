"""Phase 9a-1 metrics and colour maths: CIEDE2000 reference pairs and the metrics helpers."""

import numpy as np
import pytest

from app.services.makeup_preview.color import delta_e_2000, srgb_to_lab
from app.services.makeup_preview.metrics import delta_e_to_target, mean_delta_e_inside, outside_changed_count

# Subset of the Sharma, Wu & Dalal (2005) CIEDE2000 test pairs. Ids are local labels, not the
# paper's row numbers. Values were cross-checked against an independent CIEDE2000 implementation
# (scikit-image). Hue-wraparound cases (|h1 - h2| > 180 deg, needs the +/-360 correction) are
# local ids 23 and 25. Pairs 9-12 sit on the hue ~0/~180 boundary; 1-6 are blue-hue cases.
SHARMA_PAIRS = [
    (1, (50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    (2, (50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
    (3, (50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
    (4, (50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485), 1.0000),
    (5, (50.0, -1.1848, -84.8006), (50.0, 0.0, -82.7485), 1.0000),
    (6, (50.0, -0.9009, -85.5211), (50.0, 0.0, -82.7485), 1.0000),
    (7, (50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
    (8, (50.0, -1.0, 2.0), (50.0, 0.0, 0.0), 2.3669),
    (9, (50.0, 2.4900, -0.0010), (50.0, -2.4900, 0.0009), 7.1792),
    (10, (50.0, 2.4900, -0.0010), (50.0, -2.4900, 0.0010), 7.1792),
    (11, (50.0, 2.4900, -0.0010), (50.0, -2.4900, 0.0011), 7.2195),
    (12, (50.0, 2.4900, -0.0010), (50.0, -2.4900, 0.0012), 7.2195),
    (21, (50.0, -0.0010, 2.4900), (50.0, 0.0009, -2.4900), 4.8045),
    (22, (50.0, 2.5000, 0.0000), (50.0, 0.0000, -2.5000), 4.3065),
    (23, (50.0, 2.5000, 0.0000), (73.0, 25.0, -18.0), 27.1492),
    (24, (50.0, 2.5000, 0.0000), (61.0, -5.0, 29.0), 22.8977),
    (25, (50.0, 2.5000, 0.0000), (56.0, -27.0, -3.0), 31.9030),
    (26, (50.0, 2.5000, 0.0000), (58.0, 24.0, 15.0), 19.4535),
    (27, (50.0, 2.5000, 0.0000), (50.0, 3.1736, 0.5854), 1.0000),
    (28, (50.0, 2.5000, 0.0000), (50.0, 3.2972, 0.0000), 1.0000),
    (29, (50.0, 2.5000, 0.0000), (50.0, 1.8634, 0.5757), 1.0000),
    (30, (50.0, 2.5000, 0.0000), (50.0, 3.2592, 0.3350), 1.0000),
    (31, (60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    (32, (63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
]


def test_reference_pair_count_meets_minimum():
    assert len(SHARMA_PAIRS) >= 10


@pytest.mark.parametrize("pair_id,lab1,lab2,expected", SHARMA_PAIRS, ids=[str(p[0]) for p in SHARMA_PAIRS])
def test_delta_e_2000_matches_sharma_reference(pair_id, lab1, lab2, expected):
    got = float(delta_e_2000(np.array(lab1), np.array(lab2)))
    assert abs(got - expected) < 1e-3, (pair_id, got, expected)


def test_delta_e_2000_is_symmetric_on_reference_pairs():
    for _, lab1, lab2, _ in SHARMA_PAIRS:
        a = float(delta_e_2000(np.array(lab1), np.array(lab2)))
        b = float(delta_e_2000(np.array(lab2), np.array(lab1)))
        assert abs(a - b) < 1e-9


def test_delta_e_2000_zero_for_identical_and_broadcasts():
    lab = np.array([[50.0, 10.0, -20.0], [70.0, -5.0, 30.0]])
    assert np.allclose(delta_e_2000(lab, lab), 0.0)
    single = np.array([50.0, 2.5, 0.0])
    out = delta_e_2000(lab, single)
    assert out.shape == (2,)
    assert np.all(out > 0)


def test_srgb_to_lab_white_and_black():
    white = srgb_to_lab(np.array([[255, 255, 255]], dtype=np.uint8))[0]
    black = srgb_to_lab(np.array([[0, 0, 0]], dtype=np.uint8))[0]
    assert abs(white[0] - 100.0) < 0.05 and abs(white[1]) < 0.05 and abs(white[2]) < 0.05
    assert abs(black[0]) < 1e-3


# --- outside_changed_count -------------------------------------------------------------

def test_outside_changed_count_zero_when_identical():
    img = np.random.default_rng(1).integers(0, 256, (8, 9, 3), dtype=np.uint8)
    assert outside_changed_count(img, img.copy(), np.zeros((8, 9), np.float32)) == 0


def test_outside_changed_count_ignores_changes_under_alpha():
    img = np.zeros((4, 4, 3), np.uint8)
    rendered = img.copy()
    rendered[0, 0, 1] = 5
    rendered[3, 3, 2] = 9
    alpha = np.zeros((4, 4), np.float32)
    alpha[0, 0] = 0.7
    assert outside_changed_count(img, rendered, alpha) == 1


def test_outside_changed_count_counts_each_changed_pixel_once():
    img = np.zeros((2, 2, 3), np.uint8)
    rendered = np.full((2, 2, 3), 255, np.uint8)
    assert outside_changed_count(img, rendered, np.zeros((2, 2), np.float32)) == 4


# --- mean_delta_e_inside / delta_e_to_target -------------------------------------------

def test_mean_delta_e_inside_zero_for_identical_and_empty_mask():
    img = np.random.default_rng(2).integers(0, 256, (6, 6, 3), dtype=np.uint8)
    mask = np.full((6, 6), 255, np.uint8)
    assert mean_delta_e_inside(img, img.copy(), mask) == pytest.approx(0.0, abs=1e-9)
    assert mean_delta_e_inside(img, np.zeros_like(img), np.zeros((6, 6), np.uint8)) == 0.0


def test_mean_delta_e_inside_positive_for_changed_region_only():
    img = np.full((6, 6, 3), 120, np.uint8)
    rendered = img.copy()
    rendered[:3, :, :] = 200
    mask = np.zeros((6, 6), np.uint8)
    mask[:3, :] = 255
    assert mean_delta_e_inside(img, rendered, mask) > 1.0
    mask_outside = np.zeros((6, 6), np.uint8)
    mask_outside[4:, :] = 255
    assert mean_delta_e_inside(img, rendered, mask_outside) == pytest.approx(0.0, abs=1e-9)


def test_delta_e_to_target_drops_when_moving_toward_target():
    target_rgb = np.array([30, 60, 200], np.uint8)
    target_lab = srgb_to_lab(target_rgb.astype(np.float64) / 255.0)
    original = np.full((5, 5, 3), 200, np.uint8)
    moved = np.full((5, 5, 3), 80, np.uint8)
    moved[..., 0], moved[..., 1], moved[..., 2] = 40, 70, 190
    mask = np.full((5, 5), 255, np.uint8)
    before = delta_e_to_target(original, mask, target_lab)
    after = delta_e_to_target(moved, mask, target_lab)
    assert after < before


def test_delta_e_to_target_zero_for_empty_mask():
    img = np.full((3, 3, 3), 90, np.uint8)
    assert delta_e_to_target(img, np.zeros((3, 3), np.uint8), (50.0, 0.0, 0.0)) == 0.0
