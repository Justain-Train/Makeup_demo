"""Face parser: class mapping, crop geometry (stub model), weights verification, real model."""

import threading

import numpy as np
import pytest

from app.config import get_settings
from app.services.image_loading import load_rgb_image
from app.services.segmentation import masks as mk
from app.services.segmentation.sides import assign_sides
from app.services.segmentation.face_parser import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    FaceParserBusyError,
    FaceParserUnavailableError,
    OnnxFaceParser,
    ParserInfo,
    compute_parse_window,
    covered_region,
)
from app.services.segmentation.regions import (
    MODEL_ID_TO_CANONICAL,
    NOT_PARSED,
    NUM_MODEL_CLASSES,
    VERIFIED_MODEL_IDS,
    Region,
    Unsided,
    build_label_lut,
)
from tests.synthetic import FIXTURES

SETTINGS = get_settings()
HAVE_WEIGHTS = SETTINGS.resolved_face_parser_path.is_file()
needs_weights = pytest.mark.skipif(not HAVE_WEIGHTS, reason="face parser weights not downloaded")


# --- mapping (regression for the verified class order and left/right) ---------

def test_mapping_covers_all_19_classes_exactly_once():
    assert sorted(MODEL_ID_TO_CANONICAL) == list(range(NUM_MODEL_CLASSES)) == list(range(19))
    # only the paired classes (brow 2/3, eye 4/5, ear 7/8) are merged, into side-agnostic ids
    assert len(set(MODEL_ID_TO_CANONICAL.values())) == 16
    for a, b in ((2, 3), (4, 5), (7, 8)):
        assert MODEL_ID_TO_CANONICAL[a] is MODEL_ID_TO_CANONICAL[b]
    lut = build_label_lut()
    assert len(lut) == 256 and lut[200] == NOT_PARSED
    assert all(int(r) != NOT_PARSED for r in Region)


def test_mapping_paired_classes_are_side_agnostic():
    # The model's l_*/r_* split is not trusted (it failed on a mirrored photo); sides come
    # from landmarks in sides.assign_sides. Raw class order (kept as a regression check):
    # 2/3 brows, 4/5 eyes, 7/8 ears.
    assert MODEL_ID_TO_CANONICAL[4] is MODEL_ID_TO_CANONICAL[5] is Unsided.EYE
    assert MODEL_ID_TO_CANONICAL[2] is MODEL_ID_TO_CANONICAL[3] is Unsided.BROW
    assert MODEL_ID_TO_CANONICAL[7] is MODEL_ID_TO_CANONICAL[8] is Unsided.EAR
    assert MODEL_ID_TO_CANONICAL[10] is Region.NOSE and MODEL_ID_TO_CANONICAL[12] is Region.LIP_UPPER
    # no sided id may come straight out of the model mapping
    sided = {Region.BROW_L, Region.BROW_R, Region.EYE_L, Region.EYE_R, Region.EAR_L, Region.EAR_R}
    assert not sided & set(MODEL_ID_TO_CANONICAL.values())


def test_unsided_ids_never_collide_with_regions():
    assert {int(u) for u in Unsided}.isdisjoint({int(r) for r in Region}) and NOT_PARSED not in {int(u) for u in Unsided}


def test_verified_ids():
    # earring (9) and necklace (15) were never seen on a test photo; glasses (6) and hat (18) were.
    assert 9 not in VERIFIED_MODEL_IDS and 15 not in VERIFIED_MODEL_IDS
    assert {6, 18} <= VERIFIED_MODEL_IDS


# --- window geometry -------------------------------------------------------------

def test_parse_window_is_square_centered_and_scaled():
    w = compute_parse_window((100, 200, 300, 500), 2.0)  # bbox 200x300 -> side 600
    assert w.size == 600 and (w.x, w.y) == (200 - 300, 350 - 300)
    assert covered_region(w, 1000, 1000) == (0, 50, 500, 650)


# --- stub model: crop + inverse mapping must be exact (within 1 px) -------------

class BrightnessStubParser(OnnxFaceParser):
    """Stands in for the network: model class 1 where the pixel is bright, else 0."""

    def __init__(self, crop_scale: float = 2.0, **kw):
        self.info = ParserInfo("stub", "stub", "0" * 64, 512)
        self._init_pipeline(crop_scale, kw.get("max_concurrency", 1), kw.get("queue_timeout_s", 1.0))
        self.gate: threading.Event | None = None
        self.entered = threading.Event()

    def _infer(self, x: np.ndarray) -> np.ndarray:
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(5)
        pixels = x[0].transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
        bright = pixels.mean(axis=2) > 128
        out = np.zeros((19, 512, 512), dtype=np.float32)
        out[1] = np.where(bright, 10.0, 0.0)
        out[0] = np.where(bright, 0.0, 10.0)
        return out


def _rect_image(w: int, h: int, rect: tuple[int, int, int, int]) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    x0, y0, x1, y1 = rect
    img[y0:y1, x0:x1] = 255
    return img


@pytest.mark.parametrize(
    "w,h,rect,bbox",
    [
        (640, 360, (300, 100, 400, 200), (250, 50, 450, 250)),  # non-square, window sticks out
        (3000, 2000, (1500, 700, 2100, 1200), (1400, 600, 2200, 1300)),  # > 1024 window: downscaled path
        (800, 800, (10, 10, 120, 100), (0, 0, 130, 110)),  # face at the image corner
    ],
)
def test_crop_and_inverse_map_align_with_the_image_within_1px(w, h, rect, bbox):
    img = _rect_image(w, h, rect)
    res = BrightnessStubParser().parse(img, bbox)
    full = np.full((h, w), NOT_PARSED, dtype=np.uint8)
    oy, ox = res.origin[1], res.origin[0]
    full[oy : oy + res.labels.shape[0], ox : ox + res.labels.shape[1]] = res.labels
    xs = np.nonzero((full == int(Region.SKIN)).any(axis=0))[0]
    ys = np.nonzero((full == int(Region.SKIN)).any(axis=1))[0]
    x0, y0, x1, y1 = rect
    assert abs(xs.min() - x0) <= 1 and abs(xs.max() - (x1 - 1)) <= 1
    assert abs(ys.min() - y0) <= 1 and abs(ys.max() - (y1 - 1)) <= 1
    # label map covers exactly window ∩ image
    cx0, cy0, cx1, cy1 = covered_region(res.window, w, h)
    assert res.labels.shape == (cy1 - cy0, cx1 - cx0) and res.origin == (cx0, cy0)
    assert set(np.unique(res.labels)) <= {int(Region.SKIN), int(Region.BACKGROUND)}
    assert res.confidence.shape == res.labels.shape


def test_busy_when_all_slots_are_taken():
    parser = BrightnessStubParser(max_concurrency=1, queue_timeout_s=0.05)
    parser.gate = threading.Event()
    img = _rect_image(200, 200, (80, 80, 120, 120))
    t = threading.Thread(target=lambda: parser.parse(img, (60, 60, 140, 140)))
    t.start()
    assert parser.entered.wait(5)
    with pytest.raises(FaceParserBusyError):
        parser.parse(img, (60, 60, 140, 140))
    parser.gate.set()
    t.join(5)
    parser.parse(img, (60, 60, 140, 140))  # slot was released


def test_png_round_trip_of_label_map_and_mask():
    labels = np.random.RandomState(0).randint(0, 19, (37, 53)).astype(np.uint8)
    assert np.array_equal(mk.decode_png(mk.encode_png(labels)), labels)
    mask = np.random.RandomState(1).rand(20, 30) > 0.5
    decoded = mk.decode_png(mk.encode_png(mask))
    assert set(np.unique(decoded)) <= {0, 255} and np.array_equal(decoded == 255, mask)


# --- weights verification --------------------------------------------------------

def test_missing_weights_are_reported_actionably(tmp_path):
    with pytest.raises(FaceParserUnavailableError, match="download_face_parser"):
        OnnxFaceParser(tmp_path / "nope.onnx", "00")


def test_corrupt_or_wrong_hash_weights_are_rejected(tmp_path):
    bad = tmp_path / "w.onnx"
    bad.write_bytes(b"not an onnx file")
    with pytest.raises(FaceParserUnavailableError, match="SHA-256"):
        OnnxFaceParser(bad, "ab" * 32)
    import hashlib

    with pytest.raises(FaceParserUnavailableError, match="could not be loaded"):
        OnnxFaceParser(bad, hashlib.sha256(b"not an onnx file").hexdigest())
    with pytest.raises(FaceParserUnavailableError):
        OnnxFaceParser(bad, "")  # unpinned hash is refused


# --- real model ------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_parser():
    return OnnxFaceParser(
        SETTINGS.resolved_face_parser_path, SETTINGS.face_parser_sha256, SETTINGS.face_parser_crop_scale
    )


@needs_weights
def test_real_model_metadata_and_channels(real_parser):
    inp = real_parser._session.get_inputs()[0]
    assert list(inp.shape[1:]) == [3, 512, 512] and inp.type == "tensor(float)"
    out = real_parser._infer(np.zeros((1, 3, 512, 512), dtype=np.float32))
    assert out.shape == (19, 512, 512)  # logits, 19 channels
    assert real_parser.info.weights_sha256 == SETTINGS.face_parser_sha256


@needs_weights
def test_real_model_left_right_and_alignment_on_fixture(real_parser):
    from app.services.face_regions import alignment_check
    from tests.synthetic import fixture_landmarks

    pts, w, h = fixture_landmarks()
    img = load_rgb_image((FIXTURES / "portrait.jpg").read_bytes())
    assert img.shape[:2] == (h, w)
    bbox = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    res = real_parser.parse(img, bbox)

    labels = assign_sides(res.labels, pts, res.origin)

    def label_at(i: int) -> int:
        x, y = np.round(pts[i]).astype(int) - res.origin
        return int(labels[y, x])

    assert int(Unsided.EYE) in {int(res.labels[tuple(np.round(pts[i]).astype(int)[::-1] - res.origin[::-1])]) for i in (468, 473)}
    assert label_at(468) == int(Region.EYE_R)  # subject's right iris (image left)
    assert label_at(473) == int(Region.EYE_L)  # subject's left iris (image right)
    assert alignment_check(pts, labels, res.origin).passed
    assert not np.isin(labels, [int(u) for u in Unsided]).any()
    assert (res.labels != NOT_PARSED).all()


# --- registration: sub-pixel centre-consistency (WP4a regression) ----------------

class EchoStubParser(OnnxFaceParser):
    """Network stand-in that echoes the (normalised) input.

    Model class 1 (skin) gets the input's channel 0 as its logit, class 0 gets
    the normalised mid-grey, so the label map is "pixel brighter than 127.5".
    `last_input` keeps the tensor the network would have seen.
    """

    MID = (127.5 - float(IMAGENET_MEAN[0])) / float(IMAGENET_STD[0])

    def __init__(self, crop_scale: float = 2.0):
        self.info = ParserInfo("stub", "stub", "0" * 64, 512)
        self._init_pipeline(crop_scale, 1, 1.0)
        self.last_input: np.ndarray | None = None

    def _infer(self, x: np.ndarray) -> np.ndarray:
        self.last_input = x[0, 0].copy()
        out = np.zeros((19, 512, 512), dtype=np.float32)
        out[0] = self.MID
        out[1] = x[0, 0]
        return out


def _edge_image(w: int, h: int, edge: float, axis: int, sigma: float) -> np.ndarray:
    """Grey image with a smooth logistic edge centred at pixel index `edge` along `axis` (1 = x, 0 = y)."""
    idx = np.arange(w if axis == 1 else h, dtype=np.float64)
    profile = 255.0 / (1.0 + np.exp(-(idx - edge) / sigma))
    plane = np.tile(profile[None, :], (h, 1)) if axis == 1 else np.tile(profile[:, None], (1, w))
    return np.repeat(np.round(plane).astype(np.uint8)[:, :, None], 3, axis=2)


def _line_crossing(profile: np.ndarray, level: float, near: float, half_width: float) -> float:
    """Sub-pixel index where `profile` crosses `level`: least-squares line through the transition samples."""
    idx = np.arange(profile.size)
    span = np.abs(idx - near) <= half_width
    lo, hi = level - 0.3 * abs(level - profile[span].min()), level + 0.3 * abs(profile[span].max() - level)
    sel = span & (profile >= lo) & (profile <= hi)
    slope, icpt = np.polyfit(idx[sel], profile[sel], 1)
    return float((level - icpt) / slope)


def _measure_registration(w: int, h: int, bbox, edge: float, axis: int) -> tuple[float, int]:
    """(edge shift in the network input, in original px; label-map edge error in px; both signed, + = right/down)."""
    parser = EchoStubParser()
    window = compute_parse_window(bbox, 2.0)
    sigma = window.size / 60.0
    res = parser.parse(_edge_image(w, h, edge, axis, sigma), bbox)
    k_in = 512 / window.size
    scan = parser.last_input[256] if axis == 1 else parser.last_input[:, 256]
    near_in = (edge - (window.x if axis == 1 else window.y) + 0.5) * k_in - 0.5
    crossing = _line_crossing(scan, EchoStubParser.MID, near_in, 8 * sigma * k_in + 4)
    # centre-aligned: input index j <-> window coordinate (j + 0.5) / k_in - 0.5
    seen = (crossing + 0.5) / k_in - 0.5 + (window.x if axis == 1 else window.y)
    shift_input = seen - edge

    skin = res.labels == int(Region.SKIN)
    row = skin[skin.shape[0] // 2] if axis == 1 else skin[:, skin.shape[1] // 2]
    first = int(np.argmax(row)) + (res.origin[0] if axis == 1 else res.origin[1])
    return shift_input, first - int(np.ceil(edge))


# (image w, h, face bbox, case id). Window = 2 * bbox side.
REGISTRATION_CASES = [
    (600, 600, (150, 150, 450, 450), "window 600 == image"),
    (1000, 1000, (250, 250, 750, 750), "window 1000 == image"),
    (800, 800, (150, 150, 650, 650), "window 1000 past the image"),
    (2000, 2000, (500, 500, 1500, 1500), "window 2000 == image, pre-shrink"),
    (2650, 2650, (662, 662, 1988, 1988), "window 2650 == image, pre-shrink"),
]


def test_registration_is_centre_consistent_in_network_input_and_label_map():
    label_errors = []
    failures = []
    for w, h, bbox, name in REGISTRATION_CASES:
        for axis in (1, 0):
            for frac in (0.2, 0.5, 0.8):
                edge = (w if axis == 1 else h) / 2 + 17 + frac
                shift, label_err = _measure_registration(w, h, bbox, edge, axis)
                label_errors.append(label_err)
                if abs(shift) > 0.1 or abs(label_err) > 1:
                    failures.append((name, "x" if axis == 1 else "y", frac, round(shift, 3), label_err))
    assert not failures, failures
    # no systematic sign in the label-map error
    assert abs(float(np.mean(label_errors))) < 0.5, label_errors
