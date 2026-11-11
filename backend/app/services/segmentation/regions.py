"""Canonical anatomical region vocabulary, plus the ONLY model-specific mapping.

Everything outside `app/services/segmentation/` (zone engine, API, frontend)
uses the canonical `Region` names and ids defined here. A face-parsing model's
own class ids appear in exactly one place: `MODEL_ID_TO_CANONICAL` below, next
to the parser that consumes it. Swapping the model (for example for a
commercially trained one) means changing the parser and that mapping only.

Canonical ids for every left/right region are deliberately different from the
current model's class ids, so code that forgets to apply the mapping fails
loudly in tests instead of silently swapping sides.

"left"/"right" follow the MediaPipe LANDMARK convention: subject-right is the side
that contains landmarks 33, 133 and 468. In an unmirrored selfie that is the
image's left. The model's own left/right classes are NOT used (they were found
to be unreliable, e.g. on mirrored photos): paired classes are mapped to the
side-agnostic `Unsided` ids and `sides.assign_sides` gives them a side from the
landmark geometry. A horizontally mirrored photo therefore gets sides that follow
the image geometry, i.e. swapped relative to the real person. Mirroring cannot be
detected from pixels; a future capture flag could correct it.
"""

from enum import IntEnum
from typing import Literal

Side = Literal["left", "right"]

NOT_PARSED = 255
"""Label-map value for pixels outside the parse window (never a region id)."""


class Region(IntEnum):
    BACKGROUND = 0
    SKIN = 1
    NOSE = 2
    BROW_L = 3
    BROW_R = 4
    EYE_L = 5
    EYE_R = 6
    LIP_UPPER = 7
    LIP_LOWER = 8
    MOUTH_INTERIOR = 9
    EAR_L = 10
    EAR_R = 11
    HAIR = 12
    NECK = 13
    CLOTH = 14
    HAT = 15
    GLASSES = 16
    EARRING = 17
    NECKLACE = 18


class Unsided(IntEnum):
    """Internal ids for paired classes before a side is assigned. Never in a response."""

    EYE = 200
    BROW = 201
    EAR = 202


REGION_SIDE: dict[Region, Side] = {
    Region.BROW_R: "right",
    Region.BROW_L: "left",
    Region.EYE_R: "right",
    Region.EYE_L: "left",
    Region.EAR_R: "right",
    Region.EAR_L: "left",
}


def region_name(region: Region) -> str:
    return region.name.lower()


# --- Model-specific mapping (BiSeNet trained on CelebAMask-HQ, 19 classes) ---
#
# Model class order, verified empirically on tests/fixtures/portrait.jpg and on
# a second photo with glasses and a hat (see docs/face_segmentation.md):
#   0 background  1 skin   2 l_brow  3 r_brow  4 l_eye  5 r_eye  6 eye_g
#   7 l_ear  8 r_ear  9 ear_r  10 nose  11 mouth  12 u_lip  13 l_lip
#   14 neck  15 neck_l  16 cloth  17 hair  18 hat
#
# The model's l_*/r_* classes (2/3, 4/5, 7/8) are merged into side-agnostic ids.
# On the upright fixture l_* is the subject's left, but on a mirrored photo the
# model labelled both irises as l_eye, so its side signal cannot be trusted.
MODEL_ID_TO_CANONICAL: dict[int, Region | Unsided] = {
    0: Region.BACKGROUND,
    1: Region.SKIN,
    2: Unsided.BROW,
    3: Unsided.BROW,
    4: Unsided.EYE,
    5: Unsided.EYE,
    6: Region.GLASSES,
    7: Unsided.EAR,
    8: Unsided.EAR,
    9: Region.EARRING,  # model name `ear_r` = earrings (name only, NOT verified on a photo)
    10: Region.NOSE,
    11: Region.MOUTH_INTERIOR,  # teeth / open mouth interior, verified on the smiling fixture
    12: Region.LIP_UPPER,
    13: Region.LIP_LOWER,
    14: Region.NECK,
    15: Region.NECKLACE,  # model name `neck_l` = necklace (name only, NOT verified on a photo)
    16: Region.CLOTH,
    17: Region.HAIR,
    18: Region.HAT,
}

# Model classes whose meaning was confirmed by rendering them on real photos.
# Earring (9) and necklace (15) were never seen in the test photos.
VERIFIED_MODEL_IDS: frozenset[int] = frozenset(set(range(19)) - {9, 15})

NUM_MODEL_CLASSES = len(MODEL_ID_TO_CANONICAL)


def build_label_lut() -> "list[int]":
    """256-entry lookup table: model class id -> canonical id (255 elsewhere)."""
    lut = [NOT_PARSED] * 256
    for model_id, region in MODEL_ID_TO_CANONICAL.items():
        lut[model_id] = int(region)
    return lut
