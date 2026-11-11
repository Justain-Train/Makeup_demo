"""EasyPortrait class ids and the comparison classes. The ONLY module that knows EasyPortrait ids.

Written independently of `app.services.segmentation.regions.MODEL_ID_TO_CANONICAL`: it speaks only the
canonical `Region` vocabulary that the API returns. EasyPortrait ids are verbatim from the dataset README
(hukenovs/easyportrait): 0 BACKGROUND, 1 PERSON, 2 SKIN, 3 LEFT_BROW, 4 RIGHT_BROW, 5 LEFT_EYE, 6 RIGHT_EYE,
7 LIPS, 8 TEETH.

Left/right: the dataset's LEFT_* is taken to be the image's left, i.e. the subject's right in the MediaPipe
landmark convention used by this project, so EP LEFT_EYE is compared with our `eye_r`. This is an assumption
that `side_order_ok` checks per image with centroid ordering; exceptions are counted and reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.segmentation.regions import NOT_PARSED, Region

EP_BACKGROUND, EP_PERSON, EP_SKIN = 0, 1, 2
EP_LEFT_BROW, EP_RIGHT_BROW, EP_LEFT_EYE, EP_RIGHT_EYE = 3, 4, 5, 6
EP_LIPS, EP_TEETH = 7, 8
EP_NUM_CLASSES = 9
EP_NAMES = (
    "BACKGROUND", "PERSON", "SKIN", "LEFT_BROW", "RIGHT_BROW", "LEFT_EYE", "RIGHT_EYE", "LIPS", "TEETH",
)

NUM_CANONICAL = max(int(r) for r in Region) + 1  # 19
NOT_PARSED_COL = NUM_CANONICAL  # confusion column for "not predicted" (outside the window, or no result)
CONF_COLS = NUM_CANONICAL + 1  # 20
COL_NAMES = tuple(r.name for r in sorted(Region)) + ("NOT_PARSED",)

MIN_PIXELS = 25  # a GT region smaller than this does not count as present


@dataclass(frozen=True)
class ComparisonClass:
    name: str
    ep_ids: tuple[int, ...]
    canonical: tuple[Region, ...]
    note: str = ""
    secondary: bool = False  # secondary classes are reported as "not comparable"
    small: bool = False  # small structures: boundary F and best-shift are computed


COMPARISONS: tuple[ComparisonClass, ...] = (
    ComparisonClass("skin", (EP_SKIN,), (Region.SKIN, Region.NOSE), "EP skin vs our skin + nose"),
    ComparisonClass("lips", (EP_LIPS,), (Region.LIP_UPPER, Region.LIP_LOWER), small=True),
    ComparisonClass("eye", (EP_LEFT_EYE, EP_RIGHT_EYE), (Region.EYE_R, Region.EYE_L), "merged", small=True),
    ComparisonClass("brow", (EP_LEFT_BROW, EP_RIGHT_BROW), (Region.BROW_R, Region.BROW_L), "merged", small=True),
    ComparisonClass("eye_R", (EP_LEFT_EYE,), (Region.EYE_R,), "EP LEFT_EYE (image left) vs our eye_R"),
    ComparisonClass("eye_L", (EP_RIGHT_EYE,), (Region.EYE_L,), "EP RIGHT_EYE (image right) vs our eye_L"),
    ComparisonClass("brow_R", (EP_LEFT_BROW,), (Region.BROW_R,), "EP LEFT_BROW (image left) vs our brow_R"),
    ComparisonClass("brow_L", (EP_RIGHT_BROW,), (Region.BROW_L,), "EP RIGHT_BROW (image right) vs our brow_L"),
    ComparisonClass(
        "teeth_vs_mouth_interior", (EP_TEETH,), (Region.MOUTH_INTERIOR,),
        "not comparable: EP teeth vs our mouth interior; read recall only", secondary=True,
    ),
    ComparisonClass(
        "whole_mouth", (EP_LIPS, EP_TEETH), (Region.LIP_UPPER, Region.LIP_LOWER, Region.MOUTH_INTERIOR),
        "not comparable: EP lips + teeth vs our lips + mouth interior", secondary=True, small=True,
    ),
)
COMPARISON_BY_NAME = {c.name: c for c in COMPARISONS}
HEADLINE = ("skin", "lips", "eye", "brow")
SIDED = ("eye_R", "eye_L", "brow_R", "brow_L")
NOT_SCORED = ("PERSON", "BACKGROUND")

# (EP id of the image-left structure, EP id of the image-right one, our region for the first, our region for the second)
SIDE_PAIRS = {
    "eye": (EP_LEFT_EYE, EP_RIGHT_EYE, Region.EYE_R, Region.EYE_L),
    "brow": (EP_LEFT_BROW, EP_RIGHT_BROW, Region.BROW_R, Region.BROW_L),
}


def pred_to_columns(pred: np.ndarray) -> np.ndarray:
    """Canonical label map -> confusion column index. Anything that is not a canonical id is NOT_PARSED."""
    cols = pred.astype(np.intp, copy=True)
    cols[(pred == NOT_PARSED) | (pred >= NUM_CANONICAL)] = NOT_PARSED_COL
    return cols


def ep_mask(gt: np.ndarray, ids: tuple[int, ...]) -> np.ndarray:
    return np.isin(gt, ids)


def pred_mask(pred: np.ndarray, regions: tuple[Region, ...]) -> np.ndarray:
    return np.isin(pred, [int(r) for r in regions])


def column_ids(regions: tuple[Region, ...]) -> list[int]:
    return [int(r) for r in regions]
