"""Zone catalog (what each zone is) and the stage-to-zone map.

Zones are defined once here. `derive.py` builds the masks. Implemented: all 70 zones (the 18 initial zones
plus Phase 4 batches 1 to 6). The eyelid crease cannot be derived and is never added. The highlighter and
bronzer layers were extended in Phase 6 Batch 4.

Stages and their product layers are declared in `STAGES` below (single source of truth).

Left/right (`_l`/`_r`) are the SUBJECT's own sides.
"""

from dataclasses import dataclass
from typing import Literal

Side = Literal["left", "right"]
Source = Literal["segmentation", "landmarks"]


@dataclass(frozen=True)
class ZoneSpec:
    id: str
    name: str
    group: str
    side: Side | None
    sources: tuple[Source, ...]
    # Canonical regions the mask is a subset of. Face zones are subsets of the
    # `skin` region (and of the face_skin zone, which also removes landmark
    # eye/lip areas); eye/brow/nose zones are the parser region itself.
    clipped_to: tuple[str, ...]


def _lr(base: str, name: str, group: str, sources: tuple[Source, ...], clipped: tuple[str, ...]) -> list[ZoneSpec]:
    # "{s}" in `clipped` is replaced by l / r.
    return [
        ZoneSpec(f"{base}_{s}", f"{word} {name}", group, side, sources,
                 tuple(c.replace("{s}", s) for c in clipped))
        for s, side, word in (("l", "left", "Left"), ("r", "right", "Right"))
    ]


_SEG_LM: tuple[Source, ...] = ("segmentation", "landmarks")

ZONE_SPECS: list[ZoneSpec] = [
    ZoneSpec("face_skin", "Facial skin", "face", None, _SEG_LM, ("skin", "nose")),
    ZoneSpec("forehead", "Forehead", "forehead", None, _SEG_LM, ("skin",)),
    *_lr("cheek", "cheek", "cheek", _SEG_LM, ("skin",)),
    ZoneSpec("nose", "Nose", "nose", None, _SEG_LM, ("nose",)),
    ZoneSpec("chin", "Chin", "chin", None, _SEG_LM, ("skin",)),
    ZoneSpec("jawline", "Jawline", "jaw", None, _SEG_LM, ("skin",)),
    *_lr("under_eye", "under-eye", "eye", _SEG_LM, ("skin",)),
    *_lr("eye", "eye", "eye", ("segmentation",), ("eye_{s}",)),
    *_lr("brow", "eyebrow", "brow", ("segmentation",), ("brow_{s}",)),
    ZoneSpec("lip_upper", "Upper lip", "lip", None, ("landmarks",), ()),
    ZoneSpec("lip_lower", "Lower lip", "lip", None, ("landmarks",), ()),
    ZoneSpec("lip_full", "Full lips", "lip", None, ("landmarks",), ()),
    ZoneSpec("lip_outer_boundary", "Outer lip boundary", "lip", None, ("landmarks",), ()),
    ZoneSpec("lip_inner_boundary", "Inner lip boundary", "lip", None, ("landmarks",), ()),
    ZoneSpec("chin_center", "Chin center", "chin", None, _SEG_LM, ("skin",)),
    ZoneSpec("under_lower_lip", "Under lower lip", "chin", None, _SEG_LM, ("skin",)),
    ZoneSpec("mouth_surround", "Mouth surround", "lip", None, _SEG_LM, ("skin",)),
    ZoneSpec("cupid_bow", "Cupid's bow", "lip", None, ("landmarks",), ()),
    *_lr("lip_corner", "lip corner", "lip", ("landmarks",), ()),
    ZoneSpec("forehead_center", "Forehead center", "forehead", None, _SEG_LM, ("skin",)),
    ZoneSpec("forehead_perimeter", "Forehead perimeter", "forehead", None, _SEG_LM, ("skin",)),
    *_lr("temple", "temple", "forehead", _SEG_LM, ("skin",)),
    *_lr("jawline_outer", "outer jawline", "jaw", _SEG_LM, ("skin",)),
    *_lr("under_eye_inner", "inner under-eye", "eye", _SEG_LM, ("skin",)),
    *_lr("under_eye_outer", "outer under-eye", "eye", _SEG_LM, ("skin",)),
    *_lr("eye_corner_inner", "inner eye corner", "eye", _SEG_LM, ()),
    *_lr("eye_corner_outer", "outer eye corner", "eye", _SEG_LM, ()),
    *_lr("lash_line_upper", "upper lash line", "eye", ("landmarks",), ()),
    *_lr("lash_line_lower", "lower lash line", "eye", ("landmarks",), ()),
    *_lr("eyelid_upper", "upper eyelid", "eye", _SEG_LM, ("skin",)),
    *_lr("lower_eyelid", "lower eyelid", "eye", _SEG_LM, ("skin",)),
    *_lr("brow_head", "eyebrow head", "brow", _SEG_LM, ()),
    *_lr("brow_arch", "eyebrow arch", "brow", _SEG_LM, ()),
    *_lr("brow_tail", "eyebrow tail", "brow", _SEG_LM, ()),
    *_lr("brow_bone", "brow bone", "brow", _SEG_LM, ("skin",)),
    *_lr("cheek_apple", "cheek apple", "cheek", _SEG_LM, ("skin",)),
    *_lr("cheek_upper", "upper cheek", "cheek", _SEG_LM, ("skin",)),
    *_lr("cheek_outer", "outer cheek", "cheek", _SEG_LM, ("skin",)),
    *_lr("cheekbone", "cheekbone", "cheek", _SEG_LM, ("skin",)),
    *_lr("contour_hollow", "contour hollow", "cheek", _SEG_LM, ("skin",)),
    ZoneSpec("nose_bridge", "Nose bridge", "nose", None, _SEG_LM, ("nose",)),
    ZoneSpec("nose_tip", "Nose tip", "nose", None, _SEG_LM, ("nose",)),
    *_lr("nose_side", "nose side", "nose", _SEG_LM, ("nose",)),
    *_lr("nostril", "nostril", "nose", _SEG_LM, ("nose",)),
]

ZONE_SPEC_BY_ID: dict[str, ZoneSpec] = {z.id: z for z in ZONE_SPECS}


@dataclass(frozen=True)
class StageSpec:
    number: int
    key: str
    name: str
    layers: dict[str, list[str]]  # layer key -> zone ids (insertion order kept); [] = Phase 2 zones, none yet


_FACE = ["face_skin", "forehead", "cheek_r", "cheek_l", "nose", "chin", "jawline"]

_MOUTH_CHIN = ["mouth_surround", "under_lower_lip", "chin_center"]
_FOREHEAD_TEMPLE_JAW = ["forehead_center", "forehead_perimeter", "temple_r", "temple_l",
                        "jawline_outer_r", "jawline_outer_l"]
_LIP_DETAIL = ["cupid_bow", "lip_corner_r", "lip_corner_l"]
_UNDER_EYE_HALVES = ["under_eye_inner_r", "under_eye_inner_l", "under_eye_outer_r", "under_eye_outer_l"]
_BROW_DETAIL = ["brow_head_r", "brow_arch_r", "brow_tail_r", "brow_head_l", "brow_arch_l", "brow_tail_l"]
_EYELIDS = ["eyelid_upper_r", "eyelid_upper_l", "lower_eyelid_r", "lower_eyelid_l"]
_EYE_DETAIL = ["eye_corner_inner_r", "eye_corner_inner_l", "eye_corner_outer_r", "eye_corner_outer_l",
               "lash_line_upper_r", "lash_line_upper_l", "lash_line_lower_r", "lash_line_lower_l"]

STAGES: list[StageSpec] = [
    StageSpec(1, "primer", "Primer", {"primer": [*_FACE, "under_eye_r", "under_eye_l", *_MOUTH_CHIN,
                                                    *_FOREHEAD_TEMPLE_JAW]}),
    StageSpec(2, "color_corrector", "Color Corrector", {
        "color_corrector": ["under_eye_r", "under_eye_l", "cheek_r", "cheek_l", "nose", "chin", "forehead",
                            *_UNDER_EYE_HALVES]}),
    StageSpec(3, "foundation", "Foundation / Skin Tint", {"foundation": [*_FACE, *_MOUTH_CHIN, *_FOREHEAD_TEMPLE_JAW]}),
    StageSpec(4, "concealer", "Concealer", {"concealer": ["under_eye_r", "under_eye_l", *_UNDER_EYE_HALVES]}),
    StageSpec(5, "setting_powder", "Setting Powder", {
        "setting_powder": ["face_skin", "under_eye_r", "under_eye_l", "chin", *_MOUTH_CHIN]}),
    # Contour and bronzer are separate layers by requirement.
    StageSpec(6, "bronzer_contour", "Bronzer & Contour", {
        "contour": ["jawline", "contour_hollow_r", "contour_hollow_l", "nose_side_r", "nose_side_l"],
        "bronzer": ["temple_r", "temple_l", "forehead_perimeter", "cheek_outer_r", "cheek_outer_l",
                    "jawline_outer_r", "jawline_outer_l"]}),
    # Apple, upper and outer cheek are alternative placements and may overlap.
    StageSpec(7, "blush", "Blush", {"blush": ["cheek_apple_r", "cheek_apple_l", "cheek_upper_r", "cheek_upper_l",
                                              "cheek_outer_r", "cheek_outer_l"]}),
    StageSpec(8, "highlighter", "Highlighter", {
        "highlighter": ["brow_bone_r", "brow_bone_l", "cheekbone_r", "cheekbone_l", "nose_bridge", "nose_tip",
                        "forehead_center", "chin_center", "eye_corner_inner_r", "eye_corner_inner_l",
                        "cupid_bow"]}),
    StageSpec(9, "eyebrow_eye", "Eyebrow & Eye Makeup", {
        "eyebrow": ["brow_r", "brow_l", *_BROW_DETAIL],
        "eye": ["eye_r", "eye_l", *_EYE_DETAIL, *_EYELIDS]}),
    # Lipstick, lip liner and gloss are separate masks by requirement.
    StageSpec(10, "lips", "Lips", {
        "lipstick": ["lip_full", "lip_upper", "lip_lower", "lip_inner_boundary", *_LIP_DETAIL],
        "lip_liner": ["lip_outer_boundary", *_LIP_DETAIL],
        "gloss": ["lip_full"]}),
    StageSpec(11, "setting_spray", "Setting Spray", {"setting_spray": [*_FACE, *_MOUTH_CHIN, *_FOREHEAD_TEMPLE_JAW]}),
]
