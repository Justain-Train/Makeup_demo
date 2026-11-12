"""Frozen makeup vocabulary data. Stage/layer keys are strings only; no zone ids."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

CATEGORIES: tuple[str, ...] = (
    "skin_prep", "color_correction", "foundation", "concealer", "setting", "bronzer_contour",
    "blush", "highlighter", "eyebrows", "eye", "lip", "special_effects",
)
KINDS: tuple[str, ...] = ("product_types", "finishes", "techniques", "placements", "directions")


@dataclass(frozen=True)
class Term:
    key: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProductType(Term):
    category: str = ""
    stage_key: str | None = None
    layer_key: str | None = None


@dataclass(frozen=True)
class Placement(Term):
    categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class Technique(Term):
    categories: tuple[str, ...] = ()


def _pt(key: str, category: str, stage: str | None, layer: str | None, *aliases: str) -> ProductType:
    return ProductType(key, tuple(aliases), category, stage, layer)


_PRODUCT_TYPES = (
    _pt("primer", "skin_prep", "primer", "primer", "face primer"),
    _pt("moisturizer", "skin_prep", "primer", "primer", "moisturiser", "hydrator"),
    _pt("spf", "skin_prep", "primer", "primer", "sunscreen"),
    _pt("color_corrector", "color_correction", "color_corrector", "color_corrector", "colour corrector"),
    _pt("foundation", "foundation", "foundation", "foundation"),
    _pt("skin_tint", "foundation", "foundation", "foundation", "tinted moisturizer", "bb cream"),
    _pt("concealer", "concealer", "concealer", "concealer"),
    _pt("setting_powder", "setting", "setting_powder", "setting_powder", "loose powder", "pressed powder"),
    _pt("setting_spray", "setting", "setting_spray", "setting_spray", "fixing spray"),
    _pt("bronzer", "bronzer_contour", "bronzer_contour", "bronzer"),
    _pt("contour_powder", "bronzer_contour", "bronzer_contour", "contour", "contour"),
    _pt("contour_cream", "bronzer_contour", "bronzer_contour", "contour", "contour stick"),
    _pt("cream_blush", "blush", "blush", "blush"),
    _pt("powder_blush", "blush", "blush", "blush"),
    _pt("liquid_blush", "blush", "blush", "blush"),
    _pt("powder_highlighter", "highlighter", "highlighter", "highlighter"),
    _pt("cream_highlighter", "highlighter", "highlighter", "highlighter"),
    _pt("liquid_highlighter", "highlighter", "highlighter", "highlighter"),
    _pt("brow_pencil", "eyebrows", "eyebrow_eye", "eyebrow", "eyebrow pencil"),
    _pt("brow_powder", "eyebrows", "eyebrow_eye", "eyebrow"),
    _pt("brow_gel", "eyebrows", "eyebrow_eye", "eyebrow"),
    _pt("eyeshadow", "eye", "eyebrow_eye", "eye", "eye shadow"),
    _pt("eyeliner", "eye", "eyebrow_eye", "eye", "liner"),
    _pt("mascara", "eye", "eyebrow_eye", "eye"),
    _pt("eye_primer", "eye", "eyebrow_eye", "eye"),
    _pt("lipstick", "lip", "lips", "lipstick"),
    _pt("lip_tint", "lip", "lips", "lipstick"),
    _pt("lip_liner", "lip", "lips", "lip_liner", "lip pencil"),
    _pt("lip_gloss", "lip", "lips", "gloss", "gloss"),
    _pt("face_gems", "special_effects", None, None, "rhinestones"),
    _pt("glitter", "special_effects", None, None),
    _pt("face_paint", "special_effects", None, None),
    _pt("freckles", "special_effects", None, None, "faux freckles"),
)

_FINISHES = (
    Term("matte"), Term("satin", ("semi-matte",)), Term("dewy", ("glowy", "radiant")), Term("natural"),
    Term("shimmer", ("pearl",)), Term("metallic"), Term("glitter"), Term("glossy", ("shiny",)),
    Term("velvet"),
)

_FACE = ("foundation", "concealer", "bronzer_contour", "blush", "highlighter", "setting")
_EYE_BROW = ("eyebrows", "eye")


def _tq(key: str, cats: tuple[str, ...], *aliases: str) -> Technique:
    return Technique(key, tuple(aliases), cats)


_TECHNIQUES = (
    _tq("blending", (), "blend", "buffing"),
    _tq("stippling", _FACE + ("lip",), "dab", "pat"),
    _tq("sweeping", ("bronzer_contour", "blush", "highlighter", "eye"), "sweep"),
    _tq("smoky_blending", ("eye",), "smoky", "smokey"),
    _tq("winged", ("eye",), "wing"),
    _tq("gradient", ("eye", "lip", "blush"), "ombre"),
    _tq("layering", (), "build up"),
    _tq("defining", _EYE_BROW + ("lip",), "precise"),
    _tq("diffusing", (), "soft diffuse"),
)


def _pl(key: str, cats: tuple[str, ...], *aliases: str) -> Placement:
    return Placement(key, tuple(aliases), cats)


_PLACEMENTS = (
    _pl("lifted_cheek", ("blush", "bronzer_contour", "highlighter"), "cheekbone lift"),
    _pl("apple_of_cheek", ("blush",), "apple"),
    _pl("outer_v", ("eye",)),
    _pl("inner_corner", ("eye", "highlighter")),
    _pl("lash_line", ("eye",), "lashline"),
    _pl("lid", ("eye",), "eyelid"),
    _pl("brow_bone", ("highlighter", "eyebrows")),
    _pl("hollow_of_cheek", ("bronzer_contour",), "cheek hollow"),
    _pl("nose_bridge", ("highlighter", "bronzer_contour")),
    _pl("under_eye", ("concealer", "color_correction", "eye", "setting")),
    _pl("full_lip", ("lip",), "all over lip"),
    _pl("cupid_bow", ("lip", "highlighter"), "cupid's bow"),
    _pl("outer_lip_edge", ("lip",), "lip line"),
    _pl("forehead", ("highlighter",), "center of forehead", "middle of forehead"),
    _pl("chin", ("highlighter",), "center of chin"),
    _pl("temples", ("bronzer_contour",), "temple"),
    _pl("hairline", ("bronzer_contour",), "hairline perimeter"),
    _pl("outer_cheek", ("blush", "bronzer_contour"), "cheek outer"),
    _pl("jawline", ("bronzer_contour",), "jaw", "jaw line"),
)

_DIRECTIONS = (
    Term("upward", ("up", "lifted")), Term("outward", ("out", "toward temple")), Term("inward"),
    Term("downward", ("down",)), Term("horizontal"), Term("diagonal", ("angled",)),
    Term("sweeping_back", ("backward",)),
)


def _by_key(terms: tuple[Term, ...]) -> Mapping[str, Term]:
    return MappingProxyType({t.key: t for t in terms})


REGISTRY: Mapping[str, Mapping[str, Term]] = MappingProxyType({
    "product_types": _by_key(_PRODUCT_TYPES),
    "finishes": _by_key(_FINISHES),
    "techniques": _by_key(_TECHNIQUES),
    "placements": _by_key(_PLACEMENTS),
    "directions": _by_key(_DIRECTIONS),
})
