"""Placement key -> candidate application zone ids.

Each entry is always intersected with the product's layer; it is never used on its own.
"""

from types import MappingProxyType

PLACEMENT_ZONES = MappingProxyType({
    "lifted_cheek": (
        "cheek_upper_r", "cheek_upper_l", "cheekbone_r", "cheekbone_l",
        "contour_hollow_r", "contour_hollow_l",
    ),
    "apple_of_cheek": ("cheek_apple_r", "cheek_apple_l"),
    "outer_v": ("eye_corner_outer_r", "eye_corner_outer_l"),
    "inner_corner": ("eye_corner_inner_r", "eye_corner_inner_l"),
    "lash_line": ("lash_line_upper_r", "lash_line_upper_l", "lash_line_lower_r", "lash_line_lower_l"),
    "lid": ("eyelid_upper_r", "eyelid_upper_l"),
    "brow_bone": ("brow_bone_r", "brow_bone_l"),
    "hollow_of_cheek": ("contour_hollow_r", "contour_hollow_l"),
    "nose_bridge": ("nose_bridge", "nose_side_r", "nose_side_l"),
    "under_eye": (
        "under_eye_r", "under_eye_l", "under_eye_inner_r", "under_eye_inner_l",
        "under_eye_outer_r", "under_eye_outer_l",
    ),
    "full_lip": ("lip_full",),
    "cupid_bow": ("cupid_bow",),
    "outer_lip_edge": ("lip_outer_boundary",),
    "forehead": ("forehead_center",),
    "chin": ("chin_center",),
    "temples": ("temple_r", "temple_l"),
    "hairline": ("forehead_perimeter",),
    "outer_cheek": ("cheek_outer_r", "cheek_outer_l"),
    "jawline": ("jawline", "jawline_outer_r", "jawline_outer_l"),
})
