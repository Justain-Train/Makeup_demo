"""Pure sRGB hex to CIELAB (D65) conversion. Stdlib only."""

import re

_HEX_RE = re.compile(r"^#([0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})$")
_MAX_HEX_LEN = 7

_M = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_WHITE = (0.95047, 1.0, 1.08883)
_EPS = 216 / 24389
_KAPPA = 24389 / 27


def parse_hex(s: str) -> tuple[int, int, int]:
    if not isinstance(s, str) or len(s) > _MAX_HEX_LEN or not s.isascii():
        raise ValueError("Invalid hex color; expected '#RGB' or '#RRGGBB'")
    match = _HEX_RE.fullmatch(s)
    if match is None:
        raise ValueError("Invalid hex color; expected '#RGB' or '#RRGGBB'")
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def normalize_hex(s: str) -> str:
    r, g, b = parse_hex(s)
    return f"#{r:02X}{g:02X}{b:02X}"


def srgb_to_linear(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def rgb_to_xyz(r: int, g: int, b: int) -> tuple[float, float, float]:
    lin = [srgb_to_linear(v / 255.0) for v in (r, g, b)]
    x, y, z = (sum(m * c for m, c in zip(row, lin)) for row in _M)
    return x, y, z


def xyz_to_lab(x: float, y: float, z: float) -> tuple[float, float, float]:
    def f(t: float) -> float:
        return t ** (1 / 3) if t > _EPS else (_KAPPA * t + 16) / 116

    fx, fy, fz = f(x / _WHITE[0]), f(y / _WHITE[1]), f(z / _WHITE[2])
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def hex_to_lab(s: str) -> tuple[float, float, float]:
    l_, a, b = xyz_to_lab(*rgb_to_xyz(*parse_hex(s)))
    return round(l_, 2), round(a, 2), round(b, 2)
