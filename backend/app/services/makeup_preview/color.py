"""Vectorised sRGB <-> CIELAB (D65) and CIEDE2000. Pure numpy."""

import re

import numpy as np

from app.schemas.makeup_component import MakeupComponent

_M_RGB2XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
    dtype=np.float64,
)
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0
_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


def _float_dtype(a: np.ndarray) -> type:
    return np.float64 if a.dtype == np.float64 else np.float32


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 (0..255) or float (0..1) sRGB (..., 3) -> Lab (..., 3). float32 unless input is float64."""
    a = np.asarray(rgb)
    dt = _float_dtype(a)
    x = a.astype(dt) / dt(255.0) if a.dtype == np.uint8 else a.astype(dt)
    x = np.clip(x, 0.0, 1.0)
    lin = np.where(x <= 0.04045, x / dt(12.92), ((x + dt(0.055)) / dt(1.055)) ** dt(2.4))
    xyz = (lin @ _M_RGB2XYZ.T.astype(dt)) / _WHITE.astype(dt)
    f = np.where(xyz > dt(_EPS), np.cbrt(xyz), (dt(_KAPPA) * xyz + dt(16.0)) / dt(116.0))
    out = np.empty(f.shape, dtype=dt)
    out[..., 0] = dt(116.0) * f[..., 1] - dt(16.0)
    out[..., 1] = dt(500.0) * (f[..., 0] - f[..., 1])
    out[..., 2] = dt(200.0) * (f[..., 1] - f[..., 2])
    return out


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Lab (..., 3) -> float sRGB in [0, 1], clipped."""
    a = np.asarray(lab)
    dt = _float_dtype(a)
    a = a.astype(dt)
    fy = (a[..., 0] + dt(16.0)) / dt(116.0)
    fx = fy + a[..., 1] / dt(500.0)
    fz = fy - a[..., 2] / dt(200.0)
    f = np.stack([fx, fy, fz], axis=-1)
    f3 = f**3
    xyz = np.where(f3 > dt(_EPS), f3, (dt(116.0) * f - dt(16.0)) / dt(_KAPPA))
    xyz = xyz * _WHITE.astype(dt)
    lin = np.clip(xyz @ _M_XYZ2RGB.T.astype(dt), 0.0, 1.0)
    srgb = np.where(lin <= dt(0.0031308), lin * dt(12.92), dt(1.055) * lin ** dt(1.0 / 2.4) - dt(0.055))
    return np.clip(srgb, 0.0, 1.0).astype(dt)


def hex_to_lab(color_hex: str) -> np.ndarray | None:
    m = _HEX.match(color_hex)
    if m is None:
        return None
    h = m.group(1)
    rgb = np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0
    return srgb_to_lab(rgb)


def component_target_lab(component: MakeupComponent) -> np.ndarray | None:
    """color_lab if present and finite, else color_hex, else None. float64 (3,)."""
    if component.color_lab is not None:
        lab = np.asarray(component.color_lab, dtype=np.float64)
        if lab.shape == (3,) and np.all(np.isfinite(lab)):
            return lab
    if component.color_hex is not None:
        return hex_to_lab(component.color_hex)
    return None


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 (kL=kC=kH=1), broadcasting over leading dims. float64."""
    p1 = np.asarray(lab1, dtype=np.float64)
    p2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = p1[..., 0], p1[..., 1], p1[..., 2]
    L2, a2, b2 = p2[..., 0], p2[..., 1], p2[..., 2]
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    cbar7 = ((c1 + c2) / 2.0) ** 7
    g = 0.5 * (1.0 - np.sqrt(cbar7 / (cbar7 + 25.0**7)))
    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.hypot(a1p, b1)
    c2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dl = L2 - L1
    dc = c2p - c1p
    prod = c1p * c2p
    diff = h2p - h1p
    dh = np.where(np.abs(diff) <= 180.0, diff, np.where(diff > 180.0, diff - 360.0, diff + 360.0))
    dh = np.where(prod == 0.0, 0.0, dh)
    dhh = 2.0 * np.sqrt(prod) * np.sin(np.radians(dh) / 2.0)

    lbar = (L1 + L2) / 2.0
    cbar = (c1p + c2p) / 2.0
    hsum = h1p + h2p
    hbar = np.where(
        prod == 0.0,
        hsum,
        np.where(np.abs(h1p - h2p) <= 180.0, hsum / 2.0, np.where(hsum < 360.0, (hsum + 360.0) / 2.0, (hsum - 360.0) / 2.0)),
    )
    t = (
        1.0
        - 0.17 * np.cos(np.radians(hbar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hbar))
        + 0.32 * np.cos(np.radians(3.0 * hbar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hbar - 63.0))
    )
    dtheta = 30.0 * np.exp(-(((hbar - 275.0) / 25.0) ** 2))
    cbar7p = cbar**7
    rc = 2.0 * np.sqrt(cbar7p / (cbar7p + 25.0**7))
    sl = 1.0 + 0.015 * (lbar - 50.0) ** 2 / np.sqrt(20.0 + (lbar - 50.0) ** 2)
    sc = 1.0 + 0.045 * cbar
    sh = 1.0 + 0.015 * cbar * t
    rt = -np.sin(np.radians(2.0 * dtheta)) * rc
    tl, tc, th = dl / sl, dc / sc, dhh / sh
    return np.sqrt(np.maximum(tl**2 + tc**2 + th**2 + rt * tc * th, 0.0))
