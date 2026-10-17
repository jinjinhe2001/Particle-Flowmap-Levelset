"""Initial shapes as signed distance functions (host side, numpy; ``phi < 0`` inside).

Each constructor returns an :class:`SDF`, a callable ``phi = sdf(x, y)`` on numpy arrays that can be
combined with ``|`` (union, min), ``&`` (intersection, max) and ``-`` (difference).
Boolean combinations are exact outside / approximate inside, as usual for min/max CSG.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class SDF:
    """A signed distance function ``phi(x, y)`` evaluated on numpy arrays."""

    def __init__(self, fn: Callable[[np.ndarray, np.ndarray], np.ndarray], name: str = "sdf"):
        self.fn = fn
        self.name = name

    def __call__(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return self.fn(x, y)

    def __or__(self, other: "SDF") -> "SDF":
        return union(self, other)

    def __and__(self, other: "SDF") -> "SDF":
        return intersection(self, other)

    def __sub__(self, other: "SDF") -> "SDF":
        return difference(self, other)

    def __neg__(self) -> "SDF":
        return SDF(lambda x, y: -self(x, y), f"-{self.name}")

    def __repr__(self) -> str:
        return f"SDF({self.name})"


def circle(center=(0.5, 0.75), radius: float = 0.15) -> SDF:
    """Disk of ``radius`` around ``center``."""
    cx, cy = center
    return SDF(lambda x, y: np.hypot(x - cx, y - cy) - radius, f"circle({cx},{cy},{radius})")


def half_plane(point=(0.0, 0.5), normal=(0.0, 1.0)) -> SDF:
    """``phi = (x - point) . n``: negative on the side opposite to ``normal`` (e.g. a pool below y)."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    px, py = point
    return SDF(lambda x, y: (x - px) * n[0] + (y - py) * n[1], f"half_plane({point},{tuple(n)})")


def rectangle(lo=(0.25, 0.25), hi=(0.75, 0.75)) -> SDF:
    """Axis-aligned box ``[lo, hi]`` (exact signed distance)."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    c, e = 0.5 * (lo + hi), 0.5 * (hi - lo)

    def fn(x, y):
        qx = np.abs(x - c[0]) - e[0]
        qy = np.abs(y - c[1]) - e[1]
        outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
        return outside + np.minimum(np.maximum(qx, qy), 0.0)

    return SDF(fn, f"rectangle({tuple(lo)},{tuple(hi)})")


def slotted_disk(center=(0.5, 0.75), radius: float = 0.15, slot_width: float = 0.05, slot_top: float = 0.85) -> SDF:
    """Zalesak's slotted disk (signed distance to the disk with the slot cut out).

    The slot ``|x - cx| < w/2, y < slot_top`` is cut from the disk.
    """
    cx, cy, cr = center[0], center[1], radius
    sl, sr = cx - 0.5 * slot_width, cx + 0.5 * slot_width
    st = slot_top
    sb = cy - np.sqrt(cr**2 - (0.5 * slot_width) ** 2)

    def fn(x, y):
        rc = np.hypot(x - cx, y - cy)
        inside = (rc <= cr) & ~((np.abs(x - cx) < 0.5 * slot_width) & (y < st))
        # inside: distance to the circle or to the slot (clamp onto the slot box)
        phi_c = cr - rc
        xp = np.clip(x, sl, sr)
        yp = np.clip(y, sb, st)
        phi_s = np.hypot(x - xp, y - yp)
        d_in = np.minimum(phi_c, phi_s)
        # outside: circle distance, or slot walls / slot-bottom corners inside the slot column
        in_column = ~((x < sl) | (x > sr) | (y > st))
        d_slot = np.maximum(np.maximum(sl - x, x - sr), y - st)
        d_corner = np.maximum(-np.hypot(x - sl, y - sb), -np.hypot(x - sr, y - sb))
        d_out = np.where(in_column, np.where(y > sb, d_slot, d_corner), phi_c)
        return -np.where(inside, d_in, d_out)

    return SDF(fn, "slotted_disk")


def union(*sdfs: SDF) -> SDF:
    return SDF(lambda x, y: np.minimum.reduce([s(x, y) for s in sdfs]), "union")


def intersection(*sdfs: SDF) -> SDF:
    return SDF(lambda x, y: np.maximum.reduce([s(x, y) for s in sdfs]), "intersection")


def difference(a: SDF, b: SDF) -> SDF:
    """``a`` minus ``b``."""
    return SDF(lambda x, y: np.maximum(a(x, y), -b(x, y)), "difference")
