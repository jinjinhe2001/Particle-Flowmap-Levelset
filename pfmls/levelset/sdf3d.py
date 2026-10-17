"""Initial 3D shapes as signed distance functions (host side; ``phi < 0`` inside).

Each constructor returns an :class:`SDF3`, a callable ``phi = sdf(x, y, z)`` on numpy arrays that can be
combined with ``|`` (union, min), ``&`` (intersection, max) and ``-`` (difference). Boolean combinations
are exact outside / approximate inside, as usual for min/max CSG.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class SDF3:
    """A signed distance function ``phi(x, y, z)`` evaluated on numpy arrays."""

    def __init__(self, fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], name: str = "sdf"):
        self.fn = fn
        self.name = name

    def __call__(self, x, y, z) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        return self.fn(x, y, z)

    def __or__(self, other: "SDF3") -> "SDF3":
        return union(self, other)

    def __and__(self, other: "SDF3") -> "SDF3":
        return intersection(self, other)

    def __sub__(self, other: "SDF3") -> "SDF3":
        return difference(self, other)

    def __neg__(self) -> "SDF3":
        return SDF3(lambda x, y, z: -self(x, y, z), f"-{self.name}")

    def __repr__(self) -> str:
        return f"SDF3({self.name})"


def sphere(center=(0.35, 0.35, 0.35), radius: float = 0.15) -> SDF3:
    """Ball of ``radius`` around ``center`` (default: the 3D LeVeque / Enright test)."""
    cx, cy, cz = center
    return SDF3(lambda x, y, z: np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - radius,
                f"sphere({cx},{cy},{cz},{radius})")


def half_space(point=(0.0, 0.0, 0.3), normal=(0.0, 0.0, 1.0)) -> SDF3:
    """``phi = (x - point) . n``: negative on the side opposite to ``normal`` (e.g. a pool below z)."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    p = np.asarray(point, dtype=np.float64)
    return SDF3(lambda x, y, z: (x - p[0]) * n[0] + (y - p[1]) * n[1] + (z - p[2]) * n[2], "half_space")


def union(*sdfs: SDF3) -> SDF3:
    return SDF3(lambda x, y, z: np.minimum.reduce([s(x, y, z) for s in sdfs]), "union")


def intersection(*sdfs: SDF3) -> SDF3:
    return SDF3(lambda x, y, z: np.maximum.reduce([s(x, y, z) for s in sdfs]), "intersection")


def difference(a: SDF3, b: SDF3) -> SDF3:
    """``a`` minus ``b``."""
    return SDF3(lambda x, y, z: np.maximum(a(x, y, z), -b(x, y, z)), "difference")
