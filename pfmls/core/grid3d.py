"""Uniform cell-centred 3D grid descriptor.

The domain is ``[0, nx dx] x [0, ny dx] x [0, nz dx]`` (origin at 0). Node ``(i, j, k)`` is the centre
of cell ``(i, j, k)`` at ``((i + 0.5) dx, (j + 0.5) dx, (k + 0.5) dx)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid3D:
    """A 3D cell-centred grid with cubic cells of size ``dx``."""

    nx: int
    ny: int
    nz: int
    dx: float

    @classmethod
    def unit_cube(cls, res: int | tuple[int, int, int]) -> "Grid3D":
        """Grid covering ``[0, 1] x [0, ny/nx] x [0, nz/nx]`` with ``dx = 1 / nx``."""
        nx, ny, nz = (res, res, res) if isinstance(res, int) else (int(res[0]), int(res[1]), int(res[2]))
        return cls(nx, ny, nz, 1.0 / nx)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def size(self) -> tuple[float, float, float]:
        """Physical extent ``(Lx, Ly, Lz)``."""
        return (self.nx * self.dx, self.ny * self.dx, self.nz * self.dx)

    @property
    def num_cells(self) -> int:
        return self.nx * self.ny * self.nz

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cell-centre coordinates as three ``(nx, ny, nz)`` arrays (``indexing='ij'``)."""
        axes = [(np.arange(n) + 0.5) * self.dx for n in self.shape]
        return np.meshgrid(*axes, indexing="ij")

    def refined(self, factor: int) -> "Grid3D":
        """The grid refined ``factor`` times in each direction (same domain)."""
        return Grid3D(self.nx * factor, self.ny * factor, self.nz * factor, self.dx / factor)
