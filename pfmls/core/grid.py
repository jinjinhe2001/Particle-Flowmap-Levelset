"""Uniform cell-centred grid descriptors.

The domain is ``[0, nx*dx] x [0, ny*dx]`` (origin at 0). Node ``(i, j)`` is the centre of cell
``(i, j)`` at ``((i + 0.5) dx, (j + 0.5) dx)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid2D:
    """A 2D cell-centred grid with square cells of size ``dx``."""

    nx: int
    ny: int
    dx: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)

    @property
    def size(self) -> tuple[float, float]:
        """Physical extent ``(Lx, Ly)``."""
        return (self.nx * self.dx, self.ny * self.dx)

    @property
    def num_cells(self) -> int:
        return self.nx * self.ny

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre coordinates as two ``(nx, ny)`` arrays (``indexing='ij'``)."""
        xs = (np.arange(self.nx) + 0.5) * self.dx
        ys = (np.arange(self.ny) + 0.5) * self.dx
        return np.meshgrid(xs, ys, indexing="ij")

    def refined(self, factor: int) -> "Grid2D":
        """The grid refined ``factor`` times in each direction (same domain)."""
        return Grid2D(self.nx * factor, self.ny * factor, self.dx / factor)
