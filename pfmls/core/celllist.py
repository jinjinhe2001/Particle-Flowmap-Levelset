"""Particle binning into grid cells by a counting sort over the valid particles (no per-cell cap).

Each valid particle inside the domain belongs to the cell ``floor(x / dx)``; invalid, out-of-domain
and non-finite particles are not binned. One build is

1. an atomic histogram of the binned particles into ``counts``;
2. an exclusive scan of ``counts`` into ``start`` (``num_cells + 1`` entries, so the range of cell
   ``c`` is ``[start[c], start[c + 1])``);
3. an atomic scatter of the particle ids into a scratch list (arbitrary order within a cell), which
   also returns ``counts`` to zero for the next build;
4. a parallel placement: each particle counts the smaller ids in its cell's scratch range and
   writes itself to that rank, so particles inside a cell are listed by increasing id and neighbour
   sums are deterministic run to run.

Every pass is one thread per particle slot (idle slots exit at once). The placement reads the
particle's own cell range, so its depth is the largest per-cell count; crowded cells (thousands of
particles, e.g. a compressing flow without reseeding) stay cheap because the work is spread over
the cell's particles. All launches are asynchronous and graph-capturable; there is no host
synchronisation.
"""

from __future__ import annotations

import warp as wp


@wp.struct
class CellListView:
    """Device view of a cell list, passed by value to kernels.

    The particles of cell ``c = ci * ny + cj`` are ``ids[start[c]:end[c]]``.
    """

    start: wp.array(dtype=wp.int32)
    end: wp.array(dtype=wp.int32)
    ids: wp.array(dtype=wp.int32)
    nx: int
    ny: int
    inv_dx: float


@wp.func
def cell_of(x: wp.vec2, inv_dx: float):
    """Integer cell coordinates ``floor(x / dx)``."""
    return int(wp.floor(x[0] * inv_dx)), int(wp.floor(x[1] * inv_dx))


@wp.func
def _binned_cell(x: wp.vec2, valid: int, nx: int, ny: int, inv_dx: float) -> int:
    """Cell id of a particle, or -1 if it is invalid, outside the domain or not finite.

    The range test is done on the float coordinates: converting NaN to int gives 0 on CUDA (and
    INT_MIN on the CPU), which would otherwise bin NaN particles into the first row or column.
    """
    if valid == 0:
        return -1
    fx = x[0] * inv_dx
    fy = x[1] * inv_dx
    if fx >= 0.0 and fx < float(nx) and fy >= 0.0 and fy < float(ny):
        return int(wp.floor(fx)) * ny + int(wp.floor(fy))
    return -1


@wp.kernel
def _count(x: wp.array(dtype=wp.vec2), valid: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
           nx: int, ny: int, inv_dx: float):
    p = wp.tid()
    c = _binned_cell(x[p], valid[p], nx, ny, inv_dx)
    if c >= 0:
        wp.atomic_add(counts, c, 1)


@wp.kernel
def _scatter(x: wp.array(dtype=wp.vec2), valid: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
             start: wp.array(dtype=wp.int32), scratch: wp.array(dtype=wp.int32), nx: int, ny: int, inv_dx: float):
    p = wp.tid()
    c = _binned_cell(x[p], valid[p], nx, ny, inv_dx)
    if c >= 0:
        k = wp.atomic_sub(counts, c, 1)  # returns n, n-1, ..., 1 and leaves counts[c] = 0
        scratch[start[c] + k - 1] = p


@wp.kernel
def _place_sorted(x: wp.array(dtype=wp.vec2), valid: wp.array(dtype=wp.int32), start: wp.array(dtype=wp.int32),
                  scratch: wp.array(dtype=wp.int32), ids: wp.array(dtype=wp.int32), nx: int, ny: int, inv_dx: float):
    """ids[start[c] + rank] = p, with rank the number of smaller ids in p's cell."""
    p = wp.tid()
    c = _binned_cell(x[p], valid[p], nx, ny, inv_dx)
    if c < 0:
        return
    s = start[c]
    rank = int(0)
    for k in range(s, start[c + 1]):
        if scratch[k] < p:
            rank += 1
    ids[s + rank] = p


class CellList2D:
    """Cell list for up to ``capacity`` particles on an ``nx x ny`` grid of cell size ``dx``."""

    def __init__(self, nx: int, ny: int, dx: float, capacity: int, device=None):
        self.nx, self.ny, self.dx = int(nx), int(ny), float(dx)
        self.capacity = int(capacity)
        self.device = wp.get_device(device)
        self.num_cells = self.nx * self.ny
        n = self.num_cells
        self._counts = wp.zeros(n + 1, dtype=wp.int32, device=self.device)  # zero between builds
        self.start = wp.zeros(n + 1, dtype=wp.int32, device=self.device)
        self.end = self.start[1:]
        self.ids = wp.zeros(self.capacity, dtype=wp.int32, device=self.device)
        self._scratch = wp.zeros(self.capacity, dtype=wp.int32, device=self.device)

        self.view = CellListView()
        self.view.start = self.start[:n]
        self.view.end = self.end
        self.view.ids = self.ids
        self.view.nx = self.nx
        self.view.ny = self.ny
        self.view.inv_dx = 1.0 / self.dx

    def build(self, x: wp.array, valid: wp.array) -> None:
        """Bin the valid particles of ``x`` (first ``capacity`` entries)."""
        args = [self.nx, self.ny, 1.0 / self.dx]
        dev = self.device
        wp.launch(_count, dim=self.capacity, inputs=[x, valid, self._counts, *args], device=dev)
        wp.utils.array_scan(self._counts, self.start, inclusive=False)
        wp.launch(_scatter, dim=self.capacity, inputs=[x, valid, self._counts, self.start, self._scratch, *args],
                  device=dev)
        wp.launch(_place_sorted, dim=self.capacity, inputs=[x, valid, self.start, self._scratch, self.ids, *args],
                  device=dev)

    def counts(self):
        """Number of binned particles per cell as a host ``(nx, ny)`` array (debug/tests)."""
        s = self.start.numpy()
        return (s[1:] - s[:-1]).reshape(self.nx, self.ny)
