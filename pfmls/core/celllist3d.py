"""3D particle binning into grid cells by a counting sort over the valid particles (no per-cell cap).

Same algorithm as :mod:`pfmls.core.celllist`: an atomic histogram of the binned particles, an
exclusive scan into ``start`` (``num_cells + 1`` entries, so cell ``c`` holds ``ids[start[c]:start[c+1]]``),
an atomic scatter of the particle ids into a scratch list (which returns the histogram to zero), and a
parallel placement in which each particle writes itself at the rank of its id within its cell, so
particles inside a cell are listed by increasing id (deterministic neighbour sums). Non-finite positions
are not binned. Cell ids are ``c = (ci * ny + cj) * nz + ck``. Every launch is asynchronous and
graph-capturable.
"""

from __future__ import annotations

import warp as wp


@wp.struct
class CellListView3D:
    """Device view of a 3D cell list, passed by value to kernels."""

    start: wp.array(dtype=wp.int32)
    end: wp.array(dtype=wp.int32)
    ids: wp.array(dtype=wp.int32)
    nx: int
    ny: int
    nz: int
    inv_dx: float


@wp.func
def cell_of3(x: wp.vec3, inv_dx: float):
    """Integer cell coordinates ``floor(x / dx)``."""
    return int(wp.floor(x[0] * inv_dx)), int(wp.floor(x[1] * inv_dx)), int(wp.floor(x[2] * inv_dx))


@wp.func
def cell_id3(cells: CellListView3D, ci: int, cj: int, ck: int) -> int:
    return (ci * cells.ny + cj) * cells.nz + ck


@wp.func
def cell_range3(x: wp.vec3, r: float, cells: CellListView3D):
    """Inclusive cell-coordinate ranges ``(i0, i1, j0, j1, k0, k1)`` of the cells that overlap the cube of
    half-width ``r`` around ``x``, clamped to the grid."""
    i0 = wp.max(int(wp.floor((x[0] - r) * cells.inv_dx)), 0)
    i1 = wp.min(int(wp.floor((x[0] + r) * cells.inv_dx)), cells.nx - 1)
    j0 = wp.max(int(wp.floor((x[1] - r) * cells.inv_dx)), 0)
    j1 = wp.min(int(wp.floor((x[1] + r) * cells.inv_dx)), cells.ny - 1)
    k0 = wp.max(int(wp.floor((x[2] - r) * cells.inv_dx)), 0)
    k1 = wp.min(int(wp.floor((x[2] + r) * cells.inv_dx)), cells.nz - 1)
    return i0, i1, j0, j1, k0, k1


@wp.func
def _binned_cell(x: wp.vec3, valid: int, nx: int, ny: int, nz: int, inv_dx: float) -> int:
    """Cell id of a particle, or -1 if it is invalid, outside the domain or not finite (the range test is
    done in floating point: int(NaN) is 0 on CUDA)."""
    if valid == 0:
        return -1
    fx = x[0] * inv_dx
    fy = x[1] * inv_dx
    fz = x[2] * inv_dx
    if fx >= 0.0 and fx < float(nx) and fy >= 0.0 and fy < float(ny) and fz >= 0.0 and fz < float(nz):
        return (int(wp.floor(fx)) * ny + int(wp.floor(fy))) * nz + int(wp.floor(fz))
    return -1


@wp.kernel
def _count(x: wp.array(dtype=wp.vec3), valid: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
           nx: int, ny: int, nz: int, inv_dx: float):
    p = wp.tid()
    c = _binned_cell(x[p], valid[p], nx, ny, nz, inv_dx)
    if c >= 0:
        wp.atomic_add(counts, c, 1)


@wp.kernel
def _scatter(x: wp.array(dtype=wp.vec3), valid: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
             start: wp.array(dtype=wp.int32), scratch: wp.array(dtype=wp.int32), nx: int, ny: int, nz: int,
             inv_dx: float):
    p = wp.tid()
    c = _binned_cell(x[p], valid[p], nx, ny, nz, inv_dx)
    if c >= 0:
        k = wp.atomic_sub(counts, c, 1)  # returns n, n-1, ..., 1 and leaves counts[c] = 0
        scratch[start[c] + k - 1] = p


@wp.kernel
def _place_sorted(x: wp.array(dtype=wp.vec3), valid: wp.array(dtype=wp.int32), start: wp.array(dtype=wp.int32),
                  scratch: wp.array(dtype=wp.int32), ids: wp.array(dtype=wp.int32), nx: int, ny: int, nz: int,
                  inv_dx: float):
    """ids[start[c] + rank] = p, with rank the number of smaller ids in p's cell."""
    p = wp.tid()
    c = _binned_cell(x[p], valid[p], nx, ny, nz, inv_dx)
    if c < 0:
        return
    s = start[c]
    rank = int(0)
    for k in range(s, start[c + 1]):
        if scratch[k] < p:
            rank += 1
    ids[s + rank] = p


class CellList3D:
    """Cell list for up to ``capacity`` particles on an ``nx x ny x nz`` grid of cell size ``dx``."""

    def __init__(self, nx: int, ny: int, nz: int, dx: float, capacity: int, device=None):
        self.nx, self.ny, self.nz, self.dx = int(nx), int(ny), int(nz), float(dx)
        self.device = wp.get_device(device)
        self.num_cells = self.nx * self.ny * self.nz
        n = self.num_cells
        self._counts = wp.zeros(n + 1, dtype=wp.int32, device=self.device)  # zero between builds
        self.start = wp.zeros(n + 1, dtype=wp.int32, device=self.device)
        self.end = self.start[1:]
        self.view = CellListView3D()
        self.view.start = self.start[:n]
        self.view.end = self.end
        self.view.nx = self.nx
        self.view.ny = self.ny
        self.view.nz = self.nz
        self.view.inv_dx = 1.0 / self.dx
        self.resize(capacity)

    def allocate(self, capacity: int):
        """New id buffers ``(ids, scratch)`` for ``capacity`` particles (install them with :meth:`set_buffers`;
        allocating first keeps the list intact if the allocation fails)."""
        n = int(capacity)
        return wp.zeros(n, dtype=wp.int32, device=self.device), wp.zeros(n, dtype=wp.int32, device=self.device)

    def set_buffers(self, ids: wp.array, scratch: wp.array) -> None:
        """Install id buffers from :meth:`allocate` (the list must be rebuilt)."""
        self.capacity = int(ids.shape[0])
        self.ids, self._scratch = ids, scratch
        self.view.ids = self.ids

    def resize(self, capacity: int) -> None:
        """Reallocate the id buffers for ``capacity`` particles (the list must be rebuilt)."""
        self.set_buffers(*self.allocate(capacity))

    def memory_bytes(self) -> tuple[int, int]:
        """Device bytes held per particle capacity (id lists) and per grid (cell ranges and counts)."""
        return self.ids.capacity + self._scratch.capacity, self.start.capacity + self._counts.capacity

    def build(self, x: wp.array, valid: wp.array) -> None:
        """Bin the valid particles of ``x`` (first ``capacity`` entries)."""
        args = [self.nx, self.ny, self.nz, 1.0 / self.dx]
        dev = self.device
        wp.launch(_count, dim=self.capacity, inputs=[x, valid, self._counts, *args], device=dev)
        wp.utils.array_scan(self._counts, self.start, inclusive=False)
        wp.launch(_scatter, dim=self.capacity, inputs=[x, valid, self._counts, self.start, self._scratch, *args],
                  device=dev)
        wp.launch(_place_sorted, dim=self.capacity, inputs=[x, valid, self.start, self._scratch, self.ids, *args],
                  device=dev)

    def counts(self):
        """Number of binned particles per cell as a host ``(nx, ny, nz)`` array (debug/tests)."""
        s = self.start.numpy()
        return (s[1:] - s[:-1]).reshape(self.nx, self.ny, self.nz)
