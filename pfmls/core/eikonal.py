"""GPU-parallel eikonal redistancing (Jacobi iterations of the fast-marching update).

The discretisation is the first-order upwind quadratic update of the fast marching method (FMM)::

    a <= b are the smallest |T| of the two x-neighbours and of the two y-neighbours
    d = a + dx;  if d > b:  d = (a + b + sqrt(2 dx^2 - (b - a)^2)) / 2
    T_new = sign(phi) * min(|T_old|, d)

Initialisation is identical to the FMM's surface cells: a cell with a sign change towards an
axis neighbour gets ``sign(phi) * theta * dx`` with ``theta = phi_I / (phi_I - phi_nb)`` (smallest
magnitude over its neighbours) and is frozen; all other cells start at ``sign(phi) * BIG``.

The update is monotone (it only decreases ``|T|`` and every iterate is an upper bound of the
discrete solution), so parallel Jacobi iterations converge to the same fixed point as the FMM
would without its first-touch freezing. After ``k`` iterations every cell whose upwind dependency
chain is at most ``k`` cells long (roughly: within ``k`` cells in the L1 metric of the interface)
holds its converged value; ``nx + ny`` iterations converge on the whole grid. With fewer
iterations the result is a narrow-band redistancing: cells at the edge of the reached region hold
an upper bound of their distance and cells never reached keep their input value. The sign of the
input is always kept. Iterations run two per kernel launch (each thread recomputes the first
iteration at its four neighbours; bitwise identical to two single launches), plus one single
iteration for an odd count.
"""

from __future__ import annotations

import warp as wp

from .kernels import sgn

BIG = 1.0e10


@wp.func
def axis_crossing(p: float, q: float, s: float, dx: float, est: float) -> float:
    """Distance estimate ``s theta dx`` of a cell with value ``p`` and sign ``s`` from the sign change towards a
    neighbour with value ``q`` (``theta = p / (p - q)``), if smaller in magnitude than ``est``."""
    if sgn(q) != s:
        e = s * (p / (p - q)) * dx
        if wp.abs(e) < wp.abs(est):
            return e
    return est


@wp.kernel
def _init_surface(
    phi: wp.array2d(dtype=float),
    T: wp.array2d(dtype=float),
    frozen: wp.array2d(dtype=wp.int32),
    dx: float,
):
    i, j = wp.tid()
    nx = phi.shape[0]
    ny = phi.shape[1]
    p = phi[i, j]
    s = sgn(p)
    est = float(BIG)
    if i > 0:
        est = axis_crossing(p, phi[i - 1, j], s, dx, est)
    if i < nx - 1:
        est = axis_crossing(p, phi[i + 1, j], s, dx, est)
    if j > 0:
        est = axis_crossing(p, phi[i, j - 1], s, dx, est)
    if j < ny - 1:
        est = axis_crossing(p, phi[i, j + 1], s, dx, est)
    if wp.abs(est) < 0.5 * BIG:
        T[i, j] = est
        frozen[i, j] = 1
    else:
        T[i, j] = s * BIG
        frozen[i, j] = 0


@wp.func
def eikonal_update(a: float, b: float, dx: float) -> float:
    """First-order upwind solution of |grad T| = 1 from the axis minima a, b."""
    lo = wp.min(a, b)
    hi = wp.max(a, b)
    d = lo + dx
    if d > hi:
        d = 0.5 * (lo + hi + wp.sqrt(wp.max(2.0 * dx * dx - (hi - lo) * (hi - lo), 0.0)))
    return d


@wp.func
def _relax(cur: float, a: float, b: float, dx: float) -> float:
    d = eikonal_update(a, b, dx)
    if d < wp.abs(cur):
        return sgn(cur) * d
    return cur


@wp.func
def _jacobi_value(src: wp.array2d(dtype=float), frozen: wp.array2d(dtype=wp.int32), i: int, j: int, dx: float):
    """One Jacobi update of cell (i, j) from ``src``."""
    nx = src.shape[0]
    ny = src.shape[1]
    cur = src[i, j]
    if frozen[i, j] == 0:
        ax = float(BIG)
        ay = float(BIG)
        if i > 0:
            ax = wp.abs(src[i - 1, j])
        if i < nx - 1:
            ax = wp.min(ax, wp.abs(src[i + 1, j]))
        if j > 0:
            ay = wp.abs(src[i, j - 1])
        if j < ny - 1:
            ay = wp.min(ay, wp.abs(src[i, j + 1]))
        cur = _relax(cur, ax, ay, dx)
    return cur


@wp.kernel
def _jacobi(
    src: wp.array2d(dtype=float),
    dst: wp.array2d(dtype=float),
    frozen: wp.array2d(dtype=wp.int32),
    dx: float,
):
    i, j = wp.tid()
    dst[i, j] = _jacobi_value(src, frozen, i, j, dx)


@wp.kernel
def _jacobi2(
    src: wp.array2d(dtype=float),
    dst: wp.array2d(dtype=float),
    frozen: wp.array2d(dtype=wp.int32),
    dx: float,
):
    """Two Jacobi iterations; the first is evaluated at the cell and at its four neighbours."""
    i, j = wp.tid()
    nx = src.shape[0]
    ny = src.shape[1]
    cur = _jacobi_value(src, frozen, i, j, dx)
    if frozen[i, j] == 0:
        ax = float(BIG)
        ay = float(BIG)
        if i > 0:
            ax = wp.abs(_jacobi_value(src, frozen, i - 1, j, dx))
        if i < nx - 1:
            ax = wp.min(ax, wp.abs(_jacobi_value(src, frozen, i + 1, j, dx)))
        if j > 0:
            ay = wp.abs(_jacobi_value(src, frozen, i, j - 1, dx))
        if j < ny - 1:
            ay = wp.min(ay, wp.abs(_jacobi_value(src, frozen, i, j + 1, dx)))
        cur = _relax(cur, ax, ay, dx)
    dst[i, j] = cur


@wp.kernel
def _finalize(T: wp.array2d(dtype=float), phi_in: wp.array2d(dtype=float), phi_out: wp.array2d(dtype=float)):
    i, j = wp.tid()
    t = T[i, j]
    if wp.abs(t) < 0.5 * BIG:
        phi_out[i, j] = t
    else:
        phi_out[i, j] = phi_in[i, j]


class Eikonal2D:
    """Reusable Jacobi redistancing solver for ``(nx, ny)`` cell-centred fields.

    Differences from :class:`pfmls.core.eikonal3d.Eikonal3D`: the iterations run two per launch (in 2D the
    recomputation at the four neighbours is cheap; in 3D it would cost six), and only the axis-crossing
    interface initialisation is offered, the one the 2D benchmarks are validated with (the 3D solver adds
    a gradient-based initialisation because the crossing estimate's oblique error grows to ``sqrt(3)`` there).

    Parameters
    ----------
    iterations:
        Number of Jacobi iterations; ``None`` uses ``nx + ny`` (converged on the whole grid).
    """

    def __init__(self, nx: int, ny: int, dx: float, device=None, iterations: int | None = None):
        self.nx, self.ny, self.dx = int(nx), int(ny), float(dx)
        self.device = wp.get_device(device)
        self.iterations = self.nx + self.ny if iterations is None else int(iterations)
        shape = (self.nx, self.ny)
        self._a = wp.zeros(shape, dtype=float, device=self.device)
        self._b = wp.zeros(shape, dtype=float, device=self.device)
        self._frozen = wp.zeros(shape, dtype=wp.int32, device=self.device)

    def redistance(self, phi_in: wp.array, phi_out: wp.array, iterations: int | None = None) -> None:
        """Write the signed distance of the zero level set of ``phi_in`` into ``phi_out``.

        ``phi_in`` and ``phi_out`` may be the same array. ``iterations`` overrides the solver's
        default count for this call.
        """
        shape = (self.nx, self.ny)
        a, b = self._a, self._b
        count = self.iterations if iterations is None else int(iterations)
        wp.launch(_init_surface, dim=shape, inputs=[phi_in, a, self._frozen, self.dx], device=self.device)
        pairs, odd = divmod(count, 2)
        for kernel in [_jacobi2] * pairs + [_jacobi] * odd:
            wp.launch(kernel, dim=shape, inputs=[a, b, self._frozen, self.dx], device=self.device)
            a, b = b, a
        wp.launch(_finalize, dim=shape, inputs=[a, phi_in, phi_out], device=self.device)
