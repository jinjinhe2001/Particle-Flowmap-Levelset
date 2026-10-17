"""GPU-parallel eikonal redistancing on 3D cell-centred grids (Jacobi iterations).

The update is the first-order upwind solution of ``|grad T| = 1`` used by the fast
marching method, with ``a <= b <= c`` the smallest ``|T|`` of the two neighbours along each axis::

    d = a + dx
    if d > b:  d = (a + b + sqrt(2 dx^2 - (b - a)^2)) / 2
    if d > c:  d = (a + b + c + sqrt((a + b + c)^2 - 3 (a^2 + b^2 + c^2 - dx^2))) / 3
    T_new = sign(phi) * min(|T_old|, d)

Interface cells (a sign change towards an axis neighbour) are initialised and frozen; all other cells
start at ``sign(phi) * BIG``. Two initialisations are available:

* ``"crossing"`` (as in 2D): ``sign(phi) theta dx`` with ``theta = phi_I / (phi_I - phi_nb)``, the
  smallest axis crossing. It overestimates the distance for oblique interfaces (by up to ``sqrt(3)``).
* ``"gradient"`` (default): ``min(|crossing|, |phi_I| / |grad phi_I|)`` with a central-difference
  gradient. Exact for a linear field and the identity for a signed distance; the crossing estimate
  remains as a bound where the central differences cancel (thin features).

The iteration is monotone and converges to the same fixed point as fast marching without first-touch
freezing. After ``k`` iterations the cells whose upwind chain is at most ``k`` cells long hold their
converged value; cells never reached keep their input value (the input sign is always kept), so a
small iteration count is a narrow-band redistancing.
"""

from __future__ import annotations

import warp as wp

from .eikonal import BIG, axis_crossing
from .kernels import sgn

#: Interface initialisation modes accepted by :class:`Eikonal3D`.
INIT_MODES = {"crossing": 0, "gradient": 1}


@wp.func
def _central(lo: float, hi: float, span: int, dx: float) -> float:
    """Difference quotient over ``span`` cells (0 for a degenerate axis)."""
    if span == 0:
        return 0.0
    return (hi - lo) / (float(span) * dx)


@wp.kernel
def _init_surface(
    phi: wp.array3d(dtype=float),
    T: wp.array3d(dtype=float),
    frozen: wp.array3d(dtype=wp.uint8),
    dx: float,
    mode: int,
):
    i, j, k = wp.tid()
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]
    p = phi[i, j, k]
    s = sgn(p)
    est = float(BIG)
    if i > 0:
        est = axis_crossing(p, phi[i - 1, j, k], s, dx, est)
    if i < nx - 1:
        est = axis_crossing(p, phi[i + 1, j, k], s, dx, est)
    if j > 0:
        est = axis_crossing(p, phi[i, j - 1, k], s, dx, est)
    if j < ny - 1:
        est = axis_crossing(p, phi[i, j + 1, k], s, dx, est)
    if k > 0:
        est = axis_crossing(p, phi[i, j, k - 1], s, dx, est)
    if k < nz - 1:
        est = axis_crossing(p, phi[i, j, k + 1], s, dx, est)
    if wp.abs(est) < 0.5 * BIG:
        if mode == 1:
            ip = wp.min(i + 1, nx - 1)
            im = wp.max(i - 1, 0)
            jp = wp.min(j + 1, ny - 1)
            jm = wp.max(j - 1, 0)
            kp = wp.min(k + 1, nz - 1)
            km = wp.max(k - 1, 0)
            gx = _central(phi[im, j, k], phi[ip, j, k], ip - im, dx)
            gy = _central(phi[i, jm, k], phi[i, jp, k], jp - jm, dx)
            gz = _central(phi[i, j, km], phi[i, j, kp], kp - km, dx)
            gn = wp.sqrt(gx * gx + gy * gy + gz * gz)
            if gn * wp.abs(est) > wp.abs(p):
                est = p / gn
        T[i, j, k] = est
        frozen[i, j, k] = wp.uint8(1)
    else:
        T[i, j, k] = s * BIG
        frozen[i, j, k] = wp.uint8(0)


@wp.func
def eikonal_update3(a: float, b: float, c: float, dx: float) -> float:
    """First-order upwind solution of |grad T| = 1 from the three axis minima."""
    lo = wp.min(a, wp.min(b, c))
    hi = wp.max(a, wp.max(b, c))
    mid = wp.max(wp.min(a, b), wp.min(wp.max(a, b), c))  # median (no cancellation against BIG)
    d = lo + dx
    if d > mid:
        d = 0.5 * (lo + mid + wp.sqrt(wp.max(2.0 * dx * dx - (mid - lo) * (mid - lo), 0.0)))
        if d > hi:
            sm = lo + mid + hi
            disc = sm * sm - 3.0 * (lo * lo + mid * mid + hi * hi - dx * dx)
            d = (sm + wp.sqrt(wp.max(disc, 0.0))) / 3.0
    return d


@wp.kernel
def _jacobi(
    src: wp.array3d(dtype=float),
    dst: wp.array3d(dtype=float),
    frozen: wp.array3d(dtype=wp.uint8),
    dx: float,
):
    i, j, k = wp.tid()
    cur = src[i, j, k]
    if frozen[i, j, k] == wp.uint8(0):
        nx = src.shape[0]
        ny = src.shape[1]
        nz = src.shape[2]
        ax = float(BIG)
        ay = float(BIG)
        az = float(BIG)
        if i > 0:
            ax = wp.abs(src[i - 1, j, k])
        if i < nx - 1:
            ax = wp.min(ax, wp.abs(src[i + 1, j, k]))
        if j > 0:
            ay = wp.abs(src[i, j - 1, k])
        if j < ny - 1:
            ay = wp.min(ay, wp.abs(src[i, j + 1, k]))
        if k > 0:
            az = wp.abs(src[i, j, k - 1])
        if k < nz - 1:
            az = wp.min(az, wp.abs(src[i, j, k + 1]))
        d = eikonal_update3(ax, ay, az, dx)
        if d < wp.abs(cur):
            cur = sgn(cur) * d
    dst[i, j, k] = cur


@wp.kernel
def _finalize(T: wp.array3d(dtype=float), phi_in: wp.array3d(dtype=float), phi_out: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    t = T[i, j, k]
    if wp.abs(t) < 0.5 * BIG:
        phi_out[i, j, k] = t
    else:
        phi_out[i, j, k] = phi_in[i, j, k]


class Eikonal3D:
    """Reusable Jacobi redistancing solver for ``(nx, ny, nz)`` cell-centred fields (one iteration per launch;
    see :class:`pfmls.core.eikonal.Eikonal2D` for the differences from the 2D solver).

    Parameters
    ----------
    iterations:
        Number of Jacobi iterations; ``None`` uses ``nx + ny + nz`` (converged on the whole grid).
    init:
        Interface initialisation, ``"gradient"`` or ``"crossing"`` (see the module docstring).
    """

    def __init__(self, nx: int, ny: int, nz: int, dx: float, device=None, iterations: int | None = None,
                 init: str = "gradient"):
        if init not in INIT_MODES:
            raise ValueError(f"init must be one of {sorted(INIT_MODES)}, got {init!r}")
        self.nx, self.ny, self.nz, self.dx = int(nx), int(ny), int(nz), float(dx)
        self.device = wp.get_device(device)
        self.iterations = self.nx + self.ny + self.nz if iterations is None else int(iterations)
        self.init = init
        shape = (self.nx, self.ny, self.nz)
        self._a = wp.zeros(shape, dtype=float, device=self.device)
        self._b = wp.zeros(shape, dtype=float, device=self.device)
        self._frozen = wp.zeros(shape, dtype=wp.uint8, device=self.device)

    def redistance(self, phi_in: wp.array, phi_out: wp.array, iterations: int | None = None) -> None:
        """Write the signed distance of the zero level set of ``phi_in`` into ``phi_out``.

        ``phi_in`` and ``phi_out`` may be the same array. ``iterations`` overrides the solver's default
        count for this call.
        """
        shape = (self.nx, self.ny, self.nz)
        a, b = self._a, self._b
        count = self.iterations if iterations is None else int(iterations)
        wp.launch(_init_surface, dim=shape, inputs=[phi_in, a, self._frozen, self.dx, INIT_MODES[self.init]],
                  device=self.device)
        for _ in range(count):
            wp.launch(_jacobi, dim=shape, inputs=[a, b, self._frozen, self.dx], device=self.device)
            a, b = b, a
        wp.launch(_finalize, dim=shape, inputs=[a, phi_in, phi_out], device=self.device)
