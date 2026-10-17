"""3D grid redistancing used at long reinitialisation: quasi-Newton closest-point projection.

The projection follows the gradient-augmented reinitialisation of Li et al. 2023 (GARM-LS, ACM Trans. Graph.
42(6); see :mod:`pfmls.levelset.newton`), with the level set interpolated from the PFM-LS particles.

Pipeline (:meth:`QuasiNewtonRedistancer3D.run`, operating in place on ``(phi_r, g_r)``):

1. mark interface cells (sign change or ``|phi_I + phi_nb| < dx`` towards an axis neighbour, and
   ``|phi_eik| < 2 dx``) on top of the particle-coverage mask;
2. a few semi-Lagrangian eikonal sweeps (trilinear) on the unmarked cells within ``sweep_width`` of the
   interface;
3. Newton band: the re-marked interface cells dilated ``dilations`` times, united with every cell whose
   eikonal distance is below ``band_width`` (an isotropic band: in 3D the 6-neighbour dilation alone
   reaches only ``dilations / sqrt(3)`` cells along the diagonals);
4. quasi-Newton closest-point projection in the band. The level set and its normal are interpolated
   from the particles (KNN within the reconstruction radius, normal-filtered) with a tricubic Hermite
   fallback; the result is the signed distance to the converged point. The iteration starts at
   ``x_s - phi n``, the tolerance is ``tol dx`` and the iteration count is capped, as in 2D. Each step is
   clamped to a trust radius that starts at ``step_clamp dx``, doubles while consecutive steps keep their
   direction and halves (not below ``step_clamp dx``) when they reverse, up to ``max_step dx``. With a fixed
   clamp of 0.02 dx per step, a start point off the surface by ``e`` needs ``e / (0.02 dx)`` iterations
   (0.35 dx: 18) and a capped iteration stops short of the surface; the trust radius keeps the fixed-clamp
   path near the start point and reaches the surface within the cap. The band cells are compacted
   into a list first (a prefix sum), so the threads of the projection kernel all work.

The band covers every grid node read by the reseeding (Hermite stencils of lattice points within the
particle band), so the 2D code's Hermite eikonal sweeps outside the band are not needed: cells outside
keep their eikonal-redistanced values.
"""

from __future__ import annotations

import warp as wp

from ..core.celllist3d import CellListView3D
from ..core.kernels import sgn
from ..core.kernels3d import cell_center3, hermite3, hermite_cross_terms, safe_normalize3, trilerp
from .particles3d import ParticleView3D, compact_cells, decode_cell, knn_interp3


@wp.func
def _touches(p: float, q: float, s: float, dx: float) -> int:
    if sgn(q) != s or wp.abs(p + q) < dx:
        return 1
    return 0


@wp.kernel
def mark_interface3(phi: wp.array3d(dtype=float), phi_eik: wp.array3d(dtype=float), m: wp.array3d(dtype=wp.int32),
                    dx: float):
    """Set m = 1 next to the interface (does not clear other marks)."""
    i, j, k = wp.tid()
    if wp.abs(phi_eik[i, j, k]) >= 2.0 * dx:
        return
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]
    p = phi[i, j, k]
    s = sgn(p)
    hit = int(0)
    if i > 0:
        hit += _touches(p, phi[i - 1, j, k], s, dx)
    if i < nx - 1:
        hit += _touches(p, phi[i + 1, j, k], s, dx)
    if j > 0:
        hit += _touches(p, phi[i, j - 1, k], s, dx)
    if j < ny - 1:
        hit += _touches(p, phi[i, j + 1, k], s, dx)
    if k > 0:
        hit += _touches(p, phi[i, j, k - 1], s, dx)
    if k < nz - 1:
        hit += _touches(p, phi[i, j, k + 1], s, dx)
    if hit > 0:
        m[i, j, k] = 1


@wp.kernel
def dilate3(src: wp.array3d(dtype=wp.int32), dst: wp.array3d(dtype=wp.int32)):
    """dst = src grown by one cell in the 6-neighbourhood."""
    i, j, k = wp.tid()
    nx = src.shape[0]
    ny = src.shape[1]
    nz = src.shape[2]
    v = src[i, j, k]
    if v == 0:
        if i > 0 and src[i - 1, j, k] == 1:
            v = 1
        if i < nx - 1 and src[i + 1, j, k] == 1:
            v = 1
        if j > 0 and src[i, j - 1, k] == 1:
            v = 1
        if j < ny - 1 and src[i, j + 1, k] == 1:
            v = 1
        if k > 0 and src[i, j, k - 1] == 1:
            v = 1
        if k < nz - 1 and src[i, j, k + 1] == 1:
            v = 1
    dst[i, j, k] = v


@wp.kernel
def mark_distance_band(phi_eik: wp.array3d(dtype=float), width: float, m: wp.array3d(dtype=wp.int32)):
    """Set m = 1 where |phi_eik| < width (does not clear other marks)."""
    i, j, k = wp.tid()
    if wp.abs(phi_eik[i, j, k]) < width:
        m[i, j, k] = 1


@wp.kernel
def sl_eikonal_sweep3(
    phi: wp.array3d(dtype=float),
    g: wp.array3d(dtype=wp.vec3),
    m: wp.array3d(dtype=wp.int32),
    phi_eik: wp.array3d(dtype=float),
    width: float,
    phi_out: wp.array3d(dtype=float),
    g_out: wp.array3d(dtype=wp.vec3),
    dx: float,
    step: float,
):
    """One Jacobi sweep of characteristic-marching redistancing (trilinear) on cells with m == 0 and
    ``|phi_eik| < width``: ``x_f = x - sigma h n`` (RK4 along the normal), ``phi(x) = phi(x_f) + sigma h``."""
    i, j, k = wp.tid()
    if m[i, j, k] != 0 or wp.abs(phi_eik[i, j, k]) >= width:
        phi_out[i, j, k] = phi[i, j, k]
        g_out[i, j, k] = g[i, j, k]
        return
    sigma = sgn(phi[i, j, k])
    h = step * dx
    xc = cell_center3(i, j, k, dx)
    half = wp.vec3(0.5, 0.5, 0.5)
    n1 = safe_normalize3(trilerp(g, xc / dx - half))
    n2 = safe_normalize3(trilerp(g, (xc - 0.5 * sigma * h * n1) / dx - half))
    n3 = safe_normalize3(trilerp(g, (xc - 0.5 * sigma * h * n2) / dx - half))
    n4 = safe_normalize3(trilerp(g, (xc - sigma * h * n3) / dx - half))
    xf = xc - sigma * h * (n1 + 2.0 * n2 + 2.0 * n3 + n4) / 6.0
    q = xf / dx - half
    phi_out[i, j, k] = trilerp(phi, q) + sigma * h
    g_out[i, j, k] = safe_normalize3(trilerp(g, q))


@wp.func
def _particle_or_hermite(x: wp.vec3, phi: wp.array3d(dtype=float), g: wp.array3d(dtype=wp.vec3),
                         X: wp.array3d(dtype=wp.vec4), parts: ParticleView3D, cells: CellListView3D, dx: float):
    """(phi, normal) from the normal-filtered particle KNN, falling back to tricubic Hermite."""
    q = x / dx - wp.vec3(0.5, 0.5, 0.5)
    v_h, grad_h = hermite3(phi, g, X, q, dx)
    n_h = safe_normalize3(grad_h)
    val, n, ok = knn_interp3(x, parts, cells, n_h, 1)
    if ok == 0:
        val = v_h
        n = n_h
    return val, n


@wp.func
def _project_cell(
    i: int,
    j: int,
    k: int,
    phi: wp.array3d(dtype=float),
    g: wp.array3d(dtype=wp.vec3),
    X: wp.array3d(dtype=wp.vec4),
    parts: ParticleView3D,
    cells: CellListView3D,
    dx: float,
    step_clamp: float,
    max_step: float,
    tol: float,
    max_iter: int,
):
    """Signed distance and unit normal of cell ``(i, j, k)`` from its quasi-Newton closest point."""
    xs = cell_center3(i, j, k, dx)
    x = xs - phi[i, j, k] * safe_normalize3(g[i, j, k])
    x_prev = wp.vec3(0.0, 0.0, 0.0)
    d_prev = wp.vec3(0.0, 0.0, 0.0)
    d_last = wp.vec3(0.0, 0.0, 0.0)
    r_min = step_clamp * dx
    r_max = wp.max(max_step * dx, r_min)
    r = r_min
    tol_abs = tol * dx
    for it in range(max_iter):
        val, psi = _particle_or_hermite(x, phi, g, X, parts, cells, dx)
        pn2 = wp.dot(psi, psi)
        if pn2 < 1.0e-12:
            break
        e = xs - x
        delta = -val * psi / pn2 + e - psi * (wp.dot(e, psi) / pn2)
        c = wp.dot(delta, d_last)  # trust radius: doubled while the steps keep their direction, halved on reversal
        if c < 0.0:
            r = wp.max(0.5 * r, r_min)
        elif c > 0.0:
            r = wp.min(2.0 * r, r_max)
        d_last = delta
        n = wp.length(delta)
        if n > r:
            delta = delta * (r / n)
            n = r
        if wp.length(delta + d_prev) < 0.1 * n:
            mid = 0.5 * (x + x_prev)
            x_prev = x
            x = mid
            d_prev = wp.vec3(0.0, 0.0, 0.0)
        else:
            x_prev = x
            d_prev = delta
            x = x + delta
        if n < tol_abs:
            break
    s0, n0, ok0 = knn_interp3(xs, parts, cells, wp.vec3(0.0, 0.0, 0.0), 0)
    if ok0 == 0:
        s0, gr0 = hermite3(phi, g, X, xs / dx - wp.vec3(0.5, 0.5, 0.5), dx)
    sigma = sgn(s0)
    e = xs - x
    dist = wp.length(e)
    val, psi = _particle_or_hermite(x, phi, g, X, parts, cells, dx)
    gn = safe_normalize3(psi)
    if dist > 1.0e-4 * dx:
        gn = sigma * (e / dist)
        if wp.dot(gn, psi) < 0.0:
            gn = -gn
    return sigma * dist, gn


@wp.kernel
def newton_closest_point3(
    band: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    g: wp.array3d(dtype=wp.vec3),
    X: wp.array3d(dtype=wp.vec4),
    parts: ParticleView3D,
    cells: CellListView3D,
    dx: float,
    step_clamp: float,
    max_step: float,
    tol: float,
    max_iter: int,
    phi_out: wp.array3d(dtype=float),
    g_out: wp.array3d(dtype=wp.vec3),
):
    """Quasi-Newton projection of the cell centres ``band[0 .. count - 1]`` (flat ids) onto {phi = 0}.

    Iteration (``psi`` the interpolated normal, ``e = x_s - x``)::

        delta = -phi psi / |psi|^2 + e - psi (e . psi) / |psi|^2,   |delta| clamped to the trust radius

    The trust radius starts at ``step_clamp dx``, doubles while consecutive steps point the same way and halves
    (not below ``step_clamp dx``) when they reverse, up to ``max_step dx`` (``max_step <= step_clamp``: a
    fixed clamp). With back-and-forth damping (midpoint when two
    consecutive steps nearly cancel). Starts at ``x_s - phi n`` and stops when ``|delta| < tol dx`` or after
    ``max_iter`` iterations.
    """
    t = wp.tid()
    if t >= count[0]:
        return
    i, j, k = decode_cell(band[t], phi.shape[1], phi.shape[2])
    v, n = _project_cell(i, j, k, phi, g, X, parts, cells, dx, step_clamp, max_step, tol, max_iter)
    phi_out[i, j, k] = v
    g_out[i, j, k] = n


class QuasiNewtonRedistancer3D:
    """Scratch buffers and launch sequence for the 3D long-reinitialisation redistancing."""

    def __init__(self, nx: int, ny: int, nz: int, dx: float, device=None, dilations: int = 3,
                 band_width: float = 4.0, linear_sweeps: int = 5, sweep_width: float = 6.0, sl_step: float = 0.9,
                 max_iter: int = 16, step_clamp: float = 0.02, max_step: float = 1.0, tol: float = 1.0e-4):
        self.nx, self.ny, self.nz, self.dx = int(nx), int(ny), int(nz), float(dx)
        self.device = wp.get_device(device)
        self.dilations = int(dilations)
        self.band_width = float(band_width)
        self.linear_sweeps = int(linear_sweeps)
        self.sweep_width = float(sweep_width)
        self.sl_step = float(sl_step)
        self.max_iter = int(max_iter)
        self.step_clamp = float(step_clamp)
        self.max_step = float(max_step)
        self.tol = float(tol)
        shape = (self.nx, self.ny, self.nz)
        self._phi = wp.zeros(shape, dtype=float, device=self.device)
        self._g = wp.zeros(shape, dtype=wp.vec3, device=self.device)
        self._m = wp.zeros(shape, dtype=wp.int32, device=self.device)
        n = self.nx * self.ny * self.nz
        self._offsets = wp.zeros(n, dtype=wp.int32, device=self.device)
        self._band = wp.zeros(n, dtype=wp.int32, device=self.device)
        self._count = wp.zeros(1, dtype=wp.int32, device=self.device)

    def _launch(self, kernel, inputs):
        wp.launch(kernel, dim=(self.nx, self.ny, self.nz), inputs=inputs, device=self.device)

    def _sweeps(self, phi, g, m, phi_eik) -> None:
        a_phi, a_g, b_phi, b_g = phi, g, self._phi, self._g
        for _ in range(self.linear_sweeps):
            self._launch(sl_eikonal_sweep3, [a_phi, a_g, m, phi_eik, self.sweep_width * self.dx, b_phi, b_g,
                                             self.dx, self.sl_step])
            a_phi, a_g, b_phi, b_g = b_phi, b_g, a_phi, a_g
        if a_phi is not phi:
            wp.copy(phi, a_phi)
            wp.copy(g, a_g)

    def run(self, phi: wp.array, g: wp.array, X: wp.array, m: wp.array, phi_eik: wp.array,
            parts: ParticleView3D, cells: CellListView3D) -> None:
        """Redistance ``(phi, g)`` in place and leave the Hermite cross terms of the result in ``X``.

        ``m`` holds the particle-coverage marks on entry and the Newton band on exit; ``phi_eik`` is the
        eikonal-redistanced initial guess.
        """
        self._launch(mark_interface3, [phi, phi_eik, m, self.dx])
        self._sweeps(phi, g, m, phi_eik)
        m.zero_()
        self._launch(mark_interface3, [phi, phi_eik, m, self.dx])
        src, dst = m, self._m
        for _ in range(self.dilations):
            self._launch(dilate3, [src, dst])
            src, dst = dst, src
        if src is not m:
            wp.copy(m, src)
        self._launch(mark_distance_band, [phi_eik, self.band_width * self.dx, m])
        self._launch(hermite_cross_terms, [g, X, self.dx])
        n = self.nx * self.ny * self.nz
        flags = m.flatten()
        wp.utils.array_scan(flags, self._offsets, inclusive=False)
        wp.launch(compact_cells, dim=n, inputs=[flags, self._offsets, self._band, self._count, n], device=self.device)
        wp.copy(self._phi, phi)
        wp.copy(self._g, g)
        wp.launch(newton_closest_point3, dim=n, inputs=[self._band, self._count, phi, g, X, parts, cells, self.dx,
                                                         self.step_clamp, self.max_step, self.tol, self.max_iter,
                                                         self._phi, self._g],
                  device=self.device)
        wp.copy(phi, self._phi)
        wp.copy(g, self._g)
        self._launch(hermite_cross_terms, [g, X, self.dx])
