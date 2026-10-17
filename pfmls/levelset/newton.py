"""Grid redistancing used at long reinitialisation: quasi-Newton closest-point projection.

The projection follows the gradient-augmented reinitialisation of X. Li, X. Ni, B. Zhu, B. Wang and B. Chen,
"GARM-LS: A Gradient-Augmented Reference-Map Method for Level-Set Fluid Simulation", ACM Trans. Graph. 42(6),
2023 (the method, not its code), with the level set interpolated from the PFM-LS particles.

Pipeline (:meth:`QuasiNewtonRedistancer.run`, operating in place on ``(phi_r, g_r)``):

1. mark interface cells (sign change or ``|phi_I + phi_nb| < dx`` towards a neighbour, and
   ``|phi_eik| < 2 dx``) on top of the particle-coverage mask;
2. a few semi-Lagrangian eikonal sweeps (bilinear) on the unmarked cells;
3. re-mark the interface cells and dilate the band ``dilations`` times;
4. quasi-Newton closest-point projection in the band: the level set and its normal are interpolated
   from the particles (KNN within the reconstruction radius, normal-filtered) with a Hermite
   fallback; the result is the signed distance to the converged point. The iteration starts at
   ``x_s - phi n`` and stops at the tolerance ``tol dx`` (float32-safe) or at the iteration cap
   (default 64);
5. semi-Lagrangian eikonal sweeps (Hermite) on the cells outside the band.

The semi-Lagrangian eikonal sweep marches along the normal: with ``h = sl_step dx`` and
``sigma = sign(phi)``, ``x_f = x - sigma h n`` (RK4 along the normal field) and
``phi(x) = phi(x_f) + sigma h``.
"""

from __future__ import annotations

import warp as wp

from ..core.celllist import CellListView
from ..core.kernels import bilerp, hermite_grad, hermite_value, safe_normalize, sgn
from .particles2d import ParticleView, knn_interp


@wp.kernel
def mark_interface(
    phi: wp.array2d(dtype=float),
    phi_eik: wp.array2d(dtype=float),
    m: wp.array2d(dtype=wp.int32),
    dx: float,
):
    """Set m = 1 next to the interface (does not clear other marks)."""
    i, j = wp.tid()
    nx = phi.shape[0]
    ny = phi.shape[1]
    if wp.abs(phi_eik[i, j]) >= 2.0 * dx:
        return
    p = phi[i, j]
    s = sgn(p)
    hit = int(0)
    if i > 0:
        q = phi[i - 1, j]
        if sgn(q) != s or wp.abs(p + q) < dx:
            hit = 1
    if i < nx - 1:
        q = phi[i + 1, j]
        if sgn(q) != s or wp.abs(p + q) < dx:
            hit = 1
    if j > 0:
        q = phi[i, j - 1]
        if sgn(q) != s or wp.abs(p + q) < dx:
            hit = 1
    if j < ny - 1:
        q = phi[i, j + 1]
        if sgn(q) != s or wp.abs(p + q) < dx:
            hit = 1
    if hit == 1:
        m[i, j] = 1


@wp.kernel
def dilate(src: wp.array2d(dtype=wp.int32), dst: wp.array2d(dtype=wp.int32)):
    """dst = src grown by one cell in the 4-neighbourhood."""
    i, j = wp.tid()
    nx = src.shape[0]
    ny = src.shape[1]
    v = src[i, j]
    if v == 0:
        if i > 0 and src[i - 1, j] == 1:
            v = 1
        if i < nx - 1 and src[i + 1, j] == 1:
            v = 1
        if j > 0 and src[i, j - 1] == 1:
            v = 1
        if j < ny - 1 and src[i, j + 1] == 1:
            v = 1
    dst[i, j] = v


@wp.func
def _normal_at(phi: wp.array2d(dtype=float), g: wp.array2d(dtype=wp.vec2), x: wp.vec2, dx: float, hermite: int) -> wp.vec2:
    q = x / dx - wp.vec2(0.5, 0.5)
    if hermite == 1:
        return hermite_grad(phi, g, q, dx)
    return safe_normalize(bilerp(g, q))


@wp.kernel
def sl_eikonal_sweep(
    phi: wp.array2d(dtype=float),
    g: wp.array2d(dtype=wp.vec2),
    m: wp.array2d(dtype=wp.int32),
    phi_out: wp.array2d(dtype=float),
    g_out: wp.array2d(dtype=wp.vec2),
    dx: float,
    step: float,
    hermite: int,
):
    """One Jacobi sweep of characteristic-marching redistancing on cells with m == 0."""
    i, j = wp.tid()
    if m[i, j] != 0:
        phi_out[i, j] = phi[i, j]
        g_out[i, j] = g[i, j]
        return
    sigma = sgn(phi[i, j])
    h = step * dx
    xc = wp.vec2(float(i) + 0.5, float(j) + 0.5) * dx
    n1 = _normal_at(phi, g, xc, dx, hermite)
    n2 = _normal_at(phi, g, xc - 0.5 * sigma * h * n1, dx, hermite)
    n3 = _normal_at(phi, g, xc - 0.5 * sigma * h * n2, dx, hermite)
    n4 = _normal_at(phi, g, xc - sigma * h * n3, dx, hermite)
    xf = xc - sigma * h * (n1 + 2.0 * n2 + 2.0 * n3 + n4) / 6.0
    q = xf / dx - wp.vec2(0.5, 0.5)
    if hermite == 1:
        phi_out[i, j] = hermite_value(phi, g, q, dx) + sigma * h
        g_out[i, j] = hermite_grad(phi, g, q, dx)
    else:
        phi_out[i, j] = bilerp(phi, q) + sigma * h
        g_out[i, j] = safe_normalize(bilerp(g, q))


@wp.func
def _particle_or_hermite(x: wp.vec2, phi: wp.array2d(dtype=float), g: wp.array2d(dtype=wp.vec2),
                         parts: ParticleView, cells: CellListView, dx: float):
    """(phi, normal) from the normal-filtered particle KNN, falling back to Hermite."""
    q = x / dx - wp.vec2(0.5, 0.5)
    n_h = hermite_grad(phi, g, q, dx)
    val, n, ok = knn_interp(x, parts, cells, n_h, 1)
    if ok == 0:
        val = hermite_value(phi, g, q, dx)
        n = n_h
    return val, n


@wp.kernel
def newton_closest_point(
    phi: wp.array2d(dtype=float),
    g: wp.array2d(dtype=wp.vec2),
    m: wp.array2d(dtype=wp.int32),
    parts: ParticleView,
    cells: CellListView,
    dx: float,
    step_clamp: float,
    tol: float,
    max_iter: int,
    phi_out: wp.array2d(dtype=float),
    g_out: wp.array2d(dtype=wp.vec2),
):
    """Quasi-Newton projection of the cell centre onto {phi = 0} for cells with m == 1.

    Iteration (``psi`` the interpolated normal, ``e = x_s - x``)::

        delta = -phi psi / |psi|^2 + e - psi (e . psi) / |psi|^2,   |delta| clamped to step_clamp dx

    with back-and-forth damping (midpoint when two consecutive steps nearly
    cancel). Starts at ``x_s - phi n`` and stops when ``|delta| < tol dx`` or after ``max_iter``.
    """
    i, j = wp.tid()
    if m[i, j] != 1:
        phi_out[i, j] = phi[i, j]
        g_out[i, j] = g[i, j]
        return
    xs = wp.vec2(float(i) + 0.5, float(j) + 0.5) * dx
    x = xs - phi[i, j] * safe_normalize(g[i, j])
    x_prev = wp.vec2(0.0, 0.0)
    d_prev = wp.vec2(0.0, 0.0)
    r = step_clamp * dx
    tol_abs = tol * dx
    for it in range(max_iter):
        val, psi = _particle_or_hermite(x, phi, g, parts, cells, dx)
        pn2 = wp.dot(psi, psi)
        if pn2 < 1.0e-12:
            break
        e = xs - x
        delta = -val * psi / pn2 + e - psi * (wp.dot(e, psi) / pn2)
        n = wp.length(delta)
        if n > r:
            delta = delta * (r / n)
            n = r
        if wp.length(delta + d_prev) < 0.1 * n:
            mid = 0.5 * (x + x_prev)
            x_prev = x
            x = mid
            d_prev = wp.vec2(0.0, 0.0)
        else:
            x_prev = x
            d_prev = delta
            x = x + delta
        if n < tol_abs:
            break
    s0, n0, ok0 = knn_interp(xs, parts, cells, wp.vec2(0.0, 0.0), 0)
    if ok0 == 0:
        s0 = hermite_value(phi, g, xs / dx - wp.vec2(0.5, 0.5), dx)
    sigma = sgn(s0)
    e = xs - x
    dist = wp.length(e)
    phi_out[i, j] = sigma * dist
    val, psi = _particle_or_hermite(x, phi, g, parts, cells, dx)
    gn = safe_normalize(psi)
    if dist > 1.0e-4 * dx:
        gn = sigma * (e / dist)
        if wp.dot(gn, psi) < 0.0:
            gn = -gn
    g_out[i, j] = gn


class QuasiNewtonRedistancer:
    """Scratch buffers and launch sequence for the long-reinitialisation redistancing."""

    def __init__(self, nx: int, ny: int, dx: float, device=None, dilations: int = 3, linear_sweeps: int = 5,
                 hermite_sweeps: int = 30, sl_step: float = 0.9, max_iter: int = 64, step_clamp: float = 0.02,
                 tol: float = 1.0e-4):
        self.nx, self.ny, self.dx = int(nx), int(ny), float(dx)
        self.device = wp.get_device(device)
        self.dilations = int(dilations)
        self.linear_sweeps = int(linear_sweeps)
        self.hermite_sweeps = int(hermite_sweeps)
        self.sl_step = float(sl_step)
        self.max_iter = int(max_iter)
        self.step_clamp = float(step_clamp)
        self.tol = float(tol)
        shape = (self.nx, self.ny)
        self._phi = wp.zeros(shape, dtype=float, device=self.device)
        self._g = wp.zeros(shape, dtype=wp.vec2, device=self.device)
        self._m = wp.zeros(shape, dtype=wp.int32, device=self.device)

    def _launch(self, kernel, inputs):
        wp.launch(kernel, dim=(self.nx, self.ny), inputs=inputs, device=self.device)

    def _sweeps(self, phi, g, m, count: int, hermite: int) -> None:
        a_phi, a_g, b_phi, b_g = phi, g, self._phi, self._g
        for _ in range(count):
            self._launch(sl_eikonal_sweep, [a_phi, a_g, m, b_phi, b_g, self.dx, self.sl_step, hermite])
            a_phi, a_g, b_phi, b_g = b_phi, b_g, a_phi, a_g
        if a_phi is not phi:
            wp.copy(phi, a_phi)
            wp.copy(g, a_g)

    def run(self, phi: wp.array, g: wp.array, m: wp.array, phi_eik: wp.array,
            parts: ParticleView, cells: CellListView) -> None:
        """Redistance ``(phi, g)`` in place. ``m`` holds the particle-coverage marks on entry and
        the Newton band on exit; ``phi_eik`` is the eikonal-redistanced initial guess."""
        self._launch(mark_interface, [phi, phi_eik, m, self.dx])
        self._sweeps(phi, g, m, self.linear_sweeps, 0)
        m.zero_()
        self._launch(mark_interface, [phi, phi_eik, m, self.dx])
        src, dst = m, self._m
        for _ in range(self.dilations):
            self._launch(dilate, [src, dst])
            src, dst = dst, src
        if src is not m:
            wp.copy(m, src)
        self._launch(newton_closest_point, [phi, g, m, parts, cells, self.dx, self.step_clamp, self.tol,
                                            self.max_iter, self._phi, self._g])
        wp.copy(phi, self._phi)
        wp.copy(g, self._g)
        self._sweeps(phi, g, m, self.hermite_sweeps, 1)
