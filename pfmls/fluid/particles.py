"""Velocity-carrier particles of the APIC transport.

Particles live in fixed slots, ``ppc = s*s`` per grid cell (slot ``k`` belongs to cell ``k // ppc``), so seeding
is deterministic and needs no host synchronisation: a slot is *active* when its cell was liquid (or within
``seed_band`` of the surface) at the last reseed.  Particle-to-grid transfer uses quadratic B-spline weights
with an affine (APIC) correction; faces receiving no particle weight take a fallback value.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..core.kernels import N2
from ._kernels import OFF_C, clamp_to_domain, sample_linear


@wp.kernel
def seed_kernel(
    phi: wp.array2d(dtype=float), side: int, dx: float, seed_band: float,
    x: wp.array(dtype=wp.vec2), active: wp.array(dtype=wp.int32),
):
    """Place ``side x side`` particles on a regular sub-lattice of every cell with phi < seed_band."""
    k = wp.tid()
    ppc = side * side
    c = k // ppc
    q = k - c * ppc
    ny = phi.shape[1]
    ci = c // ny
    cj = c - ci * ny
    a = q // side
    b = q - a * side
    h = 1.0 / float(side)
    x[k] = wp.vec2((float(ci) + (float(a) + 0.5) * h) * dx, (float(cj) + (float(b) + 0.5) * h) * dx)
    if phi[ci, cj] < seed_band:
        active[k] = 1
    else:
        active[k] = 0


@wp.kernel
def deactivate_outside_kernel(
    x: wp.array(dtype=wp.vec2), phi: wp.array2d(dtype=float), inv_dx: float, kill_band: float,
    active: wp.array(dtype=wp.int32),
):
    """Drop particles that drifted further than ``kill_band`` into the air."""
    k = wp.tid()
    if active[k] == 0:
        return
    ph = sample_linear(phi, x[k], inv_dx, OFF_C)
    if ph > kill_band or not wp.isfinite(ph):
        active[k] = 0


@wp.kernel
def p2g_apic_kernel(
    x: wp.array(dtype=wp.vec2), vel: wp.array(dtype=wp.vec2), C: wp.array(dtype=wp.mat22),
    active: wp.array(dtype=wp.int32), dx: float,
    num_u: wp.array2d(dtype=float), den_u: wp.array2d(dtype=float),
    num_v: wp.array2d(dtype=float), den_v: wp.array2d(dtype=float),
):
    """Scatter u_p + C_p (x_face - x_p) to both MAC face grids with quadratic B-spline weights."""
    k = wp.tid()
    if active[k] == 0:
        return
    p = x[k]
    up = vel[k]
    cp = C[k]
    inv_dx = 1.0 / dx
    # x-faces at (i, j+0.5) dx
    s = p[0] * inv_dx
    t = p[1] * inv_dx - 0.5
    bi = int(wp.floor(s - 0.5))
    bj = int(wp.floor(t - 0.5))
    n0 = num_u.shape[0]
    n1 = num_u.shape[1]
    for a in range(3):
        ii = bi + a
        if ii >= 0 and ii < n0:
            wx = N2(s - float(ii))
            for b in range(3):
                jj = bj + b
                if jj >= 0 and jj < n1:
                    w = wx * N2(t - float(jj))
                    d = wp.vec2(float(ii) * dx, (float(jj) + 0.5) * dx) - p
                    val = up[0] + cp[0, 0] * d[0] + cp[0, 1] * d[1]
                    wp.atomic_add(num_u, ii, jj, w * val)
                    wp.atomic_add(den_u, ii, jj, w)
    # y-faces at (i+0.5, j) dx
    s = p[0] * inv_dx - 0.5
    t = p[1] * inv_dx
    bi = int(wp.floor(s - 0.5))
    bj = int(wp.floor(t - 0.5))
    n0 = num_v.shape[0]
    n1 = num_v.shape[1]
    for a in range(3):
        ii = bi + a
        if ii >= 0 and ii < n0:
            wx = N2(s - float(ii))
            for b in range(3):
                jj = bj + b
                if jj >= 0 and jj < n1:
                    w = wx * N2(t - float(jj))
                    d = wp.vec2((float(ii) + 0.5) * dx, float(jj) * dx) - p
                    val = up[1] + cp[1, 0] * d[0] + cp[1, 1] * d[1]
                    wp.atomic_add(num_v, ii, jj, w * val)
                    wp.atomic_add(den_v, ii, jj, w)


@wp.func
def _normalize(num: wp.array2d(dtype=float), den: wp.array2d(dtype=float), fallback: wp.array2d(dtype=float),
               min_weight: float, out: wp.array2d(dtype=float), i: int, j: int):
    d = den[i, j]
    if d > min_weight:
        out[i, j] = num[i, j] / d
    else:
        out[i, j] = fallback[i, j]
    num[i, j] = 0.0  # cleared for the next transfer
    den[i, j] = 0.0


@wp.kernel
def p2g_normalize_kernel(
    num_u: wp.array2d(dtype=float), den_u: wp.array2d(dtype=float), fb_u: wp.array2d(dtype=float),
    num_v: wp.array2d(dtype=float), den_v: wp.array2d(dtype=float), fb_v: wp.array2d(dtype=float),
    min_weight: float, out_u: wp.array2d(dtype=float), out_v: wp.array2d(dtype=float),
):
    """Face value = weighted particle average, or the fallback without particle weight; clears the
    accumulators (launch dim (nx+1, ny+1))."""
    i, j = wp.tid()
    if j < out_u.shape[1]:
        _normalize(num_u, den_u, fb_u, min_weight, out_u, i, j)
    if i < out_v.shape[0]:
        _normalize(num_v, den_v, fb_v, min_weight, out_v, i, j)


@wp.kernel
def clamp_positions_kernel(x: wp.array(dtype=wp.vec2), active: wp.array(dtype=wp.int32), lx: float, ly: float,
                           margin: float):
    k = wp.tid()
    if active[k] != 0:
        x[k] = clamp_to_domain(x[k], lx, ly, margin)


class ParticleSlots:
    """Fixed-capacity particle storage with deterministic per-cell seeding and APIC P2G buffers."""

    def __init__(self, nx: int, ny: int, dx: float, side: int = 2, device=None):
        self.nx, self.ny, self.dx, self.side = nx, ny, dx, side
        self.device = wp.get_device(device)
        self.capacity = nx * ny * side * side
        self.x = wp.zeros(self.capacity, dtype=wp.vec2, device=self.device)
        self.active = wp.zeros(self.capacity, dtype=wp.int32, device=self.device)
        self.num_u = wp.zeros((nx + 1, ny), dtype=float, device=self.device)
        self.den_u = wp.zeros((nx + 1, ny), dtype=float, device=self.device)
        self.num_v = wp.zeros((nx, ny + 1), dtype=float, device=self.device)
        self.den_v = wp.zeros((nx, ny + 1), dtype=float, device=self.device)

    def seed(self, phi: wp.array2d, seed_band: float) -> None:
        wp.launch(seed_kernel, dim=self.capacity, inputs=[phi, self.side, self.dx, seed_band],
                  outputs=[self.x, self.active], device=self.device)

    def deactivate_outside(self, phi: wp.array2d, kill_band: float) -> None:
        wp.launch(deactivate_outside_kernel, dim=self.capacity, inputs=[self.x, phi, 1.0 / self.dx, kill_band],
                  outputs=[self.active], device=self.device)

    def p2g(self, vel: wp.array, C: wp.array, fallback_u, fallback_v, out_u, out_v, min_weight: float) -> None:
        """APIC transfer to the faces (the accumulators are zero between transfers)."""
        wp.launch(p2g_apic_kernel, dim=self.capacity, inputs=[self.x, vel, C, self.active, self.dx],
                  outputs=[self.num_u, self.den_u, self.num_v, self.den_v], device=self.device)
        wp.launch(p2g_normalize_kernel, dim=(self.nx + 1, self.ny + 1),
                  inputs=[self.num_u, self.den_u, fallback_u, self.num_v, self.den_v, fallback_v, min_weight],
                  outputs=[out_u, out_v], device=self.device)

    def positions_numpy(self) -> np.ndarray:
        a = self.active.numpy().astype(bool)
        return self.x.numpy()[a]
