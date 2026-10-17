"""Shared Warp functions for 3D grids: trilinear and tricubic Hermite interpolation, quadratic B-spline
sampling on MAC faces.

Conventions (as in :mod:`pfmls.core.kernels`)
---------------------------------------------
All grids are cell-centred: node ``(i, j, k)`` sits at ``((i, j, k) + 0.5) h``. Sampling functions take
a position in *index space*, ``q = x / h - 0.5``, and clamp to the grid (edge extension).

MAC faces: x-faces have shape ``(nx + 1, ny, nz)`` with nodes at ``(i, j + 0.5, k + 0.5) h`` (index-space
offset ``(0, 0.5, 0.5)``); y-faces ``(nx, ny + 1, nz)``, offset ``(0.5, 0, 0.5)``; z-faces
``(nx, ny, nz + 1)``, offset ``(0.5, 0.5, 0)``.

Tricubic Hermite interpolation uses, at each of the 8 corners of the cell containing the sample, the
value ``phi``, the gradient ``G`` and the mixed derivatives
``X = (h^2 phi_xy, h^2 phi_xz, h^2 phi_yz, h^3 phi_xyz)``, which :func:`hermite_cross_terms` estimates
by central differences of ``G`` (symmetrised over the axis pairs).
"""

from __future__ import annotations

from typing import Any

import warp as wp

from .kernels import N2, dN2

#: Length below which a vector is treated as zero by :func:`safe_normalize3`.
NORMALIZE_EPS3 = 1.0e-20


@wp.func
def safe_normalize3(v: wp.vec3) -> wp.vec3:
    """v / |v|; the zero vector when |v| is (numerically) zero or v is not finite.

    Finite vectors whose length overflows float32 are rescaled before normalising (as
    :func:`pfmls.core.kernels.safe_normalize`).
    """
    n = wp.length(v)
    if n > NORMALIZE_EPS3:
        if wp.isfinite(n):
            return v / n
        m = wp.max(wp.max(wp.abs(v[0]), wp.abs(v[1])), wp.abs(v[2]))
        if wp.isfinite(m):
            w = v / m
            return w / wp.length(w)
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def cell_center3(i: int, j: int, k: int, h: float) -> wp.vec3:
    """Physical position of cell centre ``(i, j, k)``."""
    return wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * h


# ---------------------------------------------------------------------------
# Trilinear sampling (cell-centred, clamped)
# ---------------------------------------------------------------------------


@wp.func
def trilerp(f: wp.array3d(dtype=Any), q: wp.vec3):
    """Trilinear sample of ``f`` (scalar or vector) at index-space position ``q`` with edge clamping."""
    nx = f.shape[0]
    ny = f.shape[1]
    nz = f.shape[2]
    i = wp.clamp(int(q[0]), 0, nx - 1)
    j = wp.clamp(int(q[1]), 0, ny - 1)
    k = wp.clamp(int(q[2]), 0, nz - 1)
    ip = wp.min(i + 1, nx - 1)
    jp = wp.min(j + 1, ny - 1)
    kp = wp.min(k + 1, nz - 1)
    s = wp.clamp(q[0] - float(i), 0.0, 1.0)
    t = wp.clamp(q[1] - float(j), 0.0, 1.0)
    u = wp.clamp(q[2] - float(k), 0.0, 1.0)
    c00 = f[i, j, k] * (1.0 - s) + f[ip, j, k] * s
    c10 = f[i, jp, k] * (1.0 - s) + f[ip, jp, k] * s
    c01 = f[i, j, kp] * (1.0 - s) + f[ip, j, kp] * s
    c11 = f[i, jp, kp] * (1.0 - s) + f[ip, jp, kp] * s
    return (c00 * (1.0 - t) + c10 * t) * (1.0 - u) + (c01 * (1.0 - t) + c11 * t) * u


# ---------------------------------------------------------------------------
# Tricubic Hermite interpolation from (phi, grad phi, mixed derivatives)
# ---------------------------------------------------------------------------


@wp.func
def _hbasis(s: float, node: int):
    """Cubic Hermite bases of ``node`` (0 or 1): value basis, slope basis and their s-derivatives."""
    if node == 0:
        return (2.0 * s + 1.0) * (s - 1.0) * (s - 1.0), s * (s - 1.0) * (s - 1.0), 6.0 * s * (s - 1.0), \
            (3.0 * s - 1.0) * (s - 1.0)
    return s * s * (3.0 - 2.0 * s), s * s * (s - 1.0), 6.0 * s * (1.0 - s), s * (3.0 * s - 2.0)


@wp.func
def _diff_factor(lo: int, hi: int) -> float:
    if hi > lo:
        return 1.0 / float(hi - lo)
    return 0.0


@wp.kernel
def hermite_cross_terms(G: wp.array3d(dtype=wp.vec3), X: wp.array3d(dtype=wp.vec4), h: float):
    """X = (h^2 phi_xy, h^2 phi_xz, h^2 phi_yz, h^3 phi_xyz) from central differences of G.

    Each mixed derivative is the mean of its estimates from the different gradient components
    (e.g. ``phi_xy = (d_x G_y + d_y G_x) / 2``); indices are clamped (one-sided on the boundary).
    """
    i, j, k = wp.tid()
    nx = G.shape[0]
    ny = G.shape[1]
    nz = G.shape[2]
    ip = wp.min(i + 1, nx - 1)
    im = wp.max(i - 1, 0)
    jp = wp.min(j + 1, ny - 1)
    jm = wp.max(j - 1, 0)
    kp = wp.min(k + 1, nz - 1)
    km = wp.max(k - 1, 0)
    fi = _diff_factor(im, ip)
    fj = _diff_factor(jm, jp)
    fk = _diff_factor(km, kp)
    xy = 0.5 * h * ((G[ip, j, k][1] - G[im, j, k][1]) * fi + (G[i, jp, k][0] - G[i, jm, k][0]) * fj)
    xz = 0.5 * h * ((G[ip, j, k][2] - G[im, j, k][2]) * fi + (G[i, j, kp][0] - G[i, j, km][0]) * fk)
    yz = 0.5 * h * ((G[i, jp, k][2] - G[i, jm, k][2]) * fj + (G[i, j, kp][1] - G[i, j, km][1]) * fk)
    dxy_z = (G[ip, jp, k][2] - G[ip, jm, k][2] - G[im, jp, k][2] + G[im, jm, k][2]) * fi * fj
    dxz_y = (G[ip, j, kp][1] - G[ip, j, km][1] - G[im, j, kp][1] + G[im, j, km][1]) * fi * fk
    dyz_x = (G[i, jp, kp][0] - G[i, jp, km][0] - G[i, jm, kp][0] + G[i, jm, km][0]) * fj * fk
    X[i, j, k] = wp.vec4(xy, xz, yz, h * (dxy_z + dxz_y + dyz_x) / 3.0)


@wp.func
def hermite3(phi: wp.array3d(dtype=float), G: wp.array3d(dtype=wp.vec3), X: wp.array3d(dtype=wp.vec4),
             q: wp.vec3, h: float):
    """Tricubic Hermite interpolation at index-space position ``q``: returns ``(value, gradient)``.

    The gradient is the exact derivative of the interpolant (not normalised).
    """
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]
    i = wp.clamp(int(q[0]), 0, nx - 1)
    j = wp.clamp(int(q[1]), 0, ny - 1)
    k = wp.clamp(int(q[2]), 0, nz - 1)
    s = wp.clamp(q[0] - float(i), 0.0, 1.0)
    t = wp.clamp(q[1] - float(j), 0.0, 1.0)
    u = wp.clamp(q[2] - float(k), 0.0, 1.0)
    val = float(0.0)
    gx = float(0.0)
    gy = float(0.0)
    gz = float(0.0)
    for a in range(2):
        va, sa, dva, dsa = _hbasis(s, a)
        ci = wp.min(i + a, nx - 1)
        for b in range(2):
            vb, sb, dvb, dsb = _hbasis(t, b)
            cj = wp.min(j + b, ny - 1)
            for c in range(2):
                vc, sc, dvc, dsc = _hbasis(u, c)
                ck = wp.min(k + c, nz - 1)
                p = phi[ci, cj, ck]
                g = G[ci, cj, ck] * h
                x = X[ci, cj, ck]
                val += (va * vb * vc * p + sa * vb * vc * g[0] + va * sb * vc * g[1] + va * vb * sc * g[2]
                        + sa * sb * vc * x[0] + sa * vb * sc * x[1] + va * sb * sc * x[2] + sa * sb * sc * x[3])
                gx += (dva * vb * vc * p + dsa * vb * vc * g[0] + dva * sb * vc * g[1] + dva * vb * sc * g[2]
                       + dsa * sb * vc * x[0] + dsa * vb * sc * x[1] + dva * sb * sc * x[2] + dsa * sb * sc * x[3])
                gy += (va * dvb * vc * p + sa * dvb * vc * g[0] + va * dsb * vc * g[1] + va * dvb * sc * g[2]
                       + sa * dsb * vc * x[0] + sa * dvb * sc * x[1] + va * dsb * sc * x[2] + sa * dsb * sc * x[3])
                gz += (va * vb * dvc * p + sa * vb * dvc * g[0] + va * sb * dvc * g[1] + va * vb * dsc * g[2]
                       + sa * sb * dvc * x[0] + sa * vb * dsc * x[1] + va * sb * dsc * x[2] + sa * sb * dsc * x[3])
    return val, wp.vec3(gx, gy, gz) / h


# ---------------------------------------------------------------------------
# Quadratic B-spline sampling on MAC faces (value + gradient)
# ---------------------------------------------------------------------------


@wp.func
def _bspline_weights(f: float):
    """N2 and dN2 at the four stencil offsets -1..2 for fractional position ``f``."""
    return wp.vec4(N2(f + 1.0), N2(f), N2(f - 1.0), N2(f - 2.0)), \
        wp.vec4(dN2(f + 1.0), dN2(f), dN2(f - 1.0), dN2(f - 2.0))


@wp.func
def bspline_sample_grad3(f: wp.array3d(dtype=float), x: wp.vec3, inv_dx: float, offset: wp.vec3):
    """Quadratic B-spline interpolation of a node/face array and its spatial gradient.

    ``offset`` is the index-space offset of node (0,0,0). The sample point is clamped to ``[1, n-2]``
    in index space and the 4x4x4 stencil indices are clamped, as in the 2D version.
    Returns ``(value, grad)``.
    """
    nx = f.shape[0]
    ny = f.shape[1]
    nz = f.shape[2]
    s = wp.clamp(x[0] * inv_dx - offset[0], 1.0, float(nx) - 2.0 - 1.0e-6)
    t = wp.clamp(x[1] * inv_dx - offset[1], 1.0, float(ny) - 2.0 - 1.0e-6)
    u = wp.clamp(x[2] * inv_dx - offset[2], 1.0, float(nz) - 2.0 - 1.0e-6)
    iu = int(wp.floor(s))
    iv = int(wp.floor(t))
    iw = int(wp.floor(u))
    wx, dwx = _bspline_weights(s - float(iu))
    wy, dwy = _bspline_weights(t - float(iv))
    wz, dwz = _bspline_weights(u - float(iw))
    val = float(0.0)
    gx = float(0.0)
    gy = float(0.0)
    gz = float(0.0)
    for a in range(4):
        ci = wp.clamp(iu + a - 1, 0, nx - 1)
        for b in range(4):
            cj = wp.clamp(iv + b - 1, 0, ny - 1)
            w_ab = wx[a] * wy[b]
            dx_ab = dwx[a] * wy[b]
            dy_ab = wx[a] * dwy[b]
            for c in range(4):
                ck = wp.clamp(iw + c - 1, 0, nz - 1)
                v = f[ci, cj, ck]
                val += v * w_ab * wz[c]
                gx += v * dx_ab * wz[c]
                gy += v * dy_ab * wz[c]
                gz += v * w_ab * dwz[c]
    return val, wp.vec3(gx, gy, gz) * inv_dx


@wp.func
def mac_velocity_grad3(ux: wp.array3d(dtype=float), uy: wp.array3d(dtype=float), uz: wp.array3d(dtype=float),
                       x: wp.vec3, inv_dx: float):
    """Velocity and velocity gradient ``G[a, b] = d u_a / d x_b`` from MAC face arrays (quadratic B-spline)."""
    u, gu = bspline_sample_grad3(ux, x, inv_dx, wp.vec3(0.0, 0.5, 0.5))
    v, gv = bspline_sample_grad3(uy, x, inv_dx, wp.vec3(0.5, 0.0, 0.5))
    w, gw = bspline_sample_grad3(uz, x, inv_dx, wp.vec3(0.5, 0.5, 0.0))
    return wp.vec3(u, v, w), wp.mat33(gu[0], gu[1], gu[2], gv[0], gv[1], gv[2], gw[0], gw[1], gw[2])
