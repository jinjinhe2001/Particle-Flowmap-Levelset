"""Shared Warp functions: B-spline weights, grid sampling, Hermite interpolation, sign helpers.

Conventions
-----------
All grids are cell-centred: node ``(i, j)`` sits at ``((i + 0.5) h, (j + 0.5) h)``.
Sampling functions take a position in *index space*, ``q = x / h - 0.5``, and clamp to the
grid (edge extension).

MAC faces: x-faces have shape ``(nx + 1, ny)`` with nodes at ``(i h, (j + 0.5) h)`` (index-space
offset ``(0, 0.5)``); y-faces have shape ``(nx, ny + 1)`` and offset ``(0.5, 0)``.

Sign rule: ``sgn(x) = +1`` for ``x >= 0`` and ``-1`` otherwise (zero counts as outside).
Normalisation: ``safe_normalize`` returns the zero vector for a zero or non-finite input (no NaN).
"""

from __future__ import annotations

from typing import Any

import warp as wp

#: Length below which a vector is treated as zero by :func:`safe_normalize`.
NORMALIZE_EPS = 1.0e-20


@wp.func
def sgn(x: float) -> float:
    """+1 for x >= 0, -1 otherwise."""
    if x >= 0.0:
        return 1.0
    return -1.0


@wp.func
def safe_normalize(v: wp.vec2) -> wp.vec2:
    """v / |v|; the zero vector when |v| is (numerically) zero or v is not finite.

    Finite vectors whose length overflows float32 are rescaled before normalising.
    """
    n = wp.length(v)
    if n > NORMALIZE_EPS:
        if wp.isfinite(n):
            return v / n
        m = wp.max(wp.abs(v[0]), wp.abs(v[1]))
        if wp.isfinite(m):
            w = v / m
            return w / wp.length(w)
    return wp.vec2(0.0, 0.0)


# ---------------------------------------------------------------------------
# Quadratic B-spline
# ---------------------------------------------------------------------------


@wp.func
def N2(x: float) -> float:
    """Quadratic B-spline: 3/4 - x^2 on |x|<1/2, (3/2-|x|)^2/2 on 1/2<=|x|<3/2, else 0."""
    ax = wp.abs(x)
    if ax < 0.5:
        return 0.75 - ax * ax
    if ax < 1.5:
        return 0.5 * (1.5 - ax) * (1.5 - ax)
    return 0.0


@wp.func
def dN2(x: float) -> float:
    """Exact derivative of :func:`N2`."""
    ax = wp.abs(x)
    r = 0.0
    if ax < 0.5:
        r = -2.0 * ax
    elif ax < 1.5:
        r = ax - 1.5
    if x < 0.0:
        r = -r
    return r


# ---------------------------------------------------------------------------
# Catmull-Rom cubic weights (any dimension, tensor product)
# ---------------------------------------------------------------------------


@wp.func
def catmull_rom_weights(t: float) -> wp.vec4:
    """Catmull-Rom weights of the nodes -1, 0, 1, 2 at the fractional position t in [0, 1]."""
    return wp.vec4(0.5 * (-t + 2.0 * t * t - t * t * t), 0.5 * (2.0 - 5.0 * t * t + 3.0 * t * t * t),
                   0.5 * (t + 4.0 * t * t - 3.0 * t * t * t), 0.5 * (-t * t + t * t * t))


@wp.func
def catmull_rom_dweights(t: float) -> wp.vec4:
    """Derivatives d/dt of :func:`catmull_rom_weights`."""
    return wp.vec4(0.5 * (-1.0 + 4.0 * t - 3.0 * t * t), 0.5 * (-10.0 * t + 9.0 * t * t),
                   0.5 * (1.0 + 8.0 * t - 9.0 * t * t), 0.5 * (-2.0 * t + 3.0 * t * t))


# ---------------------------------------------------------------------------
# Bilinear sampling (cell-centred, clamped)
# ---------------------------------------------------------------------------


@wp.func
def bilerp(f: wp.array2d(dtype=Any), q: wp.vec2):
    """Bilinear sample of ``f`` at index-space position ``q`` with edge clamping."""
    nx = f.shape[0]
    ny = f.shape[1]
    i = wp.clamp(int(q[0]), 0, nx - 1)
    j = wp.clamp(int(q[1]), 0, ny - 1)
    ip = wp.min(i + 1, nx - 1)
    jp = wp.min(j + 1, ny - 1)
    s = wp.clamp(q[0] - float(i), 0.0, 1.0)
    t = wp.clamp(q[1] - float(j), 0.0, 1.0)
    return (f[i, j] * (1.0 - s) + f[ip, j] * s) * (1.0 - t) + (f[i, jp] * (1.0 - s) + f[ip, jp] * s) * t


# ---------------------------------------------------------------------------
# Bicubic Hermite interpolation from (phi, grad phi) node data
# ---------------------------------------------------------------------------


@wp.func
def _hv(s: float, k: int) -> float:
    """Hermite value basis: (2s+1)(s-1)^2 for node 0, s^2(3-2s) for node 1."""
    if k == 0:
        return (2.0 * s + 1.0) * (s - 1.0) * (s - 1.0)
    return s * s * (3.0 - 2.0 * s)


@wp.func
def _hs(s: float, k: int) -> float:
    """Hermite slope basis: s(s-1)^2 for node 0, s^2(s-1) for node 1."""
    if k == 0:
        return s * (s - 1.0) * (s - 1.0)
    return s * s * (s - 1.0)


@wp.func
def _dhv(s: float, k: int) -> float:
    if k == 0:
        return 6.0 * s * (s - 1.0)
    return 6.0 * s * (1.0 - s)


@wp.func
def _dhs(s: float, k: int) -> float:
    if k == 0:
        return (3.0 * s - 1.0) * (s - 1.0)
    return s * (3.0 * s - 2.0)


@wp.func
def _hermite_cross(G: wp.array2d(dtype=wp.vec2), ci: int, cj: int, h: float) -> float:
    """h^2 * d2phi/dxdy estimated as the central x-difference of G_y (clamped)."""
    nx = G.shape[0]
    r = wp.min(ci + 1, nx - 1)
    l = wp.max(ci - 1, 0)
    return (G[r, cj][1] - G[l, cj][1]) * 0.5 * h


@wp.func
def hermite_value(phi: wp.array2d(dtype=float), G: wp.array2d(dtype=wp.vec2), q: wp.vec2, h: float) -> float:
    """Bicubic Hermite interpolation of phi using node values, gradients G and a FD cross term.

    ``q`` is the index-space position; ``h`` the grid spacing.
    """
    nx = phi.shape[0]
    ny = phi.shape[1]
    i = wp.clamp(int(q[0]), 0, nx - 1)
    j = wp.clamp(int(q[1]), 0, ny - 1)
    s = wp.clamp(q[0] - float(i), 0.0, 1.0)
    t = wp.clamp(q[1] - float(j), 0.0, 1.0)
    val = float(0.0)
    for a in range(2):
        for b in range(2):
            ci = wp.min(i + a, nx - 1)
            cj = wp.min(j + b, ny - 1)
            g = G[ci, cj]
            va = _hv(s, a)
            sa = _hs(s, a)
            vb = _hv(t, b)
            sb = _hs(t, b)
            val += va * vb * phi[ci, cj]
            val += sa * vb * h * g[0]
            val += va * sb * h * g[1]
            val += sa * sb * _hermite_cross(G, ci, cj, h)
    return val


@wp.func
def hermite_grad_raw(phi: wp.array2d(dtype=float), G: wp.array2d(dtype=wp.vec2), q: wp.vec2, h: float) -> wp.vec2:
    """Gradient of :func:`hermite_value` (not normalised)."""
    nx = phi.shape[0]
    ny = phi.shape[1]
    i = wp.clamp(int(q[0]), 0, nx - 1)
    j = wp.clamp(int(q[1]), 0, ny - 1)
    s = wp.clamp(q[0] - float(i), 0.0, 1.0)
    t = wp.clamp(q[1] - float(j), 0.0, 1.0)
    gx = float(0.0)
    gy = float(0.0)
    for a in range(2):
        for b in range(2):
            ci = wp.min(i + a, nx - 1)
            cj = wp.min(j + b, ny - 1)
            g = G[ci, cj]
            cross = _hermite_cross(G, ci, cj, h)
            p = phi[ci, cj]
            va = _hv(s, a)
            sa = _hs(s, a)
            vb = _hv(t, b)
            sb = _hs(t, b)
            dva = _dhv(s, a)
            dsa = _dhs(s, a)
            dvb = _dhv(t, b)
            dsb = _dhs(t, b)
            gx += dva * vb * p + dsa * vb * h * g[0] + dva * sb * h * g[1] + dsa * sb * cross
            gy += va * dvb * p + sa * dvb * h * g[0] + va * dsb * h * g[1] + sa * dsb * cross
    return wp.vec2(gx, gy) / h


@wp.func
def hermite_grad(phi: wp.array2d(dtype=float), G: wp.array2d(dtype=wp.vec2), q: wp.vec2, h: float) -> wp.vec2:
    """Unit normal from the Hermite interpolant (zero vector if the gradient vanishes)."""
    return safe_normalize(hermite_grad_raw(phi, G, q, h))


# ---------------------------------------------------------------------------
# Quadratic B-spline sampling on MAC faces (value + gradient)
# ---------------------------------------------------------------------------


@wp.func
def bspline_sample_grad(f: wp.array2d(dtype=float), x: wp.vec2, inv_dx: float, offset: wp.vec2):
    """Quadratic B-spline interpolation of a face/node array and its spatial gradient.

    ``offset`` is the index-space offset of node (0,0) (``(0, 0.5)`` for x-faces, ``(0.5, 0)`` for
    y-faces, ``(0.5, 0.5)`` for cell centres). The sample point is clamped to ``[1, n-2]`` in index
    space and the 4x4 stencil indices are clamped.
    Returns ``(value, grad)``.
    """
    nx = f.shape[0]
    ny = f.shape[1]
    s = wp.clamp(x[0] * inv_dx - offset[0], 1.0, float(nx) - 2.0 - 1.0e-6)
    t = wp.clamp(x[1] * inv_dx - offset[1], 1.0, float(ny) - 2.0 - 1.0e-6)
    iu = int(wp.floor(s))
    iv = int(wp.floor(t))
    val = float(0.0)
    gx = float(0.0)
    gy = float(0.0)
    for a in range(-1, 3):
        for b in range(-1, 3):
            ci = wp.clamp(iu + a, 0, nx - 1)
            cj = wp.clamp(iv + b, 0, ny - 1)
            fs = s - float(iu + a)
            ft = t - float(iv + b)
            v = f[ci, cj]
            ws = N2(fs)
            wt = N2(ft)
            val += v * ws * wt
            gx += v * dN2(fs) * wt
            gy += v * ws * dN2(ft)
    return val, wp.vec2(gx, gy) * inv_dx


@wp.func
def mac_velocity_grad(ux: wp.array2d(dtype=float), uy: wp.array2d(dtype=float), x: wp.vec2, inv_dx: float):
    """Velocity and velocity gradient ``G[a, b] = d u_a / d x_b`` from MAC face arrays.

    Quadratic B-spline on each face grid, sample point and stencil
    clamped to the grid. The level sets use it (:class:`pfmls.flows.MACVelocity2D`); the APIC particles of the
    free-surface solver use the wall-reflected variant :func:`mac_velocity_grad_reflect`.
    """
    u, gu = bspline_sample_grad(ux, x, inv_dx, wp.vec2(0.0, 0.5))
    v, gv = bspline_sample_grad(uy, x, inv_dx, wp.vec2(0.5, 0.0))
    return wp.vec2(u, v), wp.mat22(gu[0], gu[1], gv[0], gv[1])


# ---------------------------------------------------------------------------
# Quadratic B-spline sampling with wall reflection (APIC particles of the free-surface solver)
# ---------------------------------------------------------------------------


@wp.func
def _wall_node(ii: int, n: int, odd: int):
    """Node index and sign for stencil index ``ii`` of an ``n``-node line reflected at the walls.

    ``odd = 1``: normal velocity component, nodes ``0`` and ``n - 1`` lie on the walls (value zero) and the field
    is continued oddly; ``odd = 0``: tangential component, continued evenly (replicated)."""
    if odd == 1:
        k = ii
        sg = float(1.0)
        if ii < 0:
            k = -ii
            sg = -1.0
        elif ii > n - 1:
            k = 2 * (n - 1) - ii
            sg = -1.0
        return wp.clamp(k, 0, n - 1), sg
    return wp.clamp(ii, 0, n - 1), 1.0


@wp.func
def sample_bspline_wall(f: wp.array2d(dtype=float), p: wp.vec2, inv_dx: float, off: wp.vec2, axis: int) -> wp.vec3:
    """Quadratic B-spline sample (value, d/dx, d/dy) of MAC component ``axis`` (0: x-faces, 1: y-faces) at the world
    position ``p``, with odd reflection across the walls normal to it and even reflection along the others.

    ``off`` is the node offset of the face grid in cells (``(0, 0.5)`` for x-faces, ``(0.5, 0)`` for y-faces). Unlike
    :func:`bspline_sample_grad` the sample point is not clamped: the 3x3 stencil is continued across the walls, so
    a sampled velocity has ``u.n = 0`` on the walls."""
    nxg = f.shape[0]
    nyg = f.shape[1]
    s = p[0] * inv_dx - off[0]
    t = p[1] * inv_dx - off[1]
    bi = int(wp.floor(s - 0.5))
    bj = int(wp.floor(t - 0.5))
    ox = int(0)
    oy = int(0)
    if axis == 0:
        ox = 1
    else:
        oy = 1
    val = float(0.0)
    gx = float(0.0)
    gy = float(0.0)
    for a in range(3):
        ii = bi + a
        wx = N2(s - float(ii))
        dwx = dN2(s - float(ii))
        ic, sx = _wall_node(ii, nxg, ox)
        for b in range(3):
            jj = bj + b
            wy = N2(t - float(jj))
            dwy = dN2(t - float(jj))
            jc, sy = _wall_node(jj, nyg, oy)
            fv = sx * sy * f[ic, jc]
            val += wx * wy * fv
            gx += dwx * wy * fv
            gy += wx * dwy * fv
    return wp.vec3(val, gx * inv_dx, gy * inv_dx)


@wp.func
def mac_velocity_grad_reflect(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), p: wp.vec2, inv_dx: float):
    """Quadratic B-spline MAC velocity and its gradient ``G[r, c] = du_r/dx_c`` at the world position ``p``, with
    wall-reflected stencils (:func:`sample_bspline_wall`; the APIC transport of the free-surface solver)."""
    a = sample_bspline_wall(u, p, inv_dx, wp.vec2(0.0, 0.5), 0)
    b = sample_bspline_wall(v, p, inv_dx, wp.vec2(0.5, 0.0), 1)
    return wp.vec2(a[0], b[0]), wp.mat22(a[1], a[2], b[1], b[2])
