"""Warp helper functions shared by the free-surface solver.

Grid convention (all arrays indexed ``[i, j]`` with ``i`` along x):

* cell-centred scalars (phi, pressure, curvature): shape ``(nx, ny)``, node ``(i, j)`` at ``((i+0.5)dx, (j+0.5)dx)``;
* x-face velocity ``u``: shape ``(nx+1, ny)``, node at ``(i dx, (j+0.5)dx)``;
* y-face velocity ``v``: shape ``(nx, ny+1)``, node at ``((i+0.5)dx, j dx)``.

A staggered component is described by its node offset ``(ox, oy)`` in cells: ``(0, 0.5)`` for ``u``,
``(0.5, 0)`` for ``v`` and ``(0.5, 0.5)`` for cell-centred data, and sampled at a world position (the core samplers
of :mod:`pfmls.core.kernels` take index-space positions instead).  Scalar samplers clamp out-of-range stencil
indices (replicated boundary values).  The B-spline MAC velocity samplers of the APIC particles
(:func:`pfmls.core.kernels.mac_velocity_grad_reflect`, :func:`mac_velocity_smooth`) reflect the stencil at the walls
of the box: evenly for the tangential component (free slip) and oddly for the normal component, whose wall node is
zero, so a sampled velocity has ``u.n = 0`` on the walls and particles next to a wall cannot be pushed into it.
"""

from __future__ import annotations

import warp as wp

from ..core.kernels import catmull_rom_dweights, catmull_rom_weights, sample_bspline_wall

OFF_U = wp.constant(wp.vec2(0.0, 0.5))
OFF_V = wp.constant(wp.vec2(0.5, 0.0))
OFF_C = wp.constant(wp.vec2(0.5, 0.5))


@wp.func
def sample_linear(f: wp.array2d(dtype=float), p: wp.vec2, inv_dx: float, off: wp.vec2) -> float:
    """Bilinear sample of a staggered grid at world position ``p`` (positions clamped to the node range)."""
    nxg = f.shape[0]
    nyg = f.shape[1]
    s = wp.clamp(p[0] * inv_dx - off[0], 0.0, float(nxg - 1))
    t = wp.clamp(p[1] * inv_dx - off[1], 0.0, float(nyg - 1))
    i0 = wp.min(int(s), nxg - 2)
    j0 = wp.min(int(t), nyg - 2)
    fx = s - float(i0)
    fy = t - float(j0)
    a = f[i0, j0] * (1.0 - fx) + f[i0 + 1, j0] * fx
    b = f[i0, j0 + 1] * (1.0 - fx) + f[i0 + 1, j0 + 1] * fx
    return a * (1.0 - fy) + b * fy


@wp.func
def cubic_cr(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    """Catmull-Rom cubic between p1 (t=0) and p2 (t=1)."""
    return 0.5 * (
        2.0 * p1
        + (p2 - p0) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
        + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t * t * t
    )


@wp.func
def sample_cubic_clamped(f: wp.array2d(dtype=float), p: wp.vec2, inv_dx: float, off: wp.vec2) -> float:
    """Bicubic Catmull-Rom sample limited to the min/max of the enclosing 2x2 nodes (no new extrema)."""
    nxg = f.shape[0]
    nyg = f.shape[1]
    s = wp.clamp(p[0] * inv_dx - off[0], 0.0, float(nxg - 1))
    t = wp.clamp(p[1] * inv_dx - off[1], 0.0, float(nyg - 1))
    i1 = wp.min(int(s), nxg - 2)
    j1 = wp.min(int(t), nyg - 2)
    fx = s - float(i1)
    fy = t - float(j1)
    i0 = wp.max(i1 - 1, 0)
    i2 = i1 + 1
    i3 = wp.min(i1 + 2, nxg - 1)
    col = wp.vec4()
    for b in range(4):
        jj = wp.clamp(j1 - 1 + b, 0, nyg - 1)
        col[b] = cubic_cr(f[i0, jj], f[i1, jj], f[i2, jj], f[i3, jj], fx)
    val = cubic_cr(col[0], col[1], col[2], col[3], fy)
    lo = wp.min(wp.min(f[i1, j1], f[i2, j1]), wp.min(f[i1, j1 + 1], f[i2, j1 + 1]))
    hi = wp.max(wp.max(f[i1, j1], f[i2, j1]), wp.max(f[i1, j1 + 1], f[i2, j1 + 1]))
    return wp.clamp(val, lo, hi)


@wp.func
def sample_cubic_grad(f: wp.array2d(dtype=float), p: wp.vec2, inv_dx: float, off: wp.vec2) -> wp.vec3:
    """Bicubic Catmull-Rom (C1, interpolating) sample: returns (value, d/dx, d/dy)."""
    nxg = f.shape[0]
    nyg = f.shape[1]
    s = wp.clamp(p[0] * inv_dx - off[0], 0.0, float(nxg - 1))
    t = wp.clamp(p[1] * inv_dx - off[1], 0.0, float(nyg - 1))
    i1 = wp.min(int(s), nxg - 2)
    j1 = wp.min(int(t), nyg - 2)
    wx = catmull_rom_weights(s - float(i1))
    dwx = catmull_rom_dweights(s - float(i1))
    wy = catmull_rom_weights(t - float(j1))
    dwy = catmull_rom_dweights(t - float(j1))
    val = float(0.0)
    gx = float(0.0)
    gy = float(0.0)
    for a in range(4):
        ii = wp.clamp(i1 - 1 + a, 0, nxg - 1)
        for b in range(4):
            jj = wp.clamp(j1 - 1 + b, 0, nyg - 1)
            fv = f[ii, jj]
            val += wx[a] * wy[b] * fv
            gx += dwx[a] * wy[b] * fv
            gy += wx[a] * dwy[b] * fv
    return wp.vec3(val, gx * inv_dx, gy * inv_dx)


@wp.func
def mac_velocity(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), p: wp.vec2, inv_dx: float) -> wp.vec2:
    """Bilinear MAC velocity at a world position."""
    return wp.vec2(sample_linear(u, p, inv_dx, OFF_U), sample_linear(v, p, inv_dx, OFF_V))


@wp.func
def mac_velocity_smooth(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), p: wp.vec2, inv_dx: float) -> wp.vec2:
    """Quadratic B-spline MAC velocity (smooth, wall-reflected; used to move particles)."""
    return wp.vec2(sample_bspline_wall(u, p, inv_dx, OFF_U, 0)[0], sample_bspline_wall(v, p, inv_dx, OFF_V, 1)[0])


@wp.func
def sample_linear_grad(f: wp.array2d(dtype=float), p: wp.vec2, inv_dx: float, off: wp.vec2) -> wp.vec3:
    """Bilinear sample of a staggered grid and the gradient of the bilinear interpolant: (value, d/dx, d/dy)."""
    nxg = f.shape[0]
    nyg = f.shape[1]
    s = wp.clamp(p[0] * inv_dx - off[0], 0.0, float(nxg - 1))
    t = wp.clamp(p[1] * inv_dx - off[1], 0.0, float(nyg - 1))
    i0 = wp.min(int(s), nxg - 2)
    j0 = wp.min(int(t), nyg - 2)
    fx = s - float(i0)
    fy = t - float(j0)
    f00 = f[i0, j0]
    f10 = f[i0 + 1, j0]
    f01 = f[i0, j0 + 1]
    f11 = f[i0 + 1, j0 + 1]
    a = f00 * (1.0 - fx) + f10 * fx
    b = f01 * (1.0 - fx) + f11 * fx
    gx = ((f10 - f00) * (1.0 - fy) + (f11 - f01) * fy) * inv_dx
    gy = (b - a) * inv_dx
    return wp.vec3(a * (1.0 - fy) + b * fy, gx, gy)


@wp.func
def mac_velocity_linear(x: wp.vec2, t: float, prm: wp.vec4, ux: wp.array2d(dtype=float), uy: wp.array2d(dtype=float),
                        inv_dx: float):
    """Bilinear (interpolating) MAC velocity and the gradient of its interpolant, in the
    :class:`pfmls.flows.Flow2D` signature.  Exact at the faces, so the velocity of a thin liquid sheet is not
    smoothed into its extrapolated surroundings; the normal component vanishes on the walls."""
    a = sample_linear_grad(ux, x, inv_dx, OFF_U)
    b = sample_linear_grad(uy, x, inv_dx, OFF_V)
    return wp.vec2(a[0], b[0]), wp.mat22(a[1], a[2], b[1], b[2])


# ------------------------------------------------------------------------------------------- liquid area
@wp.func
def _corner_triangle(a: float, b: float, c: float) -> float:
    """Area of the corner triangle cut from the unit square at a negative corner a with edge neighbours b, c."""
    return 0.5 * (a / (a - b)) * (a / (a - c))


@wp.func
def _square_corner(k: int) -> wp.vec2:
    if k == 0:
        return wp.vec2(0.0, 0.0)
    if k == 1:
        return wp.vec2(1.0, 0.0)
    if k == 2:
        return wp.vec2(1.0, 1.0)
    return wp.vec2(0.0, 1.0)


@wp.func
def square_liquid_fraction(v0: float, v1: float, v2: float, v3: float) -> float:
    """Fraction of the unit square where the marching-squares interpolant of the corner values is negative.

    Corners counter-clockwise from (0, 0); edges are linear, saddles are resolved by the mean value (as in
    marching squares), and the fraction is the shoelace area of the resulting polygon(s)."""
    neg = int(0)
    if v0 < 0.0:
        neg += 1
    if v1 < 0.0:
        neg += 1
    if v2 < 0.0:
        neg += 1
    if v3 < 0.0:
        neg += 1
    if neg == 0:
        return 0.0
    if neg == 4:
        return 1.0
    mean = v0 + v1 + v2 + v3
    if neg == 2 and v0 < 0.0 and v2 < 0.0 and mean >= 0.0:
        return _corner_triangle(v0, v1, v3) + _corner_triangle(v2, v1, v3)
    if neg == 2 and v1 < 0.0 and v3 < 0.0 and mean >= 0.0:
        return _corner_triangle(v1, v0, v2) + _corner_triangle(v3, v0, v2)
    vals = wp.vec4(v0, v1, v2, v3)
    first = wp.vec2(0.0, 0.0)
    prev = wp.vec2(0.0, 0.0)
    have = int(0)
    area2 = float(0.0)
    for k in range(4):
        a = vals[k]
        b = vals[(k + 1) % 4]
        pk = _square_corner(k)
        pn = _square_corner((k + 1) % 4)
        for e in range(2):
            q = wp.vec2(0.0, 0.0)
            use = int(0)
            if e == 0 and a < 0.0:
                q = pk
                use = 1
            if e == 1 and ((a < 0.0) != (b < 0.0)):
                q = pk + (a / (a - b)) * (pn - pk)
                use = 1
            if use == 1:
                if have == 0:
                    first = q
                    have = 1
                else:
                    area2 += prev[0] * q[1] - q[0] * prev[1]
                prev = q
    area2 += prev[0] * first[1] - first[0] * prev[1]
    return 0.5 * wp.abs(area2)


@wp.func
def _wall_value(phi: wp.array2d(dtype=float), I: int, J: int) -> float:
    """phi at the dual-cell corner (I, J) of the node lattice extended to the walls.

    Corners on a wall take the linear extrapolation of the two nearest cell centres along the wall normal (at a
    box corner, along both axes), so a level set whose interface has moved off the wall is not counted as liquid
    up to the wall."""
    nx = phi.shape[0]
    ny = phi.shape[1]
    i = wp.clamp(I, 0, nx - 1)
    j = wp.clamp(J, 0, ny - 1)
    di = int(0)
    dj = int(0)
    if I < 0:
        di = 1
    elif I > nx - 1:
        di = -1
    if J < 0:
        dj = 1
    elif J > ny - 1:
        dj = -1
    if nx < 2:
        di = 0
    if ny < 2:
        dj = 0
    f = phi[i, j]
    if di != 0:
        f += 0.5 * (phi[i, j] - phi[i + di, j])
    if dj != 0:
        f += 0.5 * (phi[i, j] - phi[i, j + dj])
    if di != 0 and dj != 0:
        f += 0.25 * (phi[i, j] - phi[i + di, j] - phi[i, j + dj] + phi[i + di, j + dj])
    return f


@wp.func
def dual_cell_liquid_area(phi: wp.array2d(dtype=float), I: int, J: int, dx: float) -> float:
    """Liquid area in the dual cell between cell centres (I-1..I, J-1..J), I in [0, nx], J in [0, ny].

    Boundary dual cells are the half (corner: quarter) cells between the outermost centres and the walls; their
    wall corners hold phi linearly extrapolated to the wall.  Summing over all dual cells gives the liquid area
    of the box (exact for a level set that is linear across each dual cell)."""
    nx = phi.shape[0]
    ny = phi.shape[1]
    wx = dx
    wy = dx
    if I == 0 or I == nx:
        wx = 0.5 * dx
    if J == 0 or J == ny:
        wy = 0.5 * dx
    # corner positions: cell centres, or wall points (index -1 / n meaning "on the wall")
    f00 = _wall_value(phi, I - 1, J - 1)
    f10 = _wall_value(phi, I, J - 1)
    f11 = _wall_value(phi, I, J)
    f01 = _wall_value(phi, I - 1, J)
    f = square_liquid_fraction(f00, f10, f11, f01)
    return f * wx * wy


@wp.func
def clamp_to_domain(p: wp.vec2, lx: float, ly: float, margin: float) -> wp.vec2:
    return wp.vec2(wp.clamp(p[0], margin, lx - margin), wp.clamp(p[1], margin, ly - margin))
