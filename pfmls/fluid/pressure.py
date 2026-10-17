"""Liquid-only pressure projection with a ghost-fluid free-surface condition.

The unknown is the scaled pressure ``P = dt p / rho`` on liquid cells (``phi < 0``).  Air cells are not
solved for; at a liquid/air cell pair the interface sits a fraction

    theta = clamp(-phi_l / (phi_a - phi_l), theta_min, 1)

of the way from the liquid centre, where the Dirichlet value ``P_s = dt sigma kappa_I / rho`` is imposed
(Gibou et al. 2002 / Enright et al. 2003 style ghost fluid, with the Laplace pressure jump of Kang, Fedkiw &
Liu 2000).  ``kappa_I`` is the curvature linearly interpolated to the interface and clamped to
``+-kappa_max``.  Multiplying by ``dx^2`` the system for a liquid cell reads

    sum_{liquid nbrs} (P_i - P_n) + sum_{air nbrs} (P_i - P_s) / theta = -dx^2 div u*.

Walls.  Closed walls are homogeneous Neumann (``u.n = 0``).  The walls selected by ``wall_separation``
(``"lid"`` by default: liquid may fall away from the top wall while the floor and side walls stay closed;
``"all"``; ``"none"``) are unilateral: they push but do not pull.  The system is first solved with every wall
closed; each selected wall face whose liquid cell is then under suction (``P < 0``) is opened, i.e. it becomes a
free-surface boundary ``P = 0`` half a cell from the liquid node, and the system is solved again; open faces
whose velocity would then point into the wall are closed again and the system is solved a third time
(warm-started, so a solve with nothing to change is already converged).  This is a two-pass approximation of the
contact complementarity condition; any inward velocity left on an open face is set to zero.  The level set must
let air in behind an open wall (:class:`pfmls.fluid.pfm_backend.PFMLevelSetBackend` with a ghost layer).

Solver.  The SPD system is solved with Jacobi-preconditioned conjugate gradients entirely on the device.  One
iteration is two fused kernels: the search-direction update evaluated on the fly inside ``A s`` (with the
block-reduced ``s.As``), and the ``x, r`` update with ``r.z`` and ``max|r|``.  On CUDA the whole loop, convergence
test included, is a graph with a device-side ``while`` node (``wp.capture_while``): its own graph, or a conditional
node of the captured solver step (:class:`pfmls.fluid.freesurface2d.FreeSurfaceSolver2D`), so a solve needs no host
synchronisation.  The iteration count and residual stay on the device (:attr:`PressureSolver.stats`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

BLOCK = 256  # threads per block of the fused CG kernels (block reductions)
WALL_THETA = 0.5  # an open wall face is a P = 0 boundary half a cell from the liquid node

# bits of the open-wall mask (one int per cell)
OPEN_LEFT = wp.constant(1)
OPEN_RIGHT = wp.constant(2)
OPEN_BOTTOM = wp.constant(4)
OPEN_TOP = wp.constant(8)
WALL_SETS = {"none": 0, "lid": 8, "all": 15}


@wp.func
def interface_theta(phi_l: float, phi_a: float, theta_min: float) -> float:
    """Fraction of the cell spacing between the liquid node and the interface."""
    return wp.clamp(-phi_l / wp.max(phi_a - phi_l, 1.0e-12), theta_min, 1.0)


@wp.func
def surface_pressure(kappa_l: float, kappa_a: float, theta: float, dt_sigma: float, kappa_max: float) -> float:
    """Scaled Laplace pressure dt*sigma*kappa at the interface point (curvature interpolated, clamped)."""
    k = wp.clamp((1.0 - theta) * kappa_l + theta * kappa_a, -kappa_max, kappa_max)
    return dt_sigma * k


# ----------------------------------------------------------------------------------------------- curvature
@wp.kernel
def curvature_kernel(phi: wp.array2d(dtype=float), inv_dx: float, kappa: wp.array2d(dtype=float)):
    """kappa = div(grad phi / |grad phi|) with compact face normals (kappa > 0 for a convex liquid blob)."""
    i, j = wp.tid()
    nx = phi.shape[0]
    ny = phi.shape[1]
    im = wp.max(i - 1, 0)
    ip = wp.min(i + 1, nx - 1)
    jm = wp.max(j - 1, 0)
    jp = wp.min(j + 1, ny - 1)
    c = phi[i, j]
    # east / west faces
    gxe = (phi[ip, j] - c) * inv_dx
    gye = 0.25 * (phi[ip, jp] + phi[i, jp] - phi[ip, jm] - phi[i, jm]) * inv_dx
    gxw = (c - phi[im, j]) * inv_dx
    gyw = 0.25 * (phi[i, jp] + phi[im, jp] - phi[i, jm] - phi[im, jm]) * inv_dx
    # north / south faces
    gyn = (phi[i, jp] - c) * inv_dx
    gxn = 0.25 * (phi[ip, jp] + phi[ip, j] - phi[im, jp] - phi[im, j]) * inv_dx
    gys = (c - phi[i, jm]) * inv_dx
    gxs = 0.25 * (phi[ip, j] + phi[ip, jm] - phi[im, j] - phi[im, jm]) * inv_dx
    ne = gxe / wp.max(wp.sqrt(gxe * gxe + gye * gye), 1.0e-8)
    nw = gxw / wp.max(wp.sqrt(gxw * gxw + gyw * gyw), 1.0e-8)
    nn = gyn / wp.max(wp.sqrt(gxn * gxn + gyn * gyn), 1.0e-8)
    ns = gys / wp.max(wp.sqrt(gxs * gxs + gys * gys), 1.0e-8)
    k = (ne - nw + nn - ns) * inv_dx
    if not wp.isfinite(k):
        k = 0.0
    kappa[i, j] = k


@wp.kernel
def smooth_band_kernel(
    fin: wp.array2d(dtype=float), phi: wp.array2d(dtype=float), band: float, fout: wp.array2d(dtype=float)
):
    """One pass of the 3x3 binomial filter [1 2 1]^2/16 on cells within ``band`` of the interface."""
    i, j = wp.tid()
    nx = fin.shape[0]
    ny = fin.shape[1]
    if wp.abs(phi[i, j]) > band:
        fout[i, j] = fin[i, j]
        return
    s = float(0.0)
    w = float(0.0)
    for a in range(-1, 2):
        for b in range(-1, 2):
            ii = wp.clamp(i + a, 0, nx - 1)
            jj = wp.clamp(j + b, 0, ny - 1)
            ww = float((2 - wp.abs(a)) * (2 - wp.abs(b)))
            s += ww * fin[ii, jj]
            w += ww
    fout[i, j] = s / w


# ------------------------------------------------------------------------------------------- system build
@wp.func
def wall_flux(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), i: int, j: int, open_mask: int) -> float:
    """Net outflow u[i+1] - u[i] + v[j+1] - v[j] of cell (i, j); closed wall faces carry no flux."""
    nx = v.shape[0]
    ny = u.shape[1]
    ul = u[i, j]
    ur = u[i + 1, j]
    vb = v[i, j]
    vt = v[i, j + 1]
    if i == 0 and (open_mask & OPEN_LEFT) == 0:
        ul = 0.0
    if i == nx - 1 and (open_mask & OPEN_RIGHT) == 0:
        ur = 0.0
    if j == 0 and (open_mask & OPEN_BOTTOM) == 0:
        vb = 0.0
    if j == ny - 1 and (open_mask & OPEN_TOP) == 0:
        vt = 0.0
    return ur - ul + vt - vb


@wp.func
def count_open(m: int) -> float:
    c = float(0.0)
    if (m & OPEN_LEFT) != 0:
        c += 1.0
    if (m & OPEN_RIGHT) != 0:
        c += 1.0
    if (m & OPEN_BOTTOM) != 0:
        c += 1.0
    if (m & OPEN_TOP) != 0:
        c += 1.0
    return c


@wp.kernel
def build_system_kernel(
    phi: wp.array2d(dtype=float),
    kappa: wp.array2d(dtype=float),
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    open_walls: wp.array2d(dtype=wp.int32),
    dx: float,
    dt_sigma: float,
    theta_min: float,
    kappa_max: float,
    flag: wp.array2d(dtype=wp.int32),
    diag: wp.array2d(dtype=float),
    rhs: wp.array2d(dtype=float),
    p: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    nx = phi.shape[0]
    ny = phi.shape[1]
    ph = phi[i, j]
    if not (ph < 0.0):
        flag[i, j] = 0
        diag[i, j] = 1.0
        rhs[i, j] = 0.0
        p[i, j] = 0.0
        return
    flag[i, j] = 1
    kl = kappa[i, j]
    om = open_walls[i, j]
    b = -wall_flux(u, v, i, j, om) * dx
    d = count_open(om) / WALL_THETA
    for k in range(4):
        ii = i
        jj = j
        if k == 0:
            ii = i - 1
        elif k == 1:
            ii = i + 1
        elif k == 2:
            jj = j - 1
        else:
            jj = j + 1
        if ii >= 0 and ii < nx and jj >= 0 and jj < ny:
            pn = phi[ii, jj]
            if pn < 0.0:
                d += 1.0
            else:
                th = interface_theta(ph, pn, theta_min)
                d += 1.0 / th
                b += surface_pressure(kl, kappa[ii, jj], th, dt_sigma, kappa_max) / th
    if not wp.isfinite(p[i, j]):
        p[i, j] = 0.0
    diag[i, j] = d
    rhs[i, j] = b


@wp.func
def apply_A(x: wp.array(dtype=float), flag: wp.array(dtype=wp.int32), diag: wp.array(dtype=float), k: int,
            nx: int, ny: int) -> float:
    """(A x)_k on the flattened grid (k = i ny + j); identity on air cells."""
    xc = x[k]
    if flag[k] == 0:
        return xc
    i = k // ny
    j = k - i * ny
    s = diag[k] * xc
    if i > 0:
        if flag[k - ny] == 1:
            s -= x[k - ny]
    if i < nx - 1:
        if flag[k + ny] == 1:
            s -= x[k + ny]
    if j > 0:
        if flag[k - 1] == 1:
            s -= x[k - 1]
    if j < ny - 1:
        if flag[k + 1] == 1:
            s -= x[k + 1]
    return s


# ------------------------------------------------------------------------------- fused PCG (flattened 1D)
# Device scalars: rz[3] and sAs[3] (rotating slots), rmax[1], it[1] (int, iterations of the current solve),
# cond[1] (int), stat[2] = (iterations, residual).  Iteration g of a solve uses slot g % 3 for the current r.z and
# s.As, (g + 2) % 3 for the previous r.z and (g + 1) % 3 for the next, derived from the device counter ``it`` (the
# unrolled loop body has no iteration-dependent slot arguments).  The search direction ping-pongs between two
# buffers; ``it`` is a multiple of the (even) block length at the start of a block, so the parity of g is known on
# the host and the buffers are passed in the right order.
@wp.func
def _precond(r: wp.array(dtype=float), flag: wp.array(dtype=wp.int32), diag: wp.array(dtype=float), k: int) -> float:
    """z = M^-1 r (Jacobi; zero on air cells)."""
    z = float(0.0)
    if flag[k] == 1:
        z = r[k] / diag[k]
    return z


@wp.func
def _direction(r: wp.array(dtype=float), s_old: wp.array(dtype=float), flag: wp.array(dtype=wp.int32),
               diag: wp.array(dtype=float), k: int, beta: float, first: int) -> float:
    """s = z + beta s_old (s = z at the first iteration)."""
    z = _precond(r, flag, diag, k)
    if first == 1:
        return z
    return z + beta * s_old[k]


@wp.kernel
def pcg_start_kernel(
    p: wp.array(dtype=float), b: wp.array(dtype=float), flag: wp.array(dtype=wp.int32), diag: wp.array(dtype=float),
    nx: int, ny: int, block: int, r: wp.array(dtype=float), rz: wp.array(dtype=float), rmax: wp.array(dtype=float),
):
    """r = b - A p, rz[0] = r.M^-1 r, rmax = max|r| (rz, rmax zeroed beforehand)."""
    k = wp.tid()
    dot = float(0.0)
    ra = float(0.0)
    if k < nx * ny:
        rr = b[k] - apply_A(p, flag, diag, k, nx, ny)
        r[k] = rr
        if flag[k] == 1:
            dot = rr * rr / diag[k]
        ra = wp.abs(rr)
        if not wp.isfinite(ra):
            ra = 1.0e30
    wp.tile_atomic_add(rz, wp.tile_sum(wp.tile(dot)), offset=0)
    m = wp.tile_max(wp.tile(ra))
    if k % block == 0:
        wp.atomic_max(rmax, 0, m[0])


@wp.kernel
def pcg_matvec_kernel(
    r: wp.array(dtype=float), s_old: wp.array(dtype=float), flag: wp.array(dtype=wp.int32),
    diag: wp.array(dtype=float), nx: int, ny: int, it: wp.array(dtype=wp.int32), k_in_block: int,
    s_new: wp.array(dtype=float), As: wp.array(dtype=float), sAs: wp.array(dtype=float), rz: wp.array(dtype=float),
    rmax: wp.array(dtype=float),
):
    """New search direction s = M^-1 r + beta s_old (evaluated at the cell and its neighbours, so the direction
    update needs no launch of its own), As = A s and s.As; clears the accumulators of the next iteration."""
    k = wp.tid()
    g = it[0] + k_in_block
    c = g % 3
    first = int(0)
    beta = float(0.0)
    if g == 0:
        first = 1
    else:
        old = rz[(g + 2) % 3]
        if wp.abs(old) > 1.0e-30:
            beta = rz[c] / old
    dot = float(0.0)
    if k < nx * ny:
        sc = _direction(r, s_old, flag, diag, k, beta, first)
        s_new[k] = sc
        a = sc
        if flag[k] == 1:
            i = k // ny
            j = k - i * ny
            a = diag[k] * sc
            if i > 0:
                if flag[k - ny] == 1:
                    a -= _direction(r, s_old, flag, diag, k - ny, beta, first)
            if i < nx - 1:
                if flag[k + ny] == 1:
                    a -= _direction(r, s_old, flag, diag, k + ny, beta, first)
            if j > 0:
                if flag[k - 1] == 1:
                    a -= _direction(r, s_old, flag, diag, k - 1, beta, first)
            if j < ny - 1:
                if flag[k + 1] == 1:
                    a -= _direction(r, s_old, flag, diag, k + 1, beta, first)
            dot = sc * a
        As[k] = a
    if k == 0:
        rz[(g + 1) % 3] = 0.0
        sAs[(g + 1) % 3] = 0.0
        rmax[0] = 0.0
    wp.tile_atomic_add(sAs, wp.tile_sum(wp.tile(dot)), offset=c)


@wp.kernel
def pcg_update_kernel(
    p: wp.array(dtype=float), r: wp.array(dtype=float), s: wp.array(dtype=float), As: wp.array(dtype=float),
    flag: wp.array(dtype=wp.int32), diag: wp.array(dtype=float), n: int, block: int, it: wp.array(dtype=wp.int32),
    k_in_block: int, sAs: wp.array(dtype=float), rz: wp.array(dtype=float), rmax: wp.array(dtype=float),
):
    """alpha = r.z / s.As; p += alpha s; r -= alpha As; next r.z and max|r|."""
    k = wp.tid()
    g = it[0] + k_in_block
    c = g % 3
    dot = float(0.0)
    ra = float(0.0)
    if k < n:
        den = sAs[c]
        alpha = float(0.0)
        if wp.abs(den) > 1.0e-30:
            alpha = rz[c] / den
        p[k] = p[k] + alpha * s[k]
        rr = r[k] - alpha * As[k]
        r[k] = rr
        if flag[k] == 1:
            dot = rr * rr / diag[k]
        ra = wp.abs(rr)
        if not wp.isfinite(ra):
            ra = 1.0e30
    wp.tile_atomic_add(rz, wp.tile_sum(wp.tile(dot)), offset=(g + 1) % 3)
    m = wp.tile_max(wp.tile(ra))
    if k % block == 0:
        wp.atomic_max(rmax, 0, m[0])


@wp.kernel
def mark_open_walls_kernel(p: wp.array2d(dtype=float), flag: wp.array2d(dtype=wp.int32), tol: float, walls: int,
                           open_walls: wp.array2d(dtype=wp.int32)):
    """Open the wall faces (among the ``walls`` bits) of liquid cells under suction (P < -tol)."""
    i, j = wp.tid()
    nx = p.shape[0]
    ny = p.shape[1]
    m = int(0)
    if flag[i, j] == 1 and p[i, j] < -tol:
        if i == 0:
            m = m | OPEN_LEFT
        if i == nx - 1:
            m = m | OPEN_RIGHT
        if j == 0:
            m = m | OPEN_BOTTOM
        if j == ny - 1:
            m = m | OPEN_TOP
    open_walls[i, j] = m & walls


@wp.kernel
def reclose_walls_kernel(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), p: wp.array2d(dtype=float),
                         inv_dx: float, open_walls: wp.array2d(dtype=wp.int32)):
    """Close again the open wall faces whose projected velocity would point into the wall."""
    i, j = wp.tid()
    m = open_walls[i, j]
    if m == 0:
        return
    nx = p.shape[0]
    ny = p.shape[1]
    g = p[i, j] * inv_dx / WALL_THETA
    if (m & OPEN_LEFT) != 0 and u[0, j] - g < 0.0:
        m = m - OPEN_LEFT
    if (m & OPEN_RIGHT) != 0 and u[nx, j] + g > 0.0:
        m = m - OPEN_RIGHT
    if (m & OPEN_BOTTOM) != 0 and v[i, 0] - g < 0.0:
        m = m - OPEN_BOTTOM
    if (m & OPEN_TOP) != 0 and v[i, ny] + g > 0.0:
        m = m - OPEN_TOP
    open_walls[i, j] = m


@wp.kernel
def accumulate_stats_kernel(stats: wp.array(dtype=float), total: wp.array(dtype=float), reset: int):
    if reset == 1:
        total[0] = stats[0]
        total[1] = stats[1]
    else:
        total[0] = total[0] + stats[0]
        total[1] = stats[1]
        stats[0] = total[0]


@wp.kernel
def pcg_reset_kernel(rz: wp.array(dtype=float), sAs: wp.array(dtype=float), rmax: wp.array(dtype=float),
                     it: wp.array(dtype=wp.int32)):
    for q in range(3):
        rz[q] = 0.0
        sAs[q] = 0.0
    rmax[0] = 0.0
    it[0] = 0


@wp.kernel
def pcg_condition_kernel(rmax: wp.array(dtype=float), tol: float, it: wp.array(dtype=wp.int32), add: int,
                         max_iters: int, scale: float, cond: wp.array(dtype=wp.int32), stat: wp.array(dtype=float)):
    """cond = 1 while max|r| > tol and the iteration budget is not exhausted; records (iterations, residual)."""
    n = it[0] + add
    it[0] = n
    res = rmax[0]
    if res > tol and n < max_iters and res < 1.0e29:
        cond[0] = 1
    else:
        cond[0] = 0
    stat[0] = float(n)
    stat[1] = res * scale


# ------------------------------------------------------------------------------------- velocity update
@wp.kernel
def subtract_gradient_u_kernel(
    u: wp.array2d(dtype=float), p: wp.array2d(dtype=float), phi: wp.array2d(dtype=float),
    kappa: wp.array2d(dtype=float), open_walls: wp.array2d(dtype=wp.int32), inv_dx: float, dt_sigma: float,
    theta_min: float, kappa_max: float, valid: wp.array2d(dtype=wp.int32),
):
    i, j = wp.tid()
    nx = phi.shape[0]
    if i == 0 or i == nx:
        c = wp.min(i, nx - 1)
        liquid = phi[c, j] < 0.0
        valid[i, j] = wp.where(liquid, 1, 0)
        if not liquid:
            return  # a wall face next to air keeps its predicted value (clamped later)
        un = float(0.0)
        if i == 0 and (open_walls[c, j] & OPEN_LEFT) != 0:
            un = wp.max(u[i, j] - p[c, j] * inv_dx / WALL_THETA, 0.0)
        if i == nx and (open_walls[c, j] & OPEN_RIGHT) != 0:
            un = wp.min(u[i, j] + p[c, j] * inv_dx / WALL_THETA, 0.0)
        u[i, j] = un
        return
    pl = phi[i - 1, j]
    pr = phi[i, j]
    if pl < 0.0 and pr < 0.0:
        u[i, j] = u[i, j] - (p[i, j] - p[i - 1, j]) * inv_dx
        valid[i, j] = 1
    elif pl < 0.0:
        th = interface_theta(pl, pr, theta_min)
        ps = surface_pressure(kappa[i - 1, j], kappa[i, j], th, dt_sigma, kappa_max)
        u[i, j] = u[i, j] - (ps - p[i - 1, j]) * inv_dx / th
        valid[i, j] = 1
    elif pr < 0.0:
        th = interface_theta(pr, pl, theta_min)
        ps = surface_pressure(kappa[i, j], kappa[i - 1, j], th, dt_sigma, kappa_max)
        u[i, j] = u[i, j] - (p[i, j] - ps) * inv_dx / th
        valid[i, j] = 1
    else:
        valid[i, j] = 0


@wp.kernel
def subtract_gradient_v_kernel(
    v: wp.array2d(dtype=float), p: wp.array2d(dtype=float), phi: wp.array2d(dtype=float),
    kappa: wp.array2d(dtype=float), open_walls: wp.array2d(dtype=wp.int32), inv_dx: float, dt_sigma: float,
    theta_min: float, kappa_max: float, valid: wp.array2d(dtype=wp.int32),
):
    i, j = wp.tid()
    ny = phi.shape[1]
    if j == 0 or j == ny:
        c = wp.min(j, ny - 1)
        liquid = phi[i, c] < 0.0
        valid[i, j] = wp.where(liquid, 1, 0)
        if not liquid:
            return  # a wall face next to air keeps its predicted value (clamped later)
        vn = float(0.0)
        if j == 0 and (open_walls[i, c] & OPEN_BOTTOM) != 0:
            vn = wp.max(v[i, j] - p[i, c] * inv_dx / WALL_THETA, 0.0)
        if j == ny and (open_walls[i, c] & OPEN_TOP) != 0:
            vn = wp.min(v[i, j] + p[i, c] * inv_dx / WALL_THETA, 0.0)
        v[i, j] = vn
        return
    pb = phi[i, j - 1]
    pt = phi[i, j]
    if pb < 0.0 and pt < 0.0:
        v[i, j] = v[i, j] - (p[i, j] - p[i, j - 1]) * inv_dx
        valid[i, j] = 1
    elif pb < 0.0:
        th = interface_theta(pb, pt, theta_min)
        ps = surface_pressure(kappa[i, j - 1], kappa[i, j], th, dt_sigma, kappa_max)
        v[i, j] = v[i, j] - (ps - p[i, j - 1]) * inv_dx / th
        valid[i, j] = 1
    elif pt < 0.0:
        th = interface_theta(pt, pb, theta_min)
        ps = surface_pressure(kappa[i, j], kappa[i, j - 1], th, dt_sigma, kappa_max)
        v[i, j] = v[i, j] - (p[i, j] - ps) * inv_dx / th
        valid[i, j] = 1
    else:
        valid[i, j] = 0


@wp.kernel
def divergence_kernel(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), phi: wp.array2d(dtype=float),
                      inv_dx: float, out: wp.array2d(dtype=float)):
    """Velocity divergence on liquid cells (0 on air cells)."""
    i, j = wp.tid()
    if phi[i, j] < 0.0:
        out[i, j] = (u[i + 1, j] - u[i, j] + v[i, j + 1] - v[i, j]) * inv_dx
    else:
        out[i, j] = 0.0


# --------------------------------------------------------------------------------------------- solver
@dataclass
class PressureConfig:
    """Pressure solve parameters.

    ``tol`` is an absolute bound on the divergence residual (1/s) in the max norm, tested every
    ``check_every`` CG iterations (on the device).
    """

    tol: float = 1.0e-4
    max_iters: int = 3000
    check_every: int = 4               # CG iterations between convergence tests (even)
    theta_min: float = 0.05
    kappa_max_dx: float = 0.5          # |kappa| <= kappa_max_dx / dx
    curvature_smoothing: int = 0       # 3x3 binomial passes on kappa near the interface
    wall_separation: str = "lid"       # unilateral walls: none | lid | all (needs the PFM-LS ghost layer, pad > 0)
    warm_start: bool = True
    use_graph: bool = True             # CUDA: whole CG loop as one graph with a device-side while node


class PressureSolver:
    """Ghost-fluid Poisson assembly, Jacobi-PCG solve and velocity update on a MAC grid."""

    def __init__(self, nx: int, ny: int, dx: float, cfg: PressureConfig | None = None, device=None):
        self.nx, self.ny, self.dx = nx, ny, dx
        self.cfg = cfg or PressureConfig()
        if self.cfg.check_every < 2 or self.cfg.check_every % 2:
            raise ValueError("check_every must be an even number >= 2")
        if self.cfg.wall_separation not in WALL_SETS:
            raise ValueError(f"wall_separation must be one of {list(WALL_SETS)}")
        self.device = wp.get_device(device)
        shp = (nx, ny)
        z = lambda dt=float: wp.zeros(shp, dtype=dt, device=self.device)  # noqa: E731
        self.p = z()
        self.kappa = z()
        self._kappa_tmp = z()
        self.flag = z(wp.int32)
        self.diag = z()
        self.rhs = z()
        self.r = z()
        self.s = z()
        self.s2 = z()
        self.As = z()
        self.div = z()
        self.open_walls = z(wp.int32)
        dev = self.device
        self.rz = wp.zeros(3, dtype=float, device=dev)
        self.sAs = wp.zeros(3, dtype=float, device=dev)
        self.rmax = wp.zeros(1, dtype=float, device=dev)
        self.it = wp.zeros(1, dtype=wp.int32, device=dev)
        self.cond = wp.zeros(1, dtype=wp.int32, device=dev)
        #: device array ``(iterations, max-norm divergence residual)`` of the last solve
        self.stats = wp.zeros(2, dtype=float, device=dev)
        self._stats_total = wp.zeros(2, dtype=float, device=dev)
        self._flat = {k: getattr(self, k).flatten() for k in ("p", "rhs", "flag", "diag", "r", "s", "s2", "As")}
        self._n = nx * ny
        self._block = BLOCK if dev.is_cuda else 1
        self._dim = ((self._n + BLOCK - 1) // BLOCK) * BLOCK if dev.is_cuda else self._n
        self._use_graph = self.cfg.use_graph and dev.is_cuda
        self._graph = None

    # -------------------------------------------------------------------------------------------------
    @property
    def last_iters(self) -> int:
        """CG iterations of the last solve (host read; diagnostics only)."""
        return int(self.stats.numpy()[0])

    @property
    def last_residual(self) -> float:
        """Max-norm divergence residual (1/s) of the last solve (host read; diagnostics only)."""
        return float(self.stats.numpy()[1])

    def compute_curvature(self, phi: wp.array2d, source: wp.array2d | None = None) -> wp.array2d:
        """Curvature of ``source`` (default ``phi``), optionally smoothed near the interface of ``phi``."""
        src = phi if source is None else source
        wp.launch(curvature_kernel, dim=(self.nx, self.ny), inputs=[src, 1.0 / self.dx], outputs=[self.kappa],
                  device=self.device)
        for _ in range(self.cfg.curvature_smoothing):
            wp.launch(smooth_band_kernel, dim=(self.nx, self.ny),
                      inputs=[self.kappa, phi, 3.0 * self.dx], outputs=[self._kappa_tmp], device=self.device)
            wp.copy(self.kappa, self._kappa_tmp)
        return self.kappa

    def project(self, u: wp.array2d, v: wp.array2d, phi: wp.array2d, dt: float, sigma: float, rho: float,
                valid_u: wp.array2d, valid_v: wp.array2d, curvature_source: wp.array2d | None = None,
                separation: bool | None = None) -> None:
        """Make ``(u, v)`` discretely divergence-free on liquid cells; fills face validity masks.

        ``curvature_source`` optionally replaces ``phi`` as the field whose curvature enters the Laplace
        pressure (e.g. a redistanced copy of a level set whose sub-cell values are noisy).  ``separation=False``
        keeps every wall closed (otherwise ``cfg.wall_separation`` selects the unilateral walls).  The wall faces of
        ``u, v`` must hold the predicted velocity (they are used only where a wall opens).
        """
        walls = WALL_SETS[self.cfg.wall_separation] if separation is not False else 0
        cfg, dev, dim = self.cfg, self.device, (self.nx, self.ny)
        dt_sigma = dt * sigma / rho
        kmax = cfg.kappa_max_dx / self.dx
        if sigma != 0.0:
            self.compute_curvature(phi, curvature_source)
        else:
            self.kappa.zero_()
        if not cfg.warm_start:
            self.p.zero_()
        self.open_walls.zero_()
        build = [phi, self.kappa, u, v, self.open_walls, self.dx, dt_sigma, cfg.theta_min, kmax]
        wp.launch(build_system_kernel, dim=dim, inputs=build, outputs=[self.flag, self.diag, self.rhs, self.p],
                  device=dev)
        self._pcg()
        if walls != 0:
            wp.launch(accumulate_stats_kernel, dim=1, inputs=[self.stats, self._stats_total, 1], device=dev)
            wp.launch(mark_open_walls_kernel, dim=dim,
                      inputs=[self.p, self.flag, cfg.tol * self.dx * self.dx, walls], outputs=[self.open_walls],
                      device=dev)
            for k in range(2):  # open the faces under suction, then close those that would push into the wall
                wp.launch(build_system_kernel, dim=dim, inputs=build,
                          outputs=[self.flag, self.diag, self.rhs, self.p], device=dev)
                self._pcg()
                wp.launch(accumulate_stats_kernel, dim=1, inputs=[self.stats, self._stats_total, 0], device=dev)
                if k == 0:  # after the last solve the clamp of the velocity update closes those faces anyway
                    wp.launch(reclose_walls_kernel, dim=dim, inputs=[u, v, self.p, 1.0 / self.dx, self.open_walls],
                              device=dev)
        inv_dx = 1.0 / self.dx
        grad = [self.p, phi, self.kappa, self.open_walls, inv_dx, dt_sigma, cfg.theta_min, kmax]
        wp.launch(subtract_gradient_u_kernel, dim=u.shape, inputs=[u] + grad + [valid_u], device=dev)
        wp.launch(subtract_gradient_v_kernel, dim=v.shape, inputs=[v] + grad + [valid_v], device=dev)

    # ------------------------------------------------------------------------------------------- PCG
    def _launch(self, kernel, inputs):
        wp.launch(kernel, dim=self._dim, inputs=inputs, block_dim=self._block, device=self.device)

    def _start(self) -> None:
        f, dev = self._flat, self.device
        wp.launch(pcg_reset_kernel, dim=1, inputs=[self.rz, self.sAs, self.rmax, self.it], device=dev)
        self._launch(pcg_start_kernel, [f["p"], f["rhs"], f["flag"], f["diag"], self.nx, self.ny, self._block,
                                        f["r"], self.rz, self.rmax])
        self._condition(0)

    def _condition(self, add: int) -> None:
        tol = self.cfg.tol * self.dx * self.dx  # residual of the dx^2-scaled system
        wp.launch(pcg_condition_kernel, dim=1,
                  inputs=[self.rmax, tol, self.it, add, self.cfg.max_iters, 1.0 / (self.dx * self.dx), self.cond,
                          self.stats], device=self.device)

    def _iterations(self) -> None:
        """``check_every`` PCG iterations (two fused kernels each) followed by the convergence test."""
        f, nx, ny, n = self._flat, self.nx, self.ny, self._n
        for k in range(self.cfg.check_every):
            s_old, s_new = (f["s"], f["s2"]) if k % 2 == 0 else (f["s2"], f["s"])
            self._launch(pcg_matvec_kernel, [f["r"], s_old, f["flag"], f["diag"], nx, ny, self.it, k, s_new, f["As"],
                                             self.sAs, self.rz, self.rmax])
            self._launch(pcg_update_kernel, [f["p"], f["r"], s_new, f["As"], f["flag"], f["diag"], n, self._block,
                                             self.it, k, self.sAs, self.rz, self.rmax])
        self._condition(self.cfg.check_every)

    def _pcg(self) -> None:
        if self.device.is_capturing:  # inside a captured solver step: the loop becomes a conditional node
            self._start()
            wp.capture_while(self.cond, self._iterations)
            return
        if not self._use_graph:
            self._start()
            while int(self.cond.numpy()[0]) == 1:
                self._iterations()
            return
        if self._graph is None:
            self._start()          # eager run first: compiles and loads the modules
            self._iterations()
            with wp.ScopedCapture(device=self.device) as cap:
                self._start()
                wp.capture_while(self.cond, self._iterations)
            self._graph = cap.graph
        wp.capture_launch(self._graph)

    def max_divergence(self, u: wp.array2d, v: wp.array2d, phi: wp.array2d) -> float:
        """Max |div u| over liquid cells (host sync; diagnostics only)."""
        wp.launch(divergence_kernel, dim=(self.nx, self.ny), inputs=[u, v, phi, 1.0 / self.dx], outputs=[self.div],
                  device=self.device)
        return float(np.abs(self.div.numpy()).max())
