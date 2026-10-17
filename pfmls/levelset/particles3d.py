"""Warp kernels for the 3D particle side of PFM-LS.

Particle state (slots ``0 .. capacity-1``, ``valid`` marks the live particles):

* ``x`` position, ``phi`` level-set value (constant between long reinitialisations),
* ``g`` unit gradient, ``g0`` gradient at the last long reinitialisation (or at birth),
* ``Js`` short flow-map Jacobian ``d psi_s / dx`` (rows), ``Jinit`` long-map Jacobian at the last short
  reinitialisation, so that the long map is ``J_long = Jinit Js``,
* ``C0`` gradient of the unit normal at the last short reinitialisation and ``C = Js^T C0 Js``,
* ``valid``: 0 free slot, 1 lattice particle (placed at a long reinitialisation), 2 particle created by
  adaptive seeding (a *newborn*).

Storage is sized by the narrow band, not by the grid: at every long reinitialisation the *seed cells*
(cells whose centre lies within ``band + lattice radius`` of the interface) are compacted by a prefix
sum, and seed cell ``b`` owns the ``ppc`` consecutive slots ``b ppc .. (b + 1) ppc - 1`` (one per point of
its ``a x a x a`` lattice). Adaptive seeding appends particles after the last used slot; slot compaction
(:func:`mark_kept_slots` and the gather kernels) reorders the live particles into cell order and frees the rest.

Particle -> point reconstruction uses the extrapolation of the original method,
``phi_p + normalize(g_p + C_p d) . d`` with ``d = x - x_p``.
"""

from __future__ import annotations

import warp as wp

from ..core.celllist3d import CellListView3D, cell_id3, cell_of3, cell_range3
from ..core.kernels import N2, dN2, sgn
from ..core.kernels3d import bspline_sample_grad3, cell_center3, hermite3, safe_normalize3, trilerp
from ..flows.analytic3d import FlowParams3D


#: How the reconstructions treat the particles created by adaptive seeding (``ParticleView3D.newborns``).
#: :func:`p2g_point3` implements all three; :func:`knn_interp3` and :func:`p2g_cells3` (reinitialisation) skip
#: newborns for ``NEWBORNS_OFF`` and treat ``NEWBORNS_FALLBACK`` like ``NEWBORNS_MIXED``.
NEWBORNS_MIXED = wp.constant(0)     # averaged together with the lattice particles
NEWBORNS_OFF = wp.constant(1)       # ignored
NEWBORNS_FALLBACK = wp.constant(2)  # used only at points that no lattice particle covers


@wp.struct
class ParticleView3D:
    """Read-only particle data used by the reconstruction functions (``radius`` in physical units;
    ``newborns`` is one of the ``NEWBORNS_*`` modes; ``valid`` tells newborns (2) from lattice particles)."""

    x: wp.array(dtype=wp.vec3)
    phi: wp.array(dtype=float)
    g: wp.array(dtype=wp.vec3)
    C: wp.array(dtype=wp.mat33)
    valid: wp.array(dtype=wp.int32)
    radius: float
    newborns: int


@wp.func
def extrapolate3(phi_p: float, g_p: wp.vec3, C_p: wp.mat33, d: wp.vec3):
    """Particle extrapolation ``(phi_p + n . d, n)`` with ``n = normalize(g_p + C_p d)``.

    For a signed distance this equals ``phi_p + g.d + d^T C d`` to second order, i.e. twice the Taylor curvature
    term when ``C = grad n`` (``hessian_scale = 1``); see docs/method.md, section 3."""
    n = safe_normalize3(g_p + C_p @ d)
    return phi_p + wp.dot(n, d), n


@wp.func
def p2g_weight3(d: wp.vec3, inv_h: float) -> float:
    """Reconstruction weight: 1 (unweighted mean) if ``inv_h == 0``, else ``N2(d_x/h) N2(d_y/h) N2(d_z/h)``."""
    if inv_h == 0.0:
        return 1.0
    return N2(d[0] * inv_h) * N2(d[1] * inv_h) * N2(d[2] * inv_h)


@wp.func
def decode_cell(c: int, ny: int, nz: int):
    """Cell coordinates of the flat id ``c = (ci ny + cj) nz + ck``."""
    return c // (ny * nz), (c // nz) % ny, c % nz


@wp.func
def lattice_point3(ci: int, cj: int, ck: int, sub: int, a: int, dx: float) -> wp.vec3:
    """Point ``sub`` of the ``a x a x a`` lattice of cell ``(ci, cj, ck)``."""
    kx = sub % a
    ky = (sub // a) % a
    kz = sub // (a * a)
    inv_a = 1.0 / float(a)
    return wp.vec3(float(ci) + (float(kx) + 0.5) * inv_a, float(cj) + (float(ky) + 0.5) * inv_a,
                   float(ck) + (float(kz) + 0.5) * inv_a) * dx


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


@wp.kernel
def set_step_constants3(clock: wp.array(dtype=float), t: float, dt: float, params: wp.array(dtype=FlowParams3D),
                        p: FlowParams3D, istep: wp.array(dtype=wp.int32), i: int):
    """Per-step device constants read by the (graph-captured) step: time, dt, flow parameters, step index."""
    clock[0] = t
    clock[1] = dt
    params[0] = p
    istep[0] = i


@wp.kernel
def mark_cells_below(phi: wp.array3d(dtype=float), threshold: float, flags: wp.array(dtype=wp.int32)):
    """flags[c] = 1 where |phi| < threshold (flat cell ids)."""
    i, j, k = wp.tid()
    c = (i * phi.shape[1] + j) * phi.shape[2] + k
    if wp.abs(phi[i, j, k]) < threshold:
        flags[c] = 1
    else:
        flags[c] = 0


@wp.kernel
def compact_cells(flags: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32),
                  cell_list: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32), num_cells: int):
    """cell_list[offsets[c]] = c for flagged entries (``offsets`` = exclusive scan); count[0] = total. Entries
    past the end of ``cell_list`` are dropped (``count`` still holds the total)."""
    c = wp.tid()
    if flags[c] != 0 and offsets[c] < cell_list.shape[0]:
        cell_list[offsets[c]] = c
    if c == num_cells - 1:
        count[0] = offsets[c] + flags[c]


@wp.kernel
def seed_lattice3(seed_cells: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32), ppc: int, a: int,
                  ny: int, nz: int, dx: float, x: wp.array(dtype=wp.vec3), valid: wp.array(dtype=wp.int32),
                  n_used: wp.array(dtype=wp.int32)):
    """Place the lattice points of the seed cells into their slots (all valid); other slots invalid."""
    s = wp.tid()
    b = s // ppc
    if b < count[0]:
        ci, cj, ck = decode_cell(seed_cells[b], ny, nz)
        x[s] = lattice_point3(ci, cj, ck, s % ppc, a, dx)
        valid[s] = 1
    else:
        valid[s] = 0
    if s == 0:
        n_used[0] = wp.min(count[0] * ppc, x.shape[0])


@wp.kernel
def sample_grid_to_particles3(phi_grid: wp.array3d(dtype=float), x: wp.array(dtype=wp.vec3),
                              valid: wp.array(dtype=wp.int32), inv_dx: float, phi: wp.array(dtype=float)):
    p = wp.tid()
    if valid[p] != 0:
        phi[p] = trilerp(phi_grid, x[p] * inv_dx - wp.vec3(0.5, 0.5, 0.5))


# ---------------------------------------------------------------------------
# Neighbour sums
# ---------------------------------------------------------------------------


@wp.func
def _shepard_numerator(xi: wp.vec3, x: wp.array(dtype=wp.vec3), phi: wp.array(dtype=float),
                       valid: wp.array(dtype=wp.int32), skip_newborns: int, cells: CellListView3D, dx: float):
    """``(sum w, (sum dw phi) (sum w) - (sum dw) (sum w phi))`` over the particles in the 3x3x3 cells around
    ``xi`` (without newborns if ``skip_newborns``), ``w = N2(r_x) N2(r_y) N2(r_z)``, ``r = (xi - x_j) / dx``: the
    second term is ``(sum w)^2`` times the gradient of the Shepard interpolant ``sum w phi / sum w``."""
    inv = 1.0 / dx
    bi, bj, bk = cell_of3(xi, cells.inv_dx)
    s0 = float(0.0)
    s1 = wp.vec3(0.0, 0.0, 0.0)
    p0 = float(0.0)
    p1 = wp.vec3(0.0, 0.0, 0.0)
    for oi in range(-1, 2):
        ci = bi + oi
        if ci < 0 or ci >= cells.nx:
            continue
        for oj in range(-1, 2):
            cj = bj + oj
            if cj < 0 or cj >= cells.ny:
                continue
            for ok in range(-1, 2):
                ck = bk + ok
                if ck < 0 or ck >= cells.nz:
                    continue
                c = cell_id3(cells, ci, cj, ck)
                for kk in range(cells.start[c], cells.end[c]):
                    j = cells.ids[kk]
                    if skip_newborns != 0 and valid[j] == 2:
                        continue
                    r = (xi - x[j]) * inv
                    wx = N2(r[0])
                    wy = N2(r[1])
                    wz = N2(r[2])
                    w = wx * wy * wz
                    dw = wp.vec3(dN2(r[0]) * wy * wz, wx * dN2(r[1]) * wz, wx * wy * dN2(r[2])) * inv
                    pj = phi[j]
                    s0 += w
                    s1 += dw
                    p0 += w * pj
                    p1 += dw * pj
    return s0, p1 * s0 - s1 * p0


@wp.kernel
def shepard_gradient3(
    x: wp.array(dtype=wp.vec3),
    phi: wp.array(dtype=float),
    valid: wp.array(dtype=wp.int32),
    cells: CellListView3D,
    dx: float,
    skip_newborns: int,
    g: wp.array(dtype=wp.vec3),
):
    """g_i = normalize(grad of the Shepard interpolant sum w phi / sum w), w = N2(r_x) N2(r_y) N2(r_z).

    Neighbours: valid particles in the 3x3x3 cells around particle i (i included; newborns excluded if
    ``skip_newborns``, and then their own gradient is not recomputed).
    """
    i = wp.tid()
    if valid[i] == 0 or (skip_newborns != 0 and valid[i] == 2):
        return
    s0, num = _shepard_numerator(x[i], x, phi, valid, skip_newborns, cells, dx)
    if s0 > 0.0:
        g[i] = safe_normalize3(num)


@wp.kernel
def shepard_gradient3_listed(
    ids: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    x: wp.array(dtype=wp.vec3),
    phi: wp.array(dtype=float),
    valid: wp.array(dtype=wp.int32),
    cells: CellListView3D,
    dx: float,
    skip_newborns: int,
    g: wp.array(dtype=wp.vec3),
):
    """:func:`shepard_gradient3` for the particles ``ids[0 .. count - 1]`` only (launched over the capacity;
    the threads past ``count`` return at once)."""
    k = wp.tid()
    if k >= count[0]:
        return
    i = ids[k]
    s0, num = _shepard_numerator(x[i], x, phi, valid, skip_newborns, cells, dx)
    if s0 > 0.0:
        g[i] = safe_normalize3(num)


@wp.kernel
def mark_p2g_particles(x: wp.array(dtype=wp.vec3), phi: wp.array(dtype=float), valid: wp.array(dtype=wp.int32),
                       cells: CellListView3D, dx: float, radius: float, phi_below: float, skip_newborns: int,
                       flags: wp.array(dtype=wp.int32)):
    """flags[p] = 1 for the valid particles within ``radius`` of their own cell centre (the particles that
    :func:`p2g_cells3` reads; ``radius`` carries a small safety margin) or with ``|phi_p| < phi_below``
    (newborns never if ``skip_newborns``)."""
    p = wp.tid()
    flags[p] = 0
    if valid[p] == 0 or (skip_newborns != 0 and valid[p] == 2):
        return
    if wp.abs(phi[p]) < phi_below:
        flags[p] = 1
        return
    xp = x[p]
    ci, cj, ck = cell_of3(xp, cells.inv_dx)
    if ci < 0 or ci >= cells.nx or cj < 0 or cj >= cells.ny or ck < 0 or ck >= cells.nz:
        return
    if wp.length(cell_center3(ci, cj, ck, dx) - xp) <= radius:
        flags[p] = 1


@wp.func
def knn_interp3(xq: wp.vec3, parts: ParticleView3D, cells: CellListView3D, filt: wp.vec3, use_filter: int):
    """Particle interpolation of (phi, unit normal) at ``xq`` from particles within ``r = parts.radius``.

    A particle j is accepted if ``|xq - x_j| < r`` and (optionally) ``filt . n_j > 0``; weights are
    ``N2(d_x / r) N2(d_y / r) N2(d_z / r)``; newborns are skipped for ``parts.newborns == NEWBORNS_OFF``.
    Returns ``(phi, n, ok)``.
    """
    r = parts.radius
    i0, i1, j0, j1, k0, k1 = cell_range3(xq, r, cells)
    wsum = float(0.0)
    acc = float(0.0)
    accn = wp.vec3(0.0, 0.0, 0.0)
    s = 1.0 / r
    for ci in range(i0, i1 + 1):
        for cj in range(j0, j1 + 1):
            for ck in range(k0, k1 + 1):
                c = cell_id3(cells, ci, cj, ck)
                for kk in range(cells.start[c], cells.end[c]):
                    p = cells.ids[kk]
                    d = xq - parts.x[p]
                    if wp.length(d) < r:
                        if parts.newborns == NEWBORNS_OFF and parts.valid[p] == 2:
                            continue
                        v, n = extrapolate3(parts.phi[p], parts.g[p], parts.C[p], d)
                        if use_filter == 0 or wp.dot(filt, n) > 0.0:
                            w = N2(d[0] * s) * N2(d[1] * s) * N2(d[2] * s)
                            wsum += w
                            acc += w * v
                            accn += w * n
    if wsum > 0.0:
        return acc / wsum, safe_normalize3(accn / wsum), 1
    return 0.0, wp.vec3(0.0, 0.0, 0.0), 0


@wp.kernel
def p2g_cells3(
    parts: ParticleView3D,
    cells: CellListView3D,
    dx: float,
    w_inv_h: float,
    out_phi: wp.array3d(dtype=float),
    out_g: wp.array3d(dtype=wp.vec3),
    out_m: wp.array3d(dtype=wp.int32),
):
    """Cell-centre reconstruction from the particles within ``parts.radius`` (< dx/2, so only the cell's
    own list is scanned; newborns skipped for ``NEWBORNS_OFF``): phi, averaged unit normal and coverage mark
    (0 and phi = g = 0 if uncovered)."""
    i, j, k = wp.tid()
    xc = cell_center3(i, j, k, dx)
    c = cell_id3(cells, i, j, k)
    wsum = float(0.0)
    acc = float(0.0)
    accn = wp.vec3(0.0, 0.0, 0.0)
    for kk in range(cells.start[c], cells.end[c]):
        p = cells.ids[kk]
        d = xc - parts.x[p]
        if wp.length(d) <= parts.radius:
            if parts.newborns == NEWBORNS_OFF and parts.valid[p] == 2:
                continue
            v, n = extrapolate3(parts.phi[p], parts.g[p], parts.C[p], d)
            w = p2g_weight3(d, w_inv_h)
            wsum += w
            acc += w * v
            accn += w * n
    if wsum > 0.0:
        out_phi[i, j, k] = acc / wsum
        out_g[i, j, k] = safe_normalize3(accn / wsum)
        out_m[i, j, k] = 1
    else:
        out_phi[i, j, k] = 0.0
        out_g[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
        out_m[i, j, k] = 0


@wp.func
def p2g_point3(xc: wp.vec3, parts: ParticleView3D, cells: CellListView3D, w_inv_h: float):
    """Weighted Taylor sum at ``xc`` over particles within ``parts.radius``; returns ``(sum w phi, sum w)``.

    Newborns (``valid = 2``) enter according to ``parts.newborns``: together with the lattice particles
    (``NEWBORNS_MIXED``), not at all (``NEWBORNS_OFF``) or only if no lattice particle lies within the radius
    (``NEWBORNS_FALLBACK``)."""
    r = parts.radius
    i0, i1, j0, j1, k0, k1 = cell_range3(xc, r, cells)
    wsum = float(0.0)
    acc = float(0.0)
    wsum_nb = float(0.0)
    acc_nb = float(0.0)
    for ci in range(i0, i1 + 1):
        for cj in range(j0, j1 + 1):
            for ck in range(k0, k1 + 1):
                c = cell_id3(cells, ci, cj, ck)
                for kk in range(cells.start[c], cells.end[c]):
                    p = cells.ids[kk]
                    d = xc - parts.x[p]
                    if wp.length(d) <= r:
                        newborn = parts.valid[p] == 2 and parts.newborns != NEWBORNS_MIXED
                        if newborn and parts.newborns == NEWBORNS_OFF:
                            continue
                        v, n = extrapolate3(parts.phi[p], parts.g[p], parts.C[p], d)
                        w = p2g_weight3(d, w_inv_h)
                        if newborn:
                            wsum_nb += w
                            acc_nb += w * v
                        else:
                            wsum += w
                            acc += w * v
    if wsum == 0.0:
        return acc_nb, wsum_nb
    return acc, wsum


@wp.kernel
def composite_grid3(
    parts: ParticleView3D,
    cells: CellListView3D,
    dx: float,
    w_inv_h: float,
    phi_grid: wp.array3d(dtype=float),
    phi_out: wp.array3d(dtype=float),
):
    """phi_out = particle reconstruction where a particle lies within ``parts.radius``, else phi_grid."""
    i, j, k = wp.tid()
    acc, wsum = p2g_point3(cell_center3(i, j, k, dx), parts, cells, w_inv_h)
    if wsum > 0.0:
        phi_out[i, j, k] = acc / wsum
    else:
        phi_out[i, j, k] = phi_grid[i, j, k]


@wp.kernel
def reconstruct_fine3(
    parts: ParticleView3D,
    cells: CellListView3D,
    h_fine: float,
    inv_dx: float,
    w_inv_h: float,
    phi_grid: wp.array3d(dtype=float),
    phi_fine: wp.array3d(dtype=float),
):
    """Fine-grid field: particle reconstruction where covered, trilinear grid sample elsewhere."""
    i, j, k = wp.tid()
    xc = cell_center3(i, j, k, h_fine)
    acc, wsum = p2g_point3(xc, parts, cells, w_inv_h)
    if wsum > 0.0:
        phi_fine[i, j, k] = acc / wsum
    else:
        phi_fine[i, j, k] = trilerp(phi_grid, xc * inv_dx - wp.vec3(0.5, 0.5, 0.5))


# ---------------------------------------------------------------------------
# Long reinitialisation: reseeding with Hermite / particle blending
# ---------------------------------------------------------------------------


@wp.kernel
def reseed3(
    old: ParticleView3D,
    cells: CellListView3D,
    phi_r: wp.array3d(dtype=float),
    g_r: wp.array3d(dtype=wp.vec3),
    X_r: wp.array3d(dtype=wp.vec4),
    seed_cells: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    ppc: int,
    a: int,
    dx: float,
    band: float,
    e1: float,
    e2: float,
    gate: float,
    grid_gate: float,
    x: wp.array(dtype=wp.vec3),
    phi: wp.array(dtype=float),
    g: wp.array(dtype=wp.vec3),
    g0: wp.array(dtype=wp.vec3),
    valid: wp.array(dtype=wp.int32),
    n_used: wp.array(dtype=wp.int32),
):
    """Re-place every slot of the seed cells on its lattice point and assign (phi, g).

    Grid value: tricubic Hermite interpolation of the redistanced grid. Particle value: KNN interpolation
    of the old particles filtered by the Hermite normal, used when it is valid, ``|phi_P| < gate dx`` and
    ``|phi_H| <= grid_gate dx``, with ``gamma = n_H . n_P``: gamma > e1 -> particle, e2 < gamma <= e1 ->
    linear blend, else grid. The slot is valid iff ``|phi| < band``; slots past the seed cells are freed.
    """
    s = wp.tid()
    if s == 0:
        n_used[0] = wp.min(count[0] * ppc, x.shape[0])
    b = s // ppc
    if b >= count[0]:
        valid[s] = 0
        return
    ci, cj, ck = decode_cell(seed_cells[b], cells.ny, cells.nz)
    xp = lattice_point3(ci, cj, ck, s % ppc, a, dx)
    q = xp / dx - wp.vec3(0.5, 0.5, 0.5)
    phi_h, grad_h = hermite3(phi_r, g_r, X_r, q, dx)
    g_h = safe_normalize3(grad_h)
    phi_p, g_p, ok = knn_interp3(xp, old, cells, g_h, 1)
    val = phi_h
    nrm = g_h
    if ok == 1 and wp.abs(phi_p) < gate * dx and wp.abs(phi_h) <= grid_gate * dx:
        gamma = wp.dot(g_h, g_p)
        if gamma > e1:
            val = phi_p
            nrm = g_p
        elif gamma > e2:
            val = ((e1 - gamma) * phi_h + (gamma - e2) * phi_p) / (e1 - e2)
            nrm = ((e1 - gamma) * g_h + (gamma - e2) * g_p) / (e1 - e2)
    nrm = safe_normalize3(nrm)
    x[s] = xp
    phi[s] = val
    g[s] = nrm
    g0[s] = nrm
    if wp.abs(val) < band:
        valid[s] = 1
    else:
        valid[s] = 0


# ---------------------------------------------------------------------------
# Short reinitialisation: normal-gradient (C0) from MAC-averaged normals
# ---------------------------------------------------------------------------


@wp.kernel
def split_to_faces3(g: wp.array3d(dtype=wp.vec3), fx: wp.array3d(dtype=float), fy: wp.array3d(dtype=float),
                    fz: wp.array3d(dtype=float)):
    """Average cell normals to faces, e.g. fx[i,j,k] = (g[i-1,j,k].x + g[i,j,k].x)/2 (clamped)."""
    i, j, k = wp.tid()
    nx = g.shape[0]
    ny = g.shape[1]
    nz = g.shape[2]
    if j < ny and k < nz:
        fx[i, j, k] = 0.5 * (g[wp.min(i, nx - 1), j, k][0] + g[wp.clamp(i - 1, 0, nx - 1), j, k][0])
    if i < nx and k < nz:
        fy[i, j, k] = 0.5 * (g[i, wp.min(j, ny - 1), k][1] + g[i, wp.clamp(j - 1, 0, ny - 1), k][1])
    if i < nx and j < ny:
        fz[i, j, k] = 0.5 * (g[i, j, wp.min(k, nz - 1)][2] + g[i, j, wp.clamp(k - 1, 0, nz - 1)][2])


@wp.kernel
def normal_gradient_from_faces3(
    x: wp.array(dtype=wp.vec3),
    valid: wp.array(dtype=wp.int32),
    fx: wp.array3d(dtype=float),
    fy: wp.array3d(dtype=float),
    fz: wp.array3d(dtype=float),
    inv_dx: float,
    scale: float,
    C0: wp.array(dtype=wp.mat33),
):
    """C0 = scale * d n / dx from the quadratic B-spline of the face-averaged normals (rows: n_x, n_y, n_z)."""
    p = wp.tid()
    if valid[p] == 0:
        return
    xp = x[p]
    vx, dgx = bspline_sample_grad3(fx, xp, inv_dx, wp.vec3(0.0, 0.5, 0.5))
    vy, dgy = bspline_sample_grad3(fy, xp, inv_dx, wp.vec3(0.5, 0.0, 0.5))
    vz, dgz = bspline_sample_grad3(fz, xp, inv_dx, wp.vec3(0.5, 0.5, 0.0))
    C0[p] = scale * wp.mat33(dgx[0], dgx[1], dgx[2], dgy[0], dgy[1], dgy[2], dgz[0], dgz[1], dgz[2])


@wp.kernel
def normal_gradient_particles3(
    x: wp.array(dtype=wp.vec3),
    g: wp.array(dtype=wp.vec3),
    valid: wp.array(dtype=wp.int32),
    cells: CellListView3D,
    radius: float,
    scale: float,
    skip_newborns: int,
    C0: wp.array(dtype=wp.mat33),
):
    """C0 = scale * P sym(G) P, with G the weighted least-squares gradient of the particle normals.

    ``G = B M^-1`` with ``M = sum w d d^T``, ``B = sum w (n_j - n_i) d^T`` over the particles ``j`` within
    ``radius`` of particle ``i`` whose normals face the same side (``n_j . n_i > 0``, so the two sides of a
    thin sheet are not mixed), ``d = x_j - x_i``, ``w = (1 - |d|^2 / r^2)^2``; ``P = I - n_i n_i^T`` removes the
    normal components (the gradient of a unit normal field is tangential and, for a signed distance,
    symmetric). Resolves curvature radii down to about ``radius``. With ``skip_newborns`` the newborns are
    neither neighbours nor fitted (they keep the ``C0`` of their birth).
    """
    i = wp.tid()
    if valid[i] == 0 or (skip_newborns != 0 and valid[i] == 2):
        return
    xi = x[i]
    ni = g[i]
    r = radius
    r2 = r * r
    i0, i1, j0, j1, k0, k1 = cell_range3(xi, r, cells)
    M = wp.mat33(0.0)
    B = wp.mat33(0.0)
    wsum = float(0.0)
    for ci in range(i0, i1 + 1):
        for cj in range(j0, j1 + 1):
            for ck in range(k0, k1 + 1):
                c = cell_id3(cells, ci, cj, ck)
                for kk in range(cells.start[c], cells.end[c]):
                    p = cells.ids[kk]
                    d = x[p] - xi
                    l2 = wp.dot(d, d)
                    if l2 >= r2 or p == i or (skip_newborns != 0 and valid[p] == 2):
                        continue
                    nj = g[p]
                    if wp.dot(nj, ni) > 0.0:
                        q = 1.0 - l2 / r2
                        w = q * q
                        wsum += w
                        M += w * wp.outer(d, d)
                        B += w * wp.outer(nj - ni, d)
    Cn = wp.mat33(0.0)
    if wsum > 0.0:
        M += (1.0e-3 * r2 * wsum) * wp.identity(n=3, dtype=float)
        if wp.determinant(M) > 1.0e-30:
            G = B @ wp.inverse(M)
            P = wp.identity(n=3, dtype=float) - wp.outer(ni, ni)
            Cn = P @ (0.5 * (G + wp.transpose(G))) @ P
    C0[i] = scale * Cn


@wp.kernel
def short_map_reset3(valid: wp.array(dtype=wp.int32), Js: wp.array(dtype=wp.mat33), Jinit: wp.array(dtype=wp.mat33)):
    """Jinit <- Jinit Js (the current long map), Js <- I."""
    p = wp.tid()
    if valid[p] == 0:
        return
    Jinit[p] = Jinit[p] @ Js[p]
    Js[p] = wp.identity(n=3, dtype=float)


@wp.kernel
def reset_maps3(n_used: wp.array(dtype=wp.int32), Js: wp.array(dtype=wp.mat33), Jinit: wp.array(dtype=wp.mat33),
                C: wp.array(dtype=wp.mat33)):
    """Long reinitialisation: Js = Jinit = I and C = 0 for the slots in use (free slots are never read; a
    newborn sets its own maps)."""
    p = wp.tid()
    if p >= n_used[0]:
        return
    Js[p] = wp.identity(n=3, dtype=float)
    Jinit[p] = wp.identity(n=3, dtype=float)
    C[p] = wp.mat33(0.0)


# ---------------------------------------------------------------------------
# Per-step particle updates
# ---------------------------------------------------------------------------


@wp.kernel
def update_particle_maps3(
    valid: wp.array(dtype=wp.int32),
    Js: wp.array(dtype=wp.mat33),
    Jinit: wp.array(dtype=wp.mat33),
    C0: wp.array(dtype=wp.mat33),
    g0: wp.array(dtype=wp.vec3),
    C: wp.array(dtype=wp.mat33),
    g: wp.array(dtype=wp.vec3),
):
    """C = Js^T C0 Js and g = normalize(J_long^T g0) with J_long = Jinit Js."""
    p = wp.tid()
    if valid[p] == 0:
        return
    J = Js[p]
    C[p] = wp.transpose(J) @ C0[p] @ J
    g[p] = safe_normalize3(wp.transpose(Jinit[p] @ J) @ g0[p])


# ---------------------------------------------------------------------------
# Adaptive seeding (one particle per under-populated band cell and step): mark_deficient_cells, compaction of
# the flagged cells into a candidate list, seed_partial_sums3, seed_accept3, prefix sum, seed_store3,
# seed_advance3
# ---------------------------------------------------------------------------


@wp.kernel
def mark_deficient_cells(phi_g: wp.array3d(dtype=float), cells: CellListView3D, band: float, seed_min: int,
                         flags: wp.array(dtype=wp.int32)):
    """flags[c] = 1 for band cells (|phi_g| < band) holding fewer than ``seed_min`` particles."""
    i, j, k = wp.tid()
    c = cell_id3(cells, i, j, k)
    if wp.abs(phi_g[i, j, k]) < band and cells.end[c] - cells.start[c] < seed_min:
        flags[c] = 1
    else:
        flags[c] = 0


@wp.func
def seed_candidate3(c: int, istep: int, cells: CellListView3D, dx: float):
    """Cell coordinates of cell ``c`` and the random point of its seeding candidate at step ``istep``."""
    ci, cj, ck = decode_cell(c, cells.ny, cells.nz)
    rng = wp.rand_init(istep, c)
    xp = (wp.vec3(float(ci), float(cj), float(ck)) + wp.vec3(wp.randf(rng), wp.randf(rng), wp.randf(rng))) * dx
    return ci, cj, ck, xp


@wp.kernel
def seed_partial_sums3(
    cand: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    parts: ParticleView3D,
    cells: CellListView3D,
    istep: wp.array(dtype=wp.int32),
    dx: float,
    pw: wp.array(dtype=float),
    pv: wp.array(dtype=wp.vec4),
    pC: wp.array(dtype=wp.mat33),
):
    """Partial neighbour sums of the seeding candidates, one thread per (candidate q, neighbour cell nb):
    ``pw = sum w``, ``pv = (sum w n, sum w phi)`` and ``pC = sum w C`` over the particles of cell nb of the
    3x3x3 block around the candidate's cell, with ``w = N2(d_x/dx) N2(d_y/dx) N2(d_z/dx)`` and ``(phi, n)`` the
    Taylor extrapolation to the candidate point. Splitting the 27 cells over threads keeps the serial loops
    short (one thread per candidate would loop over ~500-1300 particles)."""
    t = wp.tid()
    q = t // 27
    nb = t - 27 * q
    if q >= count[0]:  # the entries past the candidates are never read
        return
    ci, cj, ck, xp = seed_candidate3(cand[q], istep[0], cells, dx)
    ni = ci + nb // 9 - 1
    nj = cj + (nb // 3) % 3 - 1
    nk = ck + nb % 3 - 1
    if ni < 0 or ni >= cells.nx or nj < 0 or nj >= cells.ny or nk < 0 or nk >= cells.nz:
        pw[t] = 0.0
        pv[t] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        pC[t] = wp.mat33(0.0)
        return
    inv = 1.0 / dx
    wsum = float(0.0)
    acc = float(0.0)
    accn = wp.vec3(0.0, 0.0, 0.0)
    accC = wp.mat33(0.0)
    cc = cell_id3(cells, ni, nj, nk)
    for kk in range(cells.start[cc], cells.end[cc]):
        p = cells.ids[kk]
        d = xp - parts.x[p]
        w = N2(d[0] * inv) * N2(d[1] * inv) * N2(d[2] * inv)
        if w > 0.0:
            v, n = extrapolate3(parts.phi[p], parts.g[p], parts.C[p], d)
            wsum += w
            acc += w * v
            accn += w * n
            accC += w * parts.C[p]
    pw[t] = wsum
    pv[t] = wp.vec4(accn[0], accn[1], accn[2], acc)
    pC[t] = accC


@wp.func
def _seed_sums(q: int, pw: wp.array(dtype=float), pv: wp.array(dtype=wp.vec4), pC: wp.array(dtype=wp.mat33)):
    """Sum of the 27 partial sums of candidate ``q`` (in a fixed order: deterministic)."""
    wsum = float(0.0)
    v = wp.vec4(0.0, 0.0, 0.0, 0.0)
    Cs = wp.mat33(0.0)
    for nb in range(27):
        t = 27 * q + nb
        wsum += pw[t]
        v += pv[t]
        Cs += pC[t]
    return wsum, v, Cs


@wp.kernel
def seed_accept3(
    cand: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    cells: CellListView3D,
    phi_grid: wp.array3d(dtype=float),
    istep: wp.array(dtype=wp.int32),
    dx: float,
    band: float,
    coherence: float,
    pw: wp.array(dtype=float),
    pv: wp.array(dtype=wp.vec4),
    pC: wp.array(dtype=wp.mat33),
    accept: wp.array(dtype=wp.int32),
):
    """accept[q] = 1 if candidate q becomes a particle (rules: :func:`seed_store3`)."""
    q = wp.tid()
    accept[q] = 0
    if q >= count[0]:
        return
    ci, cj, ck, xp = seed_candidate3(cand[q], istep[0], cells, dx)
    wsum, v, Cs = _seed_sums(q, pw, pv, pC)
    if wsum <= 0.0:
        return
    val = v[3] / wsum
    if wp.abs(val) >= band or wp.length(wp.vec3(v[0], v[1], v[2])) < coherence * wsum:
        return
    if coherence > 0.0 and sgn(val) != sgn(trilerp(phi_grid, xp / dx - wp.vec3(0.5, 0.5, 0.5))):
        return
    accept[q] = 1


@wp.kernel
def seed_store3(
    cand: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    accept: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    cells: CellListView3D,
    istep: wp.array(dtype=wp.int32),
    dx: float,
    pw: wp.array(dtype=float),
    pv: wp.array(dtype=wp.vec4),
    pC: wp.array(dtype=wp.mat33),
    n_used: wp.array(dtype=wp.int32),
    dropped: wp.array(dtype=wp.int32),
    x: wp.array(dtype=wp.vec3),
    phi: wp.array(dtype=float),
    g: wp.array(dtype=wp.vec3),
    g0: wp.array(dtype=wp.vec3),
    C: wp.array(dtype=wp.mat33),
    C0: wp.array(dtype=wp.mat33),
    Js: wp.array(dtype=wp.mat33),
    Jinit: wp.array(dtype=wp.mat33),
    valid: wp.array(dtype=wp.int32),
):
    """Adaptive seeding: store the accepted candidates in slots ``n_used + offsets[q]`` (``offsets`` = exclusive
    scan of ``accept``), so rejected candidates take no storage and the slots are deterministic.

    One candidate per flagged cell (a band cell holding fewer than ``seed_min`` particles), at a random point
    of the cell. Its value, normal and normal gradient are the B-spline-weighted averages of the Taylor
    extrapolations of the particles in the surrounding 3x3x3 cells (:func:`seed_partial_sums3`); its maps start
    at the identity; it is marked ``valid = 2`` (a newborn). :func:`seed_accept3` rejects the candidate if it
    has no neighbours, if ``|phi| >= band``, if the neighbours' normals disagree (length of their weighted mean
    below ``coherence``: the point lies between the two sides of a thin feature) or if its sign differs from
    the background grid ``phi_grid`` at its position (both checks are off for ``coherence = 0``). Without these
    guards a wrong-sign newborn creates a spurious interface whose cells are under-populated band cells, which
    spawn more particles from it. Particles that do not fit into the
    storage are counted in ``dropped`` (the host grows the storage at its next check).
    """
    q = wp.tid()
    if q >= count[0] or accept[q] == 0:
        return
    s = n_used[0] + offsets[q]
    if s >= x.shape[0]:
        wp.atomic_add(dropped, 0, 1)
        return
    ci, cj, ck, xp = seed_candidate3(cand[q], istep[0], cells, dx)
    wsum, v, Cs = _seed_sums(q, pw, pv, pC)
    nrm = safe_normalize3(wp.vec3(v[0], v[1], v[2]))
    Cn = Cs / wsum
    x[s] = xp
    phi[s] = v[3] / wsum
    g[s] = nrm
    g0[s] = nrm
    C[s] = Cn
    C0[s] = Cn
    Js[s] = wp.identity(n=3, dtype=float)
    Jinit[s] = wp.identity(n=3, dtype=float)
    valid[s] = 2


@wp.kernel
def seed_advance3(count: wp.array(dtype=wp.int32), accept: wp.array(dtype=wp.int32),
                  offsets: wp.array(dtype=wp.int32), n_used: wp.array(dtype=wp.int32), capacity: int,
                  overflow: wp.array(dtype=wp.int32)):
    """n_used += accepted candidates (clamped to the capacity); overflow[0] = max(overflow[0], number of
    candidates that did not fit into the candidate buffers)."""
    cap = accept.shape[0]
    k = wp.min(count[0], cap)
    if k > 0:
        n_used[0] = wp.min(n_used[0] + offsets[k - 1] + accept[k - 1], capacity)
    overflow[0] = wp.max(overflow[0], count[0] - cap)


@wp.kernel
def gather_counters(a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32), c: wp.array(dtype=wp.int32),
                    out: wp.array(dtype=wp.int32)):
    """out = (a[0], b[0], c[0]): several device counters for a single host read."""
    out[0] = a[0]
    out[1] = b[0]
    out[2] = c[0]


# ---------------------------------------------------------------------------
# Slot compaction: the live particles are reordered into cell order, every other slot is freed
# ---------------------------------------------------------------------------


@wp.kernel
def mark_kept_slots(
    ids: wp.array(dtype=wp.int32),
    start: wp.array(dtype=wp.int32),
    num_cells: int,
    x: wp.array(dtype=wp.vec3),
    valid: wp.array(dtype=wp.int32),
    phi_grid: wp.array3d(dtype=float),
    dx: float,
    far: float,
    keep: wp.array(dtype=wp.int32),
):
    """keep[k] = 1 if the k-th entry of the cell-sorted particle list ``ids`` survives the compaction.

    Dropped: the entries past the binned particles (particles outside the domain or with a non-finite
    position are not binned) and the newborns further than ``far`` from the interface of the background grid
    ``phi_grid`` (0 = never): a newborn carries a value extrapolated with a unit gradient, whereas the
    transported level set is stretched and compressed, so the flow brings it back with a stale value (it then
    fills output holes with ghost liquid). Lattice particles are always kept: they carry the evolved level set,
    and where the grid has lost a sheet thinner than a cell they are its only record.
    """
    k = wp.tid()
    keep[k] = 0
    if k >= start[num_cells]:
        return
    p = ids[k]
    if valid[p] == 2 and far > 0.0:
        if wp.abs(trilerp(phi_grid, x[p] / dx - wp.vec3(0.5, 0.5, 0.5))) > far:
            return
    keep[k] = 1


@wp.kernel
def gather_vec3(src: wp.array(dtype=wp.vec3), ids: wp.array(dtype=wp.int32), keep: wp.array(dtype=wp.int32),
                offsets: wp.array(dtype=wp.int32), dst: wp.array(dtype=wp.vec3)):
    """dst[offsets[k]] = src[ids[k]] for the kept entries (``offsets`` = exclusive scan of ``keep``)."""
    k = wp.tid()
    if keep[k] != 0:
        dst[offsets[k]] = src[ids[k]]


@wp.kernel
def gather_float(src: wp.array(dtype=float), ids: wp.array(dtype=wp.int32), keep: wp.array(dtype=wp.int32),
                 offsets: wp.array(dtype=wp.int32), dst: wp.array(dtype=float)):
    """:func:`gather_vec3` for a scalar array."""
    k = wp.tid()
    if keep[k] != 0:
        dst[offsets[k]] = src[ids[k]]


@wp.kernel
def gather_int(src: wp.array(dtype=wp.int32), ids: wp.array(dtype=wp.int32), keep: wp.array(dtype=wp.int32),
               offsets: wp.array(dtype=wp.int32), dst: wp.array(dtype=wp.int32)):
    """:func:`gather_vec3` for an int array."""
    k = wp.tid()
    if keep[k] != 0:
        dst[offsets[k]] = src[ids[k]]


@wp.kernel
def gather_matrix_rows(src: wp.array(dtype=wp.mat33), ids: wp.array(dtype=wp.int32),
                       keep: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32),
                       r0: wp.array(dtype=wp.vec3), r1: wp.array(dtype=wp.vec3), r2: wp.array(dtype=wp.vec3)):
    """:func:`gather_vec3` of a matrix array into three row arrays (saves a matrix-sized scratch array)."""
    k = wp.tid()
    if keep[k] != 0:
        m = src[ids[k]]
        s = offsets[k]
        r0[s] = wp.vec3(m[0, 0], m[0, 1], m[0, 2])
        r1[s] = wp.vec3(m[1, 0], m[1, 1], m[1, 2])
        r2[s] = wp.vec3(m[2, 0], m[2, 1], m[2, 2])


@wp.kernel
def set_matrix_rows(keep: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32), r0: wp.array(dtype=wp.vec3),
                    r1: wp.array(dtype=wp.vec3), r2: wp.array(dtype=wp.vec3), dst: wp.array(dtype=wp.mat33)):
    """dst[s] = the matrix of rows (r0[s], r1[s], r2[s]) for the compacted slots ``s < offsets[n-1] + keep[n-1]``."""
    s = wp.tid()
    n = keep.shape[0]
    if s >= offsets[n - 1] + keep[n - 1]:
        return
    a = r0[s]
    b = r1[s]
    c = r2[s]
    dst[s] = wp.mat33(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2])


@wp.kernel
def finish_compaction(keep: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32),
                      valid: wp.array(dtype=wp.int32), n_used: wp.array(dtype=wp.int32)):
    """Free the slots past the compacted particles and set ``n_used`` to their number."""
    s = wp.tid()
    n = valid.shape[0]
    total = offsets[n - 1] + keep[n - 1]
    if s >= total:
        valid[s] = 0
    if s == 0:
        n_used[0] = total


# ---------------------------------------------------------------------------
# Repair of non-finite state
# ---------------------------------------------------------------------------


@wp.kernel
def scrub_nonfinite_particles3(x: wp.array(dtype=wp.vec3), phi: wp.array(dtype=float), g: wp.array(dtype=wp.vec3),
                               valid: wp.array(dtype=wp.int32), counter: wp.array(dtype=wp.int32), slot: int):
    """Deactivate particles with a non-finite position, value or gradient.

    The slot is also moved far outside the domain with zero data, so the current cell lists (rebuilt at the
    end of the next step) give it zero weight.
    """
    p = wp.tid()
    if valid[p] == 0:
        return
    if not (wp.isfinite(x[p]) and wp.isfinite(phi[p]) and wp.isfinite(g[p])):
        valid[p] = 0
        x[p] = wp.vec3(-1.0e6, -1.0e6, -1.0e6)
        phi[p] = 0.0
        g[p] = wp.vec3(0.0, 0.0, 0.0)
        wp.atomic_add(counter, slot, 1)


@wp.kernel
def scrub_nonfinite_grid3(phi: wp.array3d(dtype=float), far: float, counter: wp.array(dtype=wp.int32), slot: int):
    """Replace non-finite grid values by ``far`` (outside)."""
    i, j, k = wp.tid()
    if not wp.isfinite(phi[i, j, k]):
        phi[i, j, k] = far
        wp.atomic_add(counter, slot, 1)


# ---------------------------------------------------------------------------
# Velocity-dependent kernels (specialised per flow function)
# ---------------------------------------------------------------------------

_FLOW_KERNELS: dict = {}


def flow_kernels3(vel):
    """Return ``(rk4_particles, semi_lagrangian_grid)`` kernels specialised for the 3D flow ``vel``.

    Both read ``t, dt`` from ``clock`` and the flow parameters from ``params[0]`` (device arrays, so
    captured graphs stay valid when they change). ``stage = 1`` evaluates the velocity at the RK stage
    times; ``stage = 0`` freezes it at the step's start time (``frozen_time``, first order in time).
    """
    if vel in _FLOW_KERNELS:
        return _FLOW_KERNELS[vel]

    @wp.kernel(module="unique")
    def rk4_particles3(
        x: wp.array(dtype=wp.vec3),
        Js: wp.array(dtype=wp.mat33),
        valid: wp.array(dtype=wp.int32),
        clock: wp.array(dtype=float),
        params: wp.array(dtype=FlowParams3D),
        ux: wp.array3d(dtype=float),
        uy: wp.array3d(dtype=float),
        uz: wp.array3d(dtype=float),
        inv_dx: float,
        stage: float,
        x_max: wp.vec3,
        Jinit: wp.array(dtype=wp.mat33),
        C0: wp.array(dtype=wp.mat33),
        g0: wp.array(dtype=wp.vec3),
        C: wp.array(dtype=wp.mat33),
        g: wp.array(dtype=wp.vec3),
    ):
        """Classical RK4 for dx/dt = u and dJ/dt = -J grad u at stage times t, t+dt/2, t+dt, followed by the
        map update of :func:`update_particle_maps3`. Positions are clamped to the domain ``[0, x_max]`` (a MAC
        velocity sampled next to a wall need not vanish on it)."""
        p = wp.tid()
        if valid[p] == 0:
            return
        t = clock[0]
        dt = clock[1]
        prm = params[0]
        t_half = t + 0.5 * dt * stage
        t_end = t + dt * stage
        x0 = x[p]
        J0 = Js[p]
        u1, G1 = vel(x0, t, prm, ux, uy, uz, inv_dx)
        k1 = J0 @ G1
        x1 = x0 + 0.5 * dt * u1
        u2, G2 = vel(x1, t_half, prm, ux, uy, uz, inv_dx)
        k2 = (J0 - 0.5 * dt * k1) @ G2
        x2 = x0 + 0.5 * dt * u2
        u3, G3 = vel(x2, t_half, prm, ux, uy, uz, inv_dx)
        k3 = (J0 - 0.5 * dt * k2) @ G3
        x3 = x0 + dt * u3
        u4, G4 = vel(x3, t_end, prm, ux, uy, uz, inv_dx)
        k4 = (J0 - dt * k3) @ G4
        xn = x0 + (dt / 6.0) * (u1 + 2.0 * u2 + 2.0 * u3 + u4)
        x[p] = wp.vec3(wp.clamp(xn[0], 0.0, x_max[0]), wp.clamp(xn[1], 0.0, x_max[1]),
                       wp.clamp(xn[2], 0.0, x_max[2]))
        J = J0 - (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        Js[p] = J
        C[p] = wp.transpose(J) @ C0[p] @ J
        g[p] = safe_normalize3(wp.transpose(Jinit[p] @ J) @ g0[p])

    @wp.kernel(module="unique")
    def semi_lagrangian_grid3(
        phi_in: wp.array3d(dtype=float),
        phi_out: wp.array3d(dtype=float),
        clock: wp.array(dtype=float),
        params: wp.array(dtype=FlowParams3D),
        ux: wp.array3d(dtype=float),
        uy: wp.array3d(dtype=float),
        uz: wp.array3d(dtype=float),
        dx: float,
        stage: float,
    ):
        """Backward RK2 (midpoint) trace from t+dt to t and trilinear sampling."""
        i, j, k = wp.tid()
        t = clock[0]
        dt = clock[1]
        prm = params[0]
        inv_dx = 1.0 / dx
        xc = cell_center3(i, j, k, dx)
        u1, G1 = vel(xc, t + dt * stage, prm, ux, uy, uz, inv_dx)
        xm = xc - 0.5 * dt * u1
        u2, G2 = vel(xm, t + 0.5 * dt * stage, prm, ux, uy, uz, inv_dx)
        x0 = xc - dt * u2
        phi_out[i, j, k] = trilerp(phi_in, x0 * inv_dx - wp.vec3(0.5, 0.5, 0.5))

    _FLOW_KERNELS[vel] = (rk4_particles3, semi_lagrangian_grid3)
    return _FLOW_KERNELS[vel]
