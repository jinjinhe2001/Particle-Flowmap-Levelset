"""Warp kernels for the 2D particle side of PFM-LS.

Particle state (one slot per lattice point, ``valid`` marks the narrow band):

* ``x`` position, ``phi`` level-set value (constant between long reinitialisations),
* ``g`` unit gradient, ``g0`` gradient at the last long reinitialisation (or at birth),
* ``Js`` short flow-map Jacobian ``d psi_s / dx`` (rows), ``Jinit`` long-map Jacobian at the last
  short reinitialisation, so that the long map is ``J_long = Jinit Js``,
* ``C0`` gradient of the unit normal at the last short reinitialisation and ``C = Js^T C0 Js``,
* ``valid``: 0 idle slot, 1 lattice particle (placed at a long reinitialisation), 2 particle created by
  adaptive seeding (a *newborn*, stored in a slot that is idle until the next long reinitialisation).

Particle -> point reconstruction uses the extrapolation of the original method,
``phi_p + normalize(g_p + C_p d) . d`` with ``d = x - x_p``. For a signed distance with
``C = grad n`` this is ``phi_p + g.d + d^T C d``, twice the second-order Taylor term; the
short reinitialisation can scale ``C0`` by ``hessian_scale`` (1 = ``C0 = grad n``, 1/2 =
Taylor-consistent).
"""

from __future__ import annotations

import warp as wp

from ..core.celllist import CellListView, cell_of
from ..core.kernels import N2, bilerp, bspline_sample_grad, dN2, hermite_grad, hermite_value, safe_normalize

#: How the per-step composite and the fine reconstruction (:func:`p2g_point`) treat the newborns of adaptive
#: seeding (``ParticleView.newborns``). The reinitialisation kernels read whatever the cell lists hold: the level set
#: bins only the lattice particles for them unless the composite mixes the newborns in.
NEWBORNS_MIXED = wp.constant(0)     # averaged together with the lattice particles (``valid`` is not read)
NEWBORNS_OFF = wp.constant(1)       # ignored
NEWBORNS_FALLBACK = wp.constant(2)  # used only at points that no lattice particle covers


@wp.struct
class ParticleView:
    """Read-only particle data used by the reconstruction functions.

    ``radius`` is the reconstruction radius (physical units) shared by every particle -> point
    reconstruction; ``newborns`` is one of the ``NEWBORNS_*`` modes and ``valid`` tells newborns (2) from
    lattice particles (1).
    """

    x: wp.array(dtype=wp.vec2)
    phi: wp.array(dtype=float)
    g: wp.array(dtype=wp.vec2)
    C: wp.array(dtype=wp.mat22)
    valid: wp.array(dtype=wp.int32)
    radius: float
    newborns: int


@wp.func
def extrapolate(phi_p: float, g_p: wp.vec2, C_p: wp.mat22, d: wp.vec2):
    """Particle extrapolation ``(phi_p + n . d, n)`` with ``n = normalize(g_p + C_p d)``.

    For a signed distance this equals ``phi_p + g.d + d^T C d`` to second order, i.e. twice the Taylor curvature
    term when ``C = grad n`` (``hessian_scale = 1``); see docs/method.md, section 3."""
    n = safe_normalize(g_p + C_p @ d)
    return phi_p + wp.dot(n, d), n


@wp.func
def p2g_weight(d: wp.vec2, inv_h: float) -> float:
    """Reconstruction weight: 1 (unweighted mean) if ``inv_h == 0``, else ``N2(d_x/h) N2(d_y/h)``."""
    if inv_h == 0.0:
        return 1.0
    return N2(d[0] * inv_h) * N2(d[1] * inv_h)


@wp.func
def lattice_point(p: int, nax: int, spacing: float) -> wp.vec2:
    """Position of lattice slot ``p`` (``nax`` points per row, ``spacing = dx / a``)."""
    return wp.vec2(float(p % nax) + 0.5, float(p // nax) + 0.5) * spacing


# ---------------------------------------------------------------------------
# Initialisation / bookkeeping
# ---------------------------------------------------------------------------


@wp.kernel
def seed_lattice(x: wp.array(dtype=wp.vec2), nax: int, spacing: float):
    p = wp.tid()
    x[p] = lattice_point(p, nax, spacing)


@wp.kernel
def sample_grid_to_particles(phi_grid: wp.array2d(dtype=float), x: wp.array(dtype=wp.vec2), inv_dx: float,
                             phi: wp.array(dtype=float)):
    p = wp.tid()
    phi[p] = bilerp(phi_grid, x[p] * inv_dx - wp.vec2(0.5, 0.5))


@wp.kernel
def set_identity(J: wp.array(dtype=wp.mat22)):
    J[wp.tid()] = wp.identity(n=2, dtype=float)


@wp.kernel
def lattice_only(valid: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    """out = valid without the newborns (1 for lattice particles, 0 otherwise)."""
    p = wp.tid()
    if valid[p] == 1:
        out[p] = 1
    else:
        out[p] = 0


@wp.kernel
def scrub_nonfinite_particles(x: wp.array(dtype=wp.vec2), phi: wp.array(dtype=float), g: wp.array(dtype=wp.vec2),
                              valid: wp.array(dtype=wp.int32), counter: wp.array(dtype=wp.int32), slot: int):
    """Deactivate particles with a non-finite position, value or gradient.

    The slot is also moved far outside the domain with zero data, so the current cell lists (rebuilt
    at the end of the next step) give it zero weight.
    """
    p = wp.tid()
    if valid[p] == 0:
        return
    if not (wp.isfinite(x[p]) and wp.isfinite(phi[p]) and wp.isfinite(g[p])):
        valid[p] = 0
        x[p] = wp.vec2(-1.0e6, -1.0e6)
        phi[p] = 0.0
        g[p] = wp.vec2(0.0, 0.0)
        wp.atomic_add(counter, slot, 1)


@wp.kernel
def scrub_nonfinite_grid(phi: wp.array2d(dtype=float), far: float, counter: wp.array(dtype=wp.int32), slot: int):
    """Replace non-finite grid values by ``far`` (outside)."""
    i, j = wp.tid()
    if not wp.isfinite(phi[i, j]):
        phi[i, j] = far
        wp.atomic_add(counter, slot, 1)


@wp.kernel
def set_step_constants(clock: wp.array(dtype=float), t: float, dt: float, params: wp.array(dtype=wp.vec4),
                       p: wp.vec4, istep: wp.array(dtype=wp.int32), i: int):
    """Per-step device constants read by the (graph-captured) step: time, dt, flow parameters, step index."""
    clock[0] = t
    clock[1] = dt
    params[0] = p
    istep[0] = i


# ---------------------------------------------------------------------------
# Neighbour sums
# ---------------------------------------------------------------------------


@wp.kernel
def shepard_gradient(
    x: wp.array(dtype=wp.vec2),
    phi: wp.array(dtype=float),
    valid: wp.array(dtype=wp.int32),
    cells: CellListView,
    dx: float,
    g: wp.array(dtype=wp.vec2),
):
    """g_i = normalize(grad of the Shepard interpolant sum w phi / sum w), w = N2(r_x) N2(r_y).

    Neighbours: valid particles in the 3x3 cells around particle i (i included).
    """
    i = wp.tid()
    if valid[i] == 0:
        return
    xi = x[i]
    inv = 1.0 / dx
    bi, bj = cell_of(xi, cells.inv_dx)
    s0 = float(0.0)
    sx = float(0.0)
    sy = float(0.0)
    p0 = float(0.0)
    px = float(0.0)
    py = float(0.0)
    for oi in range(-1, 2):
        ci = bi + oi
        if ci < 0 or ci >= cells.nx:
            continue
        for oj in range(-1, 2):
            cj = bj + oj
            if cj < 0 or cj >= cells.ny:
                continue
            c = ci * cells.ny + cj
            for k in range(cells.start[c], cells.end[c]):
                j = cells.ids[k]
                r = (xi - x[j]) * inv
                nxr = N2(r[0])
                nyr = N2(r[1])
                w = nxr * nyr
                wx = dN2(r[0]) * nyr * inv
                wy = nxr * dN2(r[1]) * inv
                pj = phi[j]
                s0 += w
                sx += wx
                sy += wy
                p0 += w * pj
                px += wx * pj
                py += wy * pj
    if s0 > 0.0:
        g[i] = safe_normalize(wp.vec2(px * s0 - p0 * sx, py * s0 - p0 * sy))


@wp.func
def knn_interp(xq: wp.vec2, parts: ParticleView, cells: CellListView, filt: wp.vec2, use_filter: int):
    """Particle interpolation of (phi, unit normal) at ``xq`` from particles within ``r = parts.radius``.

    A particle j is accepted if ``|xq - x_j| < r`` and (optionally) ``filt . n_j > 0``; weights are
    ``N2((x_j - xq)_x / r) N2((x_j - xq)_y / r)`` (``r = p2g_radius dx``, 0.25 dx by default).
    Returns ``(phi, n, ok)``.
    """
    r = parts.radius
    i0 = int(wp.floor((xq[0] - r) * cells.inv_dx))
    i1 = int(wp.floor((xq[0] + r) * cells.inv_dx))
    j0 = int(wp.floor((xq[1] - r) * cells.inv_dx))
    j1 = int(wp.floor((xq[1] + r) * cells.inv_dx))
    i0 = wp.max(i0, 0)
    j0 = wp.max(j0, 0)
    i1 = wp.min(i1, cells.nx - 1)
    j1 = wp.min(j1, cells.ny - 1)
    wsum = float(0.0)
    acc = float(0.0)
    accn = wp.vec2(0.0, 0.0)
    s = 1.0 / r
    for ci in range(i0, i1 + 1):
        for cj in range(j0, j1 + 1):
            c = ci * cells.ny + cj
            for k in range(cells.start[c], cells.end[c]):
                p = cells.ids[k]
                d = xq - parts.x[p]
                if wp.length(d) < r:
                    v, n = extrapolate(parts.phi[p], parts.g[p], parts.C[p], d)
                    if use_filter == 0 or wp.dot(filt, n) > 0.0:
                        w = N2(-d[0] * s) * N2(-d[1] * s)
                        wsum += w
                        acc += w * v
                        accn += w * n
    if wsum > 0.0:
        return acc / wsum, safe_normalize(accn / wsum), 1
    return 0.0, wp.vec2(0.0, 0.0), 0


@wp.kernel
def p2g_cells(
    parts: ParticleView,
    cells: CellListView,
    dx: float,
    w_inv_h: float,
    out_phi: wp.array2d(dtype=float),
    out_g: wp.array2d(dtype=wp.vec2),
    out_m: wp.array2d(dtype=wp.int32),
):
    """Grid reconstruction at cell centres from particles within ``parts.radius`` (see :func:`p2g_weight`).

    Writes ``phi``, the averaged unit normal and the coverage mark (1 if any particle contributed,
    0 otherwise, in which case ``phi = 0`` and ``g = 0``). A particle within ``radius < dx/2`` of a
    cell centre lies in that cell (the radius is validated on construction), so only the cell's own
    list is scanned.
    """
    i, j = wp.tid()
    xc = wp.vec2(float(i) + 0.5, float(j) + 0.5) * dx
    c = i * cells.ny + j
    wsum = float(0.0)
    acc = float(0.0)
    accn = wp.vec2(0.0, 0.0)
    for k in range(cells.start[c], cells.end[c]):
        p = cells.ids[k]
        d = xc - parts.x[p]
        if wp.length(d) <= parts.radius:
            v, n = extrapolate(parts.phi[p], parts.g[p], parts.C[p], d)
            w = p2g_weight(d, w_inv_h)
            wsum += w
            acc += w * v
            accn += w * n
    if wsum > 0.0:
        out_phi[i, j] = acc / wsum
        out_g[i, j] = safe_normalize(accn / wsum)
        out_m[i, j] = 1
    else:
        out_phi[i, j] = 0.0
        out_g[i, j] = wp.vec2(0.0, 0.0)
        out_m[i, j] = 0


@wp.func
def p2g_point(xc: wp.vec2, parts: ParticleView, cells: CellListView, w_inv_h: float):
    """Weighted Taylor sum at ``xc`` over particles within ``parts.radius``; returns ``(sum w phi, sum w)``.

    Newborns (``valid = 2``) enter according to ``parts.newborns``: together with the lattice particles
    (``NEWBORNS_MIXED``), not at all (``NEWBORNS_OFF``) or only if no lattice particle lies within the radius
    (``NEWBORNS_FALLBACK``)."""
    radius = parts.radius
    i0 = wp.max(int(wp.floor((xc[0] - radius) * cells.inv_dx)), 0)
    i1 = wp.min(int(wp.floor((xc[0] + radius) * cells.inv_dx)), cells.nx - 1)
    j0 = wp.max(int(wp.floor((xc[1] - radius) * cells.inv_dx)), 0)
    j1 = wp.min(int(wp.floor((xc[1] + radius) * cells.inv_dx)), cells.ny - 1)
    wsum = float(0.0)
    acc = float(0.0)
    wsum_nb = float(0.0)
    acc_nb = float(0.0)
    for ci in range(i0, i1 + 1):
        for cj in range(j0, j1 + 1):
            c = ci * cells.ny + cj
            for k in range(cells.start[c], cells.end[c]):
                p = cells.ids[k]
                d = xc - parts.x[p]
                if wp.length(d) <= radius:
                    newborn = False
                    if parts.newborns != NEWBORNS_MIXED:
                        newborn = parts.valid[p] == 2
                    if newborn and parts.newborns == NEWBORNS_OFF:
                        continue
                    v, n = extrapolate(parts.phi[p], parts.g[p], parts.C[p], d)
                    w = p2g_weight(d, w_inv_h)
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
def composite_grid(
    parts: ParticleView,
    cells: CellListView,
    dx: float,
    w_inv_h: float,
    phi_grid: wp.array2d(dtype=float),
    far: float,
    phi_out: wp.array2d(dtype=float),
):
    """phi_out = particle reconstruction where a particle lies within ``parts.radius``, else phi_grid.

    A non-finite result (only possible after a float overflow upstream) is replaced by the grid
    value, or by ``far`` (outside) if that is not finite either, so it cannot persist in the grid.
    """
    i, j = wp.tid()
    xc = wp.vec2(float(i) + 0.5, float(j) + 0.5) * dx
    acc, wsum = p2g_point(xc, parts, cells, w_inv_h)
    v = phi_grid[i, j]
    if wsum > 0.0:
        vp = acc / wsum
        if wp.isfinite(vp):
            v = vp
    if not wp.isfinite(v):
        v = far
    phi_out[i, j] = v


@wp.kernel
def reconstruct_fine(
    parts: ParticleView,
    cells: CellListView,
    h_fine: float,
    inv_dx: float,
    w_inv_h: float,
    phi_grid: wp.array2d(dtype=float),
    phi_fine: wp.array2d(dtype=float),
):
    """Fine-grid field: particle reconstruction where covered, bilinear grid sample elsewhere."""
    i, j = wp.tid()
    xc = wp.vec2(float(i) + 0.5, float(j) + 0.5) * h_fine
    acc, wsum = p2g_point(xc, parts, cells, w_inv_h)
    if wsum > 0.0:
        phi_fine[i, j] = acc / wsum
    else:
        phi_fine[i, j] = bilerp(phi_grid, xc * inv_dx - wp.vec2(0.5, 0.5))


# ---------------------------------------------------------------------------
# Long reinitialisation: reseeding with Hermite / particle blending
# ---------------------------------------------------------------------------


@wp.kernel
def reseed(
    old: ParticleView,
    cells: CellListView,
    phi_r: wp.array2d(dtype=float),
    g_r: wp.array2d(dtype=wp.vec2),
    nax: int,
    dx: float,
    a: int,
    band: float,
    e1: float,
    e2: float,
    gate: float,
    grid_gate: float,
    x: wp.array(dtype=wp.vec2),
    phi: wp.array(dtype=float),
    g: wp.array(dtype=wp.vec2),
    g0: wp.array(dtype=wp.vec2),
    valid: wp.array(dtype=wp.int32),
    Js: wp.array(dtype=wp.mat22),
    Jinit: wp.array(dtype=wp.mat22),
):
    """Re-place every slot on its lattice point, assign (phi, g) and reset both maps to the identity.

    Grid value: Hermite interpolation of the redistanced grid (phi_r, g_r). Particle value: KNN
    interpolation of the old particles filtered by the Hermite normal. The particle value is used
    when it is valid, ``|phi_P| < gate dx`` and ``|phi_H| <= grid_gate dx``, with
    ``gamma = n_H . n_P``: gamma > e1 -> particle, e2 < gamma <= e1 -> linear blend, else grid.
    The slot is valid iff ``|phi| < band``.
    """
    p = wp.tid()
    xp = lattice_point(p, nax, dx / float(a))
    q = xp / dx - wp.vec2(0.5, 0.5)
    g_h = hermite_grad(phi_r, g_r, q, dx)
    phi_h = hermite_value(phi_r, g_r, q, dx)
    phi_p, g_p, ok = knn_interp(xp, old, cells, g_h, 1)
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
    nrm = safe_normalize(nrm)
    x[p] = xp
    phi[p] = val
    g[p] = nrm
    g0[p] = nrm
    Js[p] = wp.identity(n=2, dtype=float)
    Jinit[p] = wp.identity(n=2, dtype=float)
    if wp.abs(val) < band:
        valid[p] = 1
    else:
        valid[p] = 0


# ---------------------------------------------------------------------------
# Short reinitialisation: normal-gradient (C0) from MAC-averaged normals
# ---------------------------------------------------------------------------


@wp.kernel
def split_to_faces(g: wp.array2d(dtype=wp.vec2), gx: wp.array2d(dtype=float), gy: wp.array2d(dtype=float)):
    """Average cell normals to faces: gx[i,j] = (g[i-1,j].x + g[i,j].x)/2 (clamped), same for y."""
    i, j = wp.tid()
    nx = g.shape[0]
    ny = g.shape[1]
    if j < ny:
        gx[i, j] = 0.5 * (g[wp.min(i, nx - 1), j][0] + g[wp.clamp(i - 1, 0, nx - 1), j][0])
    if i < nx:
        gy[i, j] = 0.5 * (g[i, wp.min(j, ny - 1)][1] + g[i, wp.clamp(j - 1, 0, ny - 1)][1])


@wp.kernel
def normal_gradient_from_faces(
    x: wp.array(dtype=wp.vec2),
    valid: wp.array(dtype=wp.int32),
    gx: wp.array2d(dtype=float),
    gy: wp.array2d(dtype=float),
    inv_dx: float,
    scale: float,
    C0: wp.array(dtype=wp.mat22),
):
    """C0 = scale * d n / dx from the quadratic B-spline of the face-averaged normals (rows: n_x, n_y)."""
    p = wp.tid()
    if valid[p] == 0:
        return
    vx, dgx = bspline_sample_grad(gx, x[p], inv_dx, wp.vec2(0.0, 0.5))
    vy, dgy = bspline_sample_grad(gy, x[p], inv_dx, wp.vec2(0.5, 0.0))
    C0[p] = scale * wp.mat22(dgx[0], dgx[1], dgy[0], dgy[1])


@wp.kernel
def keep_newborn_curvature(valid: wp.array(dtype=wp.int32), refit: wp.array(dtype=wp.int32),
                           C: wp.array(dtype=wp.mat22), C0: wp.array(dtype=wp.mat22)):
    """C0 = C for the newborns that a short reinitialisation does not refit (``refit[p] == 0``), so that after the
    short-map reset they keep their transported ``C`` instead of the ``C0`` of their birth."""
    p = wp.tid()
    if valid[p] == 2 and refit[p] == 0:
        C0[p] = C[p]


@wp.kernel
def short_map_reset(valid: wp.array(dtype=wp.int32), Js: wp.array(dtype=wp.mat22), Jinit: wp.array(dtype=wp.mat22)):
    """Jinit <- Jinit Js (the current long map), Js <- I (idle slots keep Js = I and are skipped)."""
    p = wp.tid()
    if valid[p] == 0:
        return
    Jinit[p] = Jinit[p] @ Js[p]
    Js[p] = wp.identity(n=2, dtype=float)


# ---------------------------------------------------------------------------
# Per-step particle updates
# ---------------------------------------------------------------------------


@wp.func
def transport_maps(p: int, J: wp.mat22, Jinit: wp.array(dtype=wp.mat22), C0: wp.array(dtype=wp.mat22),
                   g0: wp.array(dtype=wp.vec2), C: wp.array(dtype=wp.mat22), g: wp.array(dtype=wp.vec2)):
    """C = Js^T C0 Js and g = normalize(J_long^T g0) with J_long = Jinit Js, for slot ``p``."""
    C[p] = wp.transpose(J) @ C0[p] @ J
    g[p] = safe_normalize(wp.transpose(Jinit[p] @ J) @ g0[p])


@wp.kernel
def update_particle_maps(
    valid: wp.array(dtype=wp.int32),
    Js: wp.array(dtype=wp.mat22),
    Jinit: wp.array(dtype=wp.mat22),
    C0: wp.array(dtype=wp.mat22),
    g0: wp.array(dtype=wp.vec2),
    C: wp.array(dtype=wp.mat22),
    g: wp.array(dtype=wp.vec2),
):
    """:func:`transport_maps` for a step without advection (the RK4 kernels apply it themselves)."""
    p = wp.tid()
    if valid[p] == 0:
        return
    transport_maps(p, Js[p], Jinit, C0, g0, C, g)


# ---------------------------------------------------------------------------
# Adaptive seeding (one particle per under-populated band cell and step): seed_evaluate, prefix sum of the
# accepted candidates, seed_store, seed_advance. Newborns take idle slots from a list of free slots
# (mark_idle, build_free_list), rebuilt at every long reinitialisation.
# ---------------------------------------------------------------------------


@wp.func
def seed_point(i: int, j: int, c: int, istep: int, dx: float) -> wp.vec2:
    """Random point of cell ``(i, j)`` (flat id ``c``): the seeding candidate of the cell at step ``istep``."""
    rng = wp.rand_init(istep, c)
    return (wp.vec2(float(i), float(j)) + wp.vec2(wp.randf(rng), wp.randf(rng))) * dx


@wp.kernel
def seed_evaluate(
    parts: ParticleView,
    cells: CellListView,
    Jinit: wp.array(dtype=wp.mat22),
    Js: wp.array(dtype=wp.mat22),
    g0: wp.array(dtype=wp.vec2),
    phi_grid: wp.array2d(dtype=float),
    istep: wp.array(dtype=wp.int32),
    dx: float,
    band: float,
    seed_min: int,
    coherence: float,
    accept: wp.array(dtype=wp.int32),
    cand_x: wp.array(dtype=wp.vec2),
    cand_phi: wp.array(dtype=float),
    cand_g: wp.array(dtype=wp.vec2),
    cand_C: wp.array(dtype=wp.mat22),
):
    """accept[c] = 1 if the seeding candidate of cell c becomes a particle; its data go to ``cand_*[c]``.

    A band cell (``|phi_grid| < band``) holding fewer than ``seed_min`` particles gets one candidate at a
    random point. Its value is the B-spline-weighted mean (``w = N2(d_x/dx) N2(d_y/dx)``) of the first-order
    Taylor predictions ``v_p = phi_p + G_p . d`` of the lattice particles in the surrounding 3x3 cells, with
    ``G_p = (Jinit Js)^T g0`` the transported, unnormalised gradient; its gradient and normal gradient are the
    weighted means of ``G_p`` and ``C_p``. The transported gradient keeps the prediction consistent with the
    transported level set where the band is compressed (a unit normal would underestimate the change of
    ``phi`` across a thin filament, so the sign inside it would be lost). Rejected: no neighbours; a mean
    first-order distance ``v_p / |G_p|`` of at least ``band``; or predictions that disagree in sign,
    ``|sum w v_p| < coherence sum w |v_p|`` (the point lies at an interface the neighbours place differently;
    no check for ``coherence = 0``); or a sign that no neighbour carries (an extrapolation across an interface
    no particle resolves, e.g. beyond the tip of a filament). Newborns are not used as neighbours, so seeding
    never builds on seeding.
    """
    i, j = wp.tid()
    c = i * cells.ny + j
    accept[c] = 0
    if cells.end[c] - cells.start[c] >= seed_min or wp.abs(phi_grid[i, j]) >= band:
        return
    xp = seed_point(i, j, c, istep[0], dx)
    inv = 1.0 / dx
    wsum = float(0.0)
    acc = float(0.0)
    acc_abs = float(0.0)
    acc_dist = float(0.0)
    accG = wp.vec2(0.0, 0.0)
    accC = wp.mat22(0.0)
    inside = int(0)  # neighbours with phi_p < 0 / >= 0
    outside = int(0)
    for oi in range(-1, 2):
        ci = i + oi
        if ci < 0 or ci >= cells.nx:
            continue
        for oj in range(-1, 2):
            cj = j + oj
            if cj < 0 or cj >= cells.ny:
                continue
            cc = ci * cells.ny + cj
            for k in range(cells.start[cc], cells.end[cc]):
                p = cells.ids[k]
                if parts.valid[p] == 2:
                    continue
                d = xp - parts.x[p]
                w = N2(d[0] * inv) * N2(d[1] * inv)
                if w > 0.0:
                    G = wp.transpose(Jinit[p] @ Js[p]) @ g0[p]
                    v = parts.phi[p] + wp.dot(G, d)
                    lg = wp.length(G)
                    wsum += w
                    acc += w * v
                    acc_abs += w * wp.abs(v)
                    if lg > 0.0:
                        acc_dist += w * v / lg
                    else:
                        acc_dist += w * v
                    accG += w * G
                    accC += w * parts.C[p]
                    if parts.phi[p] < 0.0:
                        inside += 1
                    else:
                        outside += 1
    if wsum <= 0.0:
        return
    if wp.abs(acc_dist / wsum) >= band or wp.abs(acc) < coherence * acc_abs:
        return
    if (acc < 0.0 and inside == 0) or (acc >= 0.0 and outside == 0):
        return
    accept[c] = 1
    cand_x[c] = xp
    cand_phi[c] = acc / wsum
    cand_g[c] = accG / wsum
    cand_C[c] = accC / wsum


@wp.kernel
def seed_store(
    accept: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    cand_x: wp.array(dtype=wp.vec2),
    cand_phi: wp.array(dtype=float),
    cand_g: wp.array(dtype=wp.vec2),
    cand_C: wp.array(dtype=wp.mat22),
    free: wp.array(dtype=wp.int32),
    free_count: wp.array(dtype=wp.int32),
    n_new: wp.array(dtype=wp.int32),
    dropped: wp.array(dtype=wp.int32),
    x: wp.array(dtype=wp.vec2),
    phi: wp.array(dtype=float),
    g: wp.array(dtype=wp.vec2),
    g0: wp.array(dtype=wp.vec2),
    C: wp.array(dtype=wp.mat22),
    C0: wp.array(dtype=wp.mat22),
    Js: wp.array(dtype=wp.mat22),
    Jinit: wp.array(dtype=wp.mat22),
    valid: wp.array(dtype=wp.int32),
):
    """Store the accepted candidate of cell c in the free slot ``free[n_new + offsets[c]]`` (``offsets`` = exclusive
    scan of ``accept``, so the slots are deterministic) as a newborn (``valid = 2``): maps at the identity, ``g0``
    the interpolated gradient (``g`` its direction), ``C0 = C`` the interpolated normal gradient. From then on it
    is advected like the lattice particles. Candidates beyond the free slots are counted in ``dropped``."""
    c = wp.tid()
    if accept[c] == 0:
        return
    k = n_new[0] + offsets[c]
    if k >= free_count[0]:
        wp.atomic_add(dropped, 0, 1)
        return
    s = free[k]
    G = cand_g[c]
    x[s] = cand_x[c]
    phi[s] = cand_phi[c]
    g[s] = safe_normalize(G)
    g0[s] = G
    C[s] = cand_C[c]
    C0[s] = cand_C[c]
    Js[s] = wp.identity(n=2, dtype=float)
    Jinit[s] = wp.identity(n=2, dtype=float)
    valid[s] = 2


@wp.kernel
def seed_advance(accept: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32), n_new: wp.array(dtype=wp.int32),
                 free_count: wp.array(dtype=wp.int32)):
    """n_new += the accepted candidates (at most the number of free slots)."""
    n = accept.shape[0]
    n_new[0] = wp.min(n_new[0] + offsets[n - 1] + accept[n - 1], free_count[0])


@wp.kernel
def mark_idle(valid: wp.array(dtype=wp.int32), flags: wp.array(dtype=wp.int32)):
    """flags[s] = 1 for the idle slots (``valid = 0``)."""
    s = wp.tid()
    if valid[s] == 0:
        flags[s] = 1
    else:
        flags[s] = 0


@wp.kernel
def build_free_list(flags: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32), free: wp.array(dtype=wp.int32),
                    free_count: wp.array(dtype=wp.int32), n_new: wp.array(dtype=wp.int32)):
    """free[0 .. free_count - 1] = the idle slots in increasing order (``offsets`` = exclusive scan of ``flags``);
    none of them is used yet (``n_new = 0``)."""
    s = wp.tid()
    if flags[s] != 0:
        free[offsets[s]] = s
    n = flags.shape[0]
    if s == n - 1:
        free_count[0] = offsets[s] + flags[s]
        n_new[0] = 0


# ---------------------------------------------------------------------------
# Velocity-dependent kernels (specialised per flow function)
# ---------------------------------------------------------------------------

_FLOW_KERNELS: dict = {}


def flow_kernels(vel):
    """Return ``(rk4_particles, semi_lagrangian_grid)`` kernels specialised for the flow ``vel``.

    Both read ``t, dt`` from ``clock`` and the flow parameters from ``params[0]`` (device arrays,
    so captured graphs stay valid when they change). ``stage = 1`` evaluates the velocity at the RK
    stage times; ``stage = 0`` freezes it at the step's start time ``t`` (``frozen_time``, first
    order in time). The particle kernel also updates ``C`` and ``g`` from the new short
    map (:func:`transport_maps`).
    """
    if vel in _FLOW_KERNELS:
        return _FLOW_KERNELS[vel]

    @wp.kernel(module="unique")
    def rk4_particles(
        x: wp.array(dtype=wp.vec2),
        Js: wp.array(dtype=wp.mat22),
        valid: wp.array(dtype=wp.int32),
        clock: wp.array(dtype=float),
        params: wp.array(dtype=wp.vec4),
        ux: wp.array2d(dtype=float),
        uy: wp.array2d(dtype=float),
        inv_dx: float,
        stage: float,
        Jinit: wp.array(dtype=wp.mat22),
        C0: wp.array(dtype=wp.mat22),
        g0: wp.array(dtype=wp.vec2),
        C: wp.array(dtype=wp.mat22),
        g: wp.array(dtype=wp.vec2),
    ):
        """Classical RK4 for dx/dt = u and dJ/dt = -J grad u at stage times t, t+dt/2, t+dt, then
        the map transport of ``C`` and ``g``."""
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
        u1, G1 = vel(x0, t, prm, ux, uy, inv_dx)
        k1 = J0 @ G1
        x1 = x0 + 0.5 * dt * u1
        u2, G2 = vel(x1, t_half, prm, ux, uy, inv_dx)
        k2 = (J0 - 0.5 * dt * k1) @ G2
        x2 = x0 + 0.5 * dt * u2
        u3, G3 = vel(x2, t_half, prm, ux, uy, inv_dx)
        k3 = (J0 - 0.5 * dt * k2) @ G3
        x3 = x0 + dt * u3
        u4, G4 = vel(x3, t_end, prm, ux, uy, inv_dx)
        k4 = (J0 - dt * k3) @ G4
        x[p] = x0 + (dt / 6.0) * (u1 + 2.0 * u2 + 2.0 * u3 + u4)
        J = J0 - (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        Js[p] = J
        transport_maps(p, J, Jinit, C0, g0, C, g)

    @wp.kernel(module="unique")
    def semi_lagrangian_grid(
        phi_in: wp.array2d(dtype=float),
        phi_out: wp.array2d(dtype=float),
        clock: wp.array(dtype=float),
        params: wp.array(dtype=wp.vec4),
        ux: wp.array2d(dtype=float),
        uy: wp.array2d(dtype=float),
        dx: float,
        stage: float,
    ):
        """Backward RK2 (midpoint) trace from t+dt to t and bilinear sampling."""
        i, j = wp.tid()
        t = clock[0]
        dt = clock[1]
        prm = params[0]
        inv_dx = 1.0 / dx
        xc = wp.vec2(float(i) + 0.5, float(j) + 0.5) * dx
        u1, G1 = vel(xc, t + dt * stage, prm, ux, uy, inv_dx)
        xm = xc - 0.5 * dt * u1
        u2, G2 = vel(xm, t + 0.5 * dt * stage, prm, ux, uy, inv_dx)
        x0 = xc - dt * u2
        phi_out[i, j] = bilerp(phi_in, x0 * inv_dx - wp.vec2(0.5, 0.5))

    _FLOW_KERNELS[vel] = (rk4_particles, semi_lagrangian_grid)
    return _FLOW_KERNELS[vel]
