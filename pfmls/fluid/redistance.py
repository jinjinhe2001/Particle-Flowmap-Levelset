"""Closest-point redistancing of a cell-centred level set (the curvature source of the free-surface solver).

Cells in a narrow band get their distance to the closest point of the zero set of the bicubic interpolant (Chopp
2001 iteration, high order and nearly idempotent); the remaining cells are relaxed with parallel (Jacobi) updates
of ``|grad phi| = 1`` (the first-order upwind update of :func:`pfmls.core.eikonal.eikonal_update`).  The sign of
every cell is preserved by construction.  :class:`pfmls.fluid.pfm_backend.PFMLevelSetBackend` takes the
surface-tension curvature from such a redistanced copy of the PFM-LS composite level set.
"""

from __future__ import annotations

import warp as wp

from ..core.eikonal import eikonal_update
from ._kernels import OFF_C, sample_cubic_grad

#: "Far" value (in dx) of cells outside the closest-point band before the eikonal passes reach them.
FAR = 1.0e6


@wp.func
def _sgn(x: float) -> float:
    """-1 for x < 0, +1 otherwise; a NaN counts as air (+1), unlike :func:`pfmls.core.kernels.sgn`."""
    if x < 0.0:
        return -1.0
    return 1.0


@wp.kernel
def redistance_init_kernel(phi: wp.array2d(dtype=float), dx: float, big: float, band: float, iters: int,
                           out: wp.array2d(dtype=float), fixed: wp.array2d(dtype=wp.int32)):
    """Closest-point distance near the interface, +-big elsewhere.

    For cells with a sign change to a 4-neighbour, or with ``|phi| < band``, the closest point on the zero set
    of the bicubic interpolant is found with Chopp's (2001) iteration
    ``x <- x - phi g/|g|^2`` followed by the tangential pull ``(x0 - x) - ((x0 - x).g) g/|g|^2``.
    """
    i, j = wp.tid()
    nx = phi.shape[0]
    ny = phi.shape[1]
    c = phi[i, j]
    s = _sgn(c)
    iface = int(0)
    if _sgn(phi[wp.max(i - 1, 0), j]) != s or _sgn(phi[wp.min(i + 1, nx - 1), j]) != s:
        iface = 1
    if _sgn(phi[i, wp.max(j - 1, 0)]) != s or _sgn(phi[i, wp.min(j + 1, ny - 1)]) != s:
        iface = 1
    if iface == 0 and wp.abs(c) >= band:
        out[i, j] = s * big
        fixed[i, j] = 0
        return
    inv_dx = 1.0 / dx
    x0 = wp.vec2((float(i) + 0.5) * dx, (float(j) + 0.5) * dx)
    x = x0
    ok = int(0)
    for _k in range(iters):
        r = sample_cubic_grad(phi, x, inv_dx, OFF_C)
        g = wp.vec2(r[1], r[2])
        gg = wp.dot(g, g)
        if gg < 1.0e-12:
            break
        xp = x - (r[0] / gg) * g
        q = x0 - xp
        x = xp + q - (wp.dot(q, g) / gg) * g
        dd = x - x0
        if wp.length(dd) > band + 2.0 * dx:
            break
        if wp.abs(r[0]) < 1.0e-4 * dx:
            ok = 1
    r = sample_cubic_grad(phi, x, inv_dx, OFF_C)
    if wp.abs(r[0]) < 1.0e-3 * dx:
        ok = 1
    d = wp.length(x - x0)
    if ok == 1 and wp.isfinite(d) and d < band + 2.0 * dx:
        out[i, j] = s * d
        fixed[i, j] = 1
    elif iface == 1:
        g = wp.vec2(r[1], r[2])
        out[i, j] = s * wp.min(wp.abs(c) / wp.max(wp.length(g), 1.0e-6), dx)
        fixed[i, j] = 1
    else:
        out[i, j] = s * big
        fixed[i, j] = 0


@wp.kernel
def eikonal_jacobi_kernel(phi: wp.array2d(dtype=float), fixed: wp.array2d(dtype=wp.int32), dx: float,
                          out: wp.array2d(dtype=float)):
    i, j = wp.tid()
    c = phi[i, j]
    if fixed[i, j] == 1:
        out[i, j] = c
        return
    nx = phi.shape[0]
    ny = phi.shape[1]
    # neighbours clamped at the walls (a wall cell counts itself)
    a = wp.min(wp.abs(phi[wp.max(i - 1, 0), j]), wp.abs(phi[wp.min(i + 1, nx - 1), j]))
    b = wp.min(wp.abs(phi[i, wp.max(j - 1, 0)]), wp.abs(phi[i, wp.min(j + 1, ny - 1)]))
    out[i, j] = _sgn(c) * wp.min(wp.abs(c), eikonal_update(a, b, dx))


def redistance_closest_point(phi: wp.array2d, dx: float, band: float, newton_iters: int, jacobi_passes: int,
                             work_a: wp.array2d, work_b: wp.array2d, fixed: wp.array2d, device) -> wp.array2d:
    """Closest-point redistancing of ``phi`` into one of the work arrays (returned; ``phi`` is not modified).

    Cells within ``band`` of the interface (or with a sign change to a neighbour) get their distance to the zero set
    of the bicubic interpolant (``newton_iters`` iterations, :func:`redistance_init_kernel`) and are fixed; the
    others start at ``+-FAR dx`` and are relaxed by ``jacobi_passes`` eikonal passes.
    """
    dim = phi.shape
    wp.launch(redistance_init_kernel, dim=dim, inputs=[phi, dx, FAR * dx, band, newton_iters],
              outputs=[work_a, fixed], device=device)
    a, b = work_a, work_b
    for _ in range(jacobi_passes):
        wp.launch(eikonal_jacobi_kernel, dim=dim, inputs=[a, fixed, dx], outputs=[b], device=device)
        a, b = b, a
    return a
