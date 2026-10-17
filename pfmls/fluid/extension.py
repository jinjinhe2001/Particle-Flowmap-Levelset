"""Velocity extrapolation from liquid-adjacent faces into the air.

After the projection a face is *valid* if it touches a liquid cell.  Each Jacobi pass assigns to every invalid
face the average of its valid 4-neighbours on the same component grid and marks it valid; ``n`` passes fill an
``n``-face-wide band.  Faces that are still invalid afterwards are either set to zero (``zero_rest``) or keep
their value (the free-surface solver's ballistic far-field air).  Wall faces next to air are extrapolated like any
other face (they are not zero sources, which would freeze sub-cell liquid next to walls) and then keep only the
component pointing away from the wall (:func:`clamp_wall_faces`).  Both MAC components are processed by one launch
per pass.
"""

from __future__ import annotations

import warp as wp


@wp.func
def _extend_face(fin: wp.array2d(dtype=float), vin: wp.array2d(dtype=wp.int32), fout: wp.array2d(dtype=float),
                 vout: wp.array2d(dtype=wp.int32), i: int, j: int):
    n0 = fin.shape[0]
    n1 = fin.shape[1]
    if vin[i, j] != 0:
        fout[i, j] = fin[i, j]
        vout[i, j] = vin[i, j]
        return
    s = float(0.0)
    c = int(0)
    if i > 0:
        if vin[i - 1, j] != 0:
            s += fin[i - 1, j]
            c += 1
    if i < n0 - 1:
        if vin[i + 1, j] != 0:
            s += fin[i + 1, j]
            c += 1
    if j > 0:
        if vin[i, j - 1] != 0:
            s += fin[i, j - 1]
            c += 1
    if j < n1 - 1:
        if vin[i, j + 1] != 0:
            s += fin[i, j + 1]
            c += 1
    if c > 0:
        fout[i, j] = s / float(c)
        vout[i, j] = 2
    else:
        fout[i, j] = fin[i, j]
        vout[i, j] = 0


@wp.kernel
def extend_pass_kernel(
    fu: wp.array2d(dtype=float), vu: wp.array2d(dtype=wp.int32), fu_out: wp.array2d(dtype=float),
    vu_out: wp.array2d(dtype=wp.int32),
    fv: wp.array2d(dtype=float), vv: wp.array2d(dtype=wp.int32), fv_out: wp.array2d(dtype=float),
    vv_out: wp.array2d(dtype=wp.int32),
):
    """One Jacobi extension pass of both face grids (launch dim (nx+1, ny+1))."""
    i, j = wp.tid()
    if i < fu.shape[0] and j < fu.shape[1]:
        _extend_face(fu, vu, fu_out, vu_out, i, j)
    if i < fv.shape[0] and j < fv.shape[1]:
        _extend_face(fv, vv, fv_out, vv_out, i, j)


@wp.kernel
def zero_invalid_kernel(u: wp.array2d(dtype=float), vu: wp.array2d(dtype=wp.int32), v: wp.array2d(dtype=float),
                        vv: wp.array2d(dtype=wp.int32)):
    i, j = wp.tid()
    if i < u.shape[0] and j < u.shape[1]:
        if vu[i, j] == 0:
            u[i, j] = 0.0
    if i < v.shape[0] and j < v.shape[1]:
        if vv[i, j] == 0:
            v[i, j] = 0.0


@wp.kernel
def clamp_wall_kernel(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float)):
    """Remove the velocity components pointing into the walls (launch dim (nx+1, ny+1))."""
    i, j = wp.tid()
    if j < u.shape[1]:
        if i == 0:
            u[i, j] = wp.max(u[i, j], 0.0)
        elif i == u.shape[0] - 1:
            u[i, j] = wp.min(u[i, j], 0.0)
    if i < v.shape[0]:
        if j == 0:
            v[i, j] = wp.max(v[i, j], 0.0)
        elif j == v.shape[1] - 1:
            v[i, j] = wp.min(v[i, j], 0.0)


def clamp_wall_faces(u: wp.array2d, v: wp.array2d, device=None) -> None:
    """Remove velocity components pointing into the walls of the box."""
    wp.launch(clamp_wall_kernel, dim=(u.shape[0], v.shape[1]), inputs=[u, v], device=device)


class FaceExtrapolator:
    """Ping-pong buffers for repeated extension passes of a MAC field (both components per launch)."""

    def __init__(self, u_shape, v_shape, device=None):
        self.device = wp.get_device(device)
        self.dim = (u_shape[0], v_shape[1])
        z = lambda shp, dt=float: wp.zeros(shp, dtype=dt, device=self.device)  # noqa: E731
        self._fu, self._fv = z(u_shape), z(v_shape)
        self._vu, self._vv = z(u_shape, wp.int32), z(v_shape, wp.int32)
        self._vu2, self._vv2 = z(u_shape, wp.int32), z(v_shape, wp.int32)

    def extend(self, u: wp.array2d, v: wp.array2d, valid_u: wp.array2d, valid_v: wp.array2d, passes: int,
               zero_rest: bool = True) -> None:
        """Extend ``u, v`` in place from faces with ``valid != 0`` by ``passes`` passes (the masks are left
        unchanged)."""
        dev = self.device
        wp.copy(self._vu2, valid_u)
        wp.copy(self._vv2, valid_v)
        a = (u, self._vu2, v, self._vv2)
        b = (self._fu, self._vu, self._fv, self._vv)
        for _ in range(passes):
            wp.launch(extend_pass_kernel, dim=self.dim, inputs=[a[0], a[1], b[0], b[1], a[2], a[3], b[2], b[3]],
                      device=dev)
            a, b = b, a
        if a[0] is not u:
            wp.copy(u, a[0])
            wp.copy(v, a[2])
        if zero_rest:
            wp.launch(zero_invalid_kernel, dim=self.dim, inputs=[u, a[1], v, a[3]], device=dev)
