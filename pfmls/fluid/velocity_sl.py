"""Semi-Lagrangian MAC velocity advection (the ``sl`` option and the fallback of the APIC transport).

Each face is traced back with a midpoint (RK2) step in the bilinear MAC velocity and the component is sampled
at the foot point either bilinearly or with a clamped Catmull-Rom bicubic (no new extrema).
"""

from __future__ import annotations

import warp as wp

from ._kernels import OFF_U, OFF_V, mac_velocity, sample_cubic_clamped, sample_linear


@wp.func
def _backtrace(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), pos: wp.vec2, inv_dx: float, dt: float):
    vel = mac_velocity(u, v, pos, inv_dx)
    mid = pos - 0.5 * dt * vel
    vel2 = mac_velocity(u, v, mid, inv_dx)
    return pos - dt * vel2


@wp.func
def _sl_sample(f: wp.array2d(dtype=float), p0: wp.vec2, inv_dx: float, off: wp.vec2, cubic: int) -> float:
    if cubic != 0:
        return sample_cubic_clamped(f, p0, inv_dx, off)
    return sample_linear(f, p0, inv_dx, off)


@wp.kernel
def sl_advect_kernel(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), dx: float, dt: float, cubic: int,
                     out_u: wp.array2d(dtype=float), out_v: wp.array2d(dtype=float)):
    """Both MAC components (launch dim (nx+1, ny+1))."""
    i, j = wp.tid()
    inv_dx = 1.0 / dx
    if j < u.shape[1]:
        p0 = _backtrace(u, v, wp.vec2(float(i) * dx, (float(j) + 0.5) * dx), inv_dx, dt)
        out_u[i, j] = _sl_sample(u, p0, inv_dx, OFF_U, cubic)
    if i < v.shape[0]:
        p0 = _backtrace(u, v, wp.vec2((float(i) + 0.5) * dx, float(j) * dx), inv_dx, dt)
        out_v[i, j] = _sl_sample(v, p0, inv_dx, OFF_V, cubic)


def sl_advect(u, v, dx: float, dt: float, out_u, out_v, cubic: bool = True, device=None):
    """Semi-Lagrangian advection of the MAC field (u, v) by itself over dt into (out_u, out_v)."""
    wp.launch(sl_advect_kernel, dim=(u.shape[0], v.shape[1]), inputs=[u, v, dx, dt, int(cubic)],
              outputs=[out_u, out_v], device=device)


class SemiLagrangianTransport:
    """Velocity transport by semi-Lagrangian advection on the MAC grid.

    With strong surface tension (capillary-dominated flow) this scheme is not stable on its own: an oscillating
    drop gains speed steadily.  Use it for gravity-dominated flows only.
    """

    def __init__(self, cubic: bool = True):
        self.cubic = cubic

    def initialize(self, solver) -> None:
        self._u = wp.zeros_like(solver.u)
        self._v = wp.zeros_like(solver.v)

    def advect(self, solver, dt: float) -> None:
        sl_advect(solver.u, solver.v, solver.dx, dt, self._u, self._v, self.cubic, solver.device)
        wp.copy(solver.u, self._u)
        wp.copy(solver.v, self._v)

    def reseed_due(self) -> bool:
        return False

    def after_projection(self, solver, reseed: bool) -> None:
        pass

    def end_step(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict, solver) -> None:
        pass
