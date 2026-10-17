"""APIC velocity transport (Jiang et al. 2015) on the MAC grid, the default of the free-surface solver.

Particles carry a velocity ``u_p`` and an affine matrix ``C_p = grad u`` gathered from the grid after every
projection.  Each step they are moved with RK4 in the (frozen) grid velocity and splatted back to the faces
with quadratic B-spline weights, ``u_face = sum w (u_p + C_p (x_face - x_p)) / sum w``.  Faces without particle
weight fall back to semi-Lagrangian advection.  The liquid level set is the authority on where liquid is:
particles that drift more than ``kill_band`` into the air are dropped, and all slots are reseeded on a regular
lattice every ``reseed_every`` steps.
"""

from __future__ import annotations

import warp as wp

from ..core.kernels import mac_velocity_grad_reflect
from ._kernels import mac_velocity_smooth
from .particles import ParticleSlots, clamp_positions_kernel
from .velocity_sl import sl_advect


@wp.func
def rk4_position(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), p: wp.vec2, inv_dx: float, dt: float):
    u1 = mac_velocity_smooth(u, v, p, inv_dx)
    u2 = mac_velocity_smooth(u, v, p + 0.5 * dt * u1, inv_dx)
    u3 = mac_velocity_smooth(u, v, p + 0.5 * dt * u2, inv_dx)
    u4 = mac_velocity_smooth(u, v, p + dt * u3, inv_dx)
    return p + (dt / 6.0) * (u1 + 2.0 * u2 + 2.0 * u3 + u4)


@wp.kernel
def advect_positions_kernel(x: wp.array(dtype=wp.vec2), active: wp.array(dtype=wp.int32),
                            u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), inv_dx: float, dt: float):
    k = wp.tid()
    if active[k] == 0:
        return
    x[k] = rk4_position(u, v, x[k], inv_dx, dt)


@wp.kernel
def g2p_apic_kernel(x: wp.array(dtype=wp.vec2), active: wp.array(dtype=wp.int32),
                    u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), inv_dx: float,
                    vel: wp.array(dtype=wp.vec2), C: wp.array(dtype=wp.mat22)):
    k = wp.tid()
    if active[k] == 0:
        return
    ug, gg = mac_velocity_grad_reflect(u, v, x[k], inv_dx)
    vel[k] = ug
    C[k] = gg


class APICTransport:
    """Affine particle-in-cell velocity transport with semi-Lagrangian fallback."""

    def __init__(self, side: int = 2, reseed_every: int = 8, seed_band_dx: float = 0.5, kill_band_dx: float = 1.0,
                 min_weight: float = 1.0e-6):
        self.side = side
        self.reseed_every = reseed_every
        self.seed_band_dx = seed_band_dx
        self.kill_band_dx = kill_band_dx
        self.min_weight = min_weight
        self.counter = 0

    def initialize(self, solver) -> None:
        dev = solver.device
        self.parts = ParticleSlots(solver.nx, solver.ny, solver.dx, self.side, dev)
        n = self.parts.capacity
        self.vel = wp.zeros(n, dtype=wp.vec2, device=dev)
        self.C = wp.zeros(n, dtype=wp.mat22, device=dev)
        self.fb_u = wp.zeros_like(solver.u)
        self.fb_v = wp.zeros_like(solver.v)
        self._reseed(solver)

    def _reseed(self, solver) -> None:
        self.parts.seed(solver.phi, self.seed_band_dx * solver.dx)
        self._gather(solver)

    def _gather(self, solver) -> None:
        wp.launch(g2p_apic_kernel, dim=self.parts.capacity,
                  inputs=[self.parts.x, self.parts.active, solver.u, solver.v, 1.0 / solver.dx],
                  outputs=[self.vel, self.C], device=solver.device)

    def advect(self, solver, dt: float) -> None:
        """Replace ``solver.u, solver.v`` by the advected velocity."""
        dev, p = solver.device, self.parts
        sl_advect(solver.u, solver.v, solver.dx, dt, self.fb_u, self.fb_v, True, dev)
        wp.launch(advect_positions_kernel, dim=p.capacity, inputs=[p.x, p.active, solver.u, solver.v,
                                                                   1.0 / solver.dx, dt], device=dev)
        wp.launch(clamp_positions_kernel, dim=p.capacity, inputs=[p.x, p.active, solver.lx, solver.ly,
                                                                  1.0e-3 * solver.dx], device=dev)
        p.deactivate_outside(solver.phi, self.kill_band_dx * solver.dx)
        p.p2g(self.vel, self.C, self.fb_u, self.fb_v, solver.u, solver.v, self.min_weight)

    def reseed_due(self) -> bool:
        """Whether the coming step ends with a reseed (host-side schedule)."""
        return self.reseed_every > 0 and (self.counter + 1) % self.reseed_every == 0

    def after_projection(self, solver, reseed: bool) -> None:
        """Gather the projected velocity and its gradient onto the particles (or reseed them)."""
        if reseed:
            self._reseed(solver)
        else:
            self._gather(solver)

    def end_step(self) -> None:
        self.counter += 1

    def particle_positions(self):
        return self.parts.positions_numpy()

    def state_dict(self) -> dict:
        return {"x": self.parts.x.numpy(), "active": self.parts.active.numpy(), "vel": self.vel.numpy(),
                "C": self.C.numpy(), "counter": self.counter}

    def load_state_dict(self, state: dict, solver) -> None:
        self.parts.x.assign(state["x"])
        self.parts.active.assign(state["active"])
        self.vel.assign(state["vel"])
        self.C.assign(state["C"])
        self.counter = int(state["counter"])
