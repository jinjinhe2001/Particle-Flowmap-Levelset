"""Incompressible free-surface flow on a 2D MAC grid coupled to the PFM-LS level set.

One step of ``FreeSurfaceSolver2D`` (velocity ``u^n`` is projected and extrapolated into the air):

1. NaN scrub of the velocity (and of phi inside the level-set backend);
2. level set: ``phi^{n+1}`` = advance(``phi^n``, ``u^n``, dt) with PFM-LS
   (:class:`~pfmls.fluid.pfm_backend.PFMLevelSetBackend`);
3. velocity transport (``apic`` | ``sl``): ``u* = A(u^n)``;
4. gravity: ``u* += g dt`` (wall faces keep their predicted value);
5. ghost-fluid projection with ``p = sigma kappa`` on ``phi^{n+1} = 0`` (liquid cells only), free-slip walls,
   unilateral where ``pressure.wall_separation`` allows it (see :mod:`pfmls.fluid.pressure`);
6. extrapolation of the projected velocity into the air (``extension_passes`` faces); air beyond that band keeps
   its advected velocity plus gravity (``ballistic_air``), so liquid fragments thinner than a cell, which
   the pressure solve does not see, fly and fall instead of freezing in place;
7. transport bookkeeping (APIC: gather of the projected velocity onto the particles, or a reseed);
8. diagnostics into device buffers (liquid area, smoothed-Heaviside volume, liquid cells, kinetic energy, CG
   iterations and residual, max liquid-face speed), copied to the host in batches.

Time step.  ``dt = min(dt, capillary_cfl * dt_cap, cfl * dx / |u|_max)``: the capillary limit ``dt_cap =
sqrt(rho dx^3 / (2 pi sigma))`` (Brackbill et al. 1992) keeps refined scenes with surface tension stable, and the
advective limit catches the short velocity spikes of capillary pinch-off and coalescence.  ``|u|_max`` is the
largest liquid-face speed of the step before last, copied to pinned host memory asynchronously and read behind an
event, so the host runs one step ahead of the device.  When the advective limit is active the step is rounded down
to ``dt_nominal * 2^(-k / cfl_levels)`` (``dt_nominal = min(dt, capillary_cfl * dt_cap)``), a small set of values.

CUDA graphs (``use_graph``).  A step is three asynchronous launches: the device work before the level-set step
(scrub, ghost layer) and everything after it (transport, projection with the CG loop as a conditional node,
extrapolation, bookkeeping, diagnostics) are captured into CUDA graphs, one per step type (time-step level,
reseed or not), and replayed; the PFM-LS step in between replays its own graphs and keeps its host-side schedule.
The only host reads inside the loop are the lagged speed above and, on long-reinitialisation steps (every 12),
the level-set volume guard.  Steps of another length (the last step shortened to land on an end time) run
eagerly.

Units are arbitrary but consistent; the defaults follow the paper (rho = 1, g = 1, unit-height box).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import warp as wp

from ..core.graphs import GraphCache
from ..core.grid import Grid2D
from ._kernels import dual_cell_liquid_area
from .extension import FaceExtrapolator, clamp_wall_faces
from .pfm_backend import PFMBackendConfig, PFMLevelSetBackend
from .pressure import BLOCK, PressureConfig, PressureSolver
from .velocity_apic import APICTransport
from .velocity_sl import SemiLagrangianTransport

__all__ = ["FluidConfig", "FreeSurfaceSolver2D", "failure_reason", "make_transport"]

VELOCITIES = ("apic", "sl")


@dataclass(frozen=True)
class FluidConfig:
    """Solver parameters (immutable; use :func:`dataclasses.replace` to derive variants).

    The domain is ``[0, nx dx] x [0, ny dx]`` with ``dx = height / ny``.  Invalid values raise ``ValueError``.
    """

    #: cells along x and y
    nx: int = 200
    ny: int = 200
    #: domain height (sets dx = height / ny)
    height: float = 1.0
    #: liquid density
    rho: float = 1.0
    #: surface tension coefficient
    sigma: float = 0.0
    #: body force per unit mass
    gravity: tuple[float, float] = (0.0, -1.0)
    #: nominal time step (an upper bound; see cfl and capillary_cfl)
    dt: float = 5.0e-4
    #: > 0: dt <= cfl dx / max|u| (max|u| on liquid faces, lagged by one step); 0 = off
    cfl: float = 0.5
    #: > 0: dt <= capillary_cfl sqrt(rho dx^3 / (2 pi sigma)); 0 = off
    capillary_cfl: float = 0.5
    #: CFL-limited steps are rounded down to dt_nominal 2^(-k / cfl_levels), so their CUDA graphs are reused
    cfl_levels: int = 8
    #: velocity transport: "apic" (affine particle-in-cell) or "sl" (semi-Lagrangian)
    velocity: str = "apic"
    #: Jacobi passes extrapolating the projected velocity into the air
    extension_passes: int = 10
    #: air faces beyond the extension band keep their advected velocity plus gravity (sub-cell liquid
    #: fragments keep moving); False sets them to zero
    ballistic_air: bool = True
    #: APIC particles per cell = particles_per_side^2
    particles_per_side: int = 2
    #: APIC particle reseeding interval in steps
    reseed_every: int = 8
    #: pressure projection
    pressure: PressureConfig = field(default_factory=PressureConfig)
    #: PFM-LS interface
    pfm_levelset: PFMBackendConfig = field(default_factory=PFMBackendConfig)
    #: CUDA: replay captured graphs of the step
    use_graph: bool = True

    def __post_init__(self):
        if self.nx < 1 or self.ny < 1:
            raise ValueError(f"nx and ny must be >= 1, got {self.nx} x {self.ny}")
        if not (self.height > 0.0 and self.rho > 0.0):
            raise ValueError("height and rho must be > 0")
        if not (math.isfinite(self.dt) and self.dt > 0.0):
            raise ValueError(f"dt must be a finite number > 0, got {self.dt}")
        for name in ("sigma", "cfl", "capillary_cfl"):
            if not getattr(self, name) >= 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        for name in ("cfl_levels", "particles_per_side", "reseed_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.extension_passes < 0:
            raise ValueError(f"extension_passes must be >= 0, got {self.extension_passes}")
        if len(self.gravity) != 2:
            raise ValueError(f"gravity must have two components, got {self.gravity}")
        if self.velocity not in VELOCITIES:
            raise ValueError(f"velocity must be one of {VELOCITIES}, got {self.velocity!r}")
        if not isinstance(self.pfm_levelset, PFMBackendConfig):
            raise TypeError(f"pfm_levelset must be a PFMBackendConfig, got {type(self.pfm_levelset).__name__}")

    @property
    def dx(self) -> float:
        return self.height / self.ny


#: Configuration fields a checkpoint must share with the solver that loads it.
CHECKPOINT_FIELDS = ("nx", "ny", "height", "velocity", "particles_per_side", "pfm_levelset")


def make_transport(cfg: FluidConfig):
    """The velocity transport selected by ``cfg.velocity``.

    A transport replaces the solver's face velocity by the advected one before the projection (``advect``) and
    records the projected velocity afterwards (``after_projection``); its host-side schedule (``reseed_due``,
    ``end_step``) is separate from the device work, so that the solver can replay the latter from CUDA graphs.
    """
    if cfg.velocity == "apic":
        return APICTransport(side=cfg.particles_per_side, reseed_every=cfg.reseed_every)
    return SemiLagrangianTransport()


def failure_reason(health: dict) -> str | None:
    """First hard failure in a :meth:`FreeSurfaceSolver2D.health` record (None if healthy): a repaired non-finite
    velocity or level set, or a pressure solve that hit its iteration cap."""
    if health["velocity_nan_scrubs"] > 0:
        return f"non-finite velocity repaired on {health['velocity_nan_scrubs']} faces"
    if health["nan_scrubs"] > 0 or health["nan_particles"] > 0:
        return f"non-finite level set repaired ({health['nan_scrubs']} cells, {health['nan_particles']} particles)"
    if health["cg_cap_hits"] > 0:
        return f"pressure solve hit its iteration cap in {health['cg_cap_hits']} steps"
    return None


# --------------------------------------------------------------------------------------------- kernels


@wp.kernel
def body_force_kernel(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), fdt: wp.vec2):
    """Body force on every face (wall faces keep the predicted velocity; the projection closes or opens them)."""
    i, j = wp.tid()
    if j < u.shape[1]:
        u[i, j] = u[i, j] + fdt[0]
    if i < v.shape[0]:
        v[i, j] = v[i, j] + fdt[1]


@wp.kernel
def scrub_velocity_kernel(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), counter: wp.array(dtype=wp.int32)):
    i, j = wp.tid()
    if j < u.shape[1]:
        if not wp.isfinite(u[i, j]):
            u[i, j] = 0.0
            wp.atomic_add(counter, 0, 1)
    if i < v.shape[0]:
        if not wp.isfinite(v[i, j]):
            v[i, j] = 0.0
            wp.atomic_add(counter, 0, 1)


@wp.kernel
def max_speed_kernel(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), valid_u: wp.array2d(dtype=wp.int32),
                     valid_v: wp.array2d(dtype=wp.int32), n1: int, total: int, block: int,
                     out: wp.array(dtype=float)):
    """max |u| over the faces touching liquid (valid == 1) of both grids into out[0] (block reduction)."""
    k = wp.tid()
    a = float(0.0)
    if k < total:
        i = k // n1
        j = k - i * n1
        if j < u.shape[1]:
            if valid_u[i, j] == 1:
                a = wp.abs(u[i, j])
        if i < v.shape[0]:
            if valid_v[i, j] == 1:
                a = wp.max(a, wp.abs(v[i, j]))
        if not wp.isfinite(a):
            a = 1.0e30
    m = wp.tile_max(wp.tile(a))
    if k % block == 0:
        wp.atomic_max(out, 0, m[0])


@wp.kernel
def measure_kernel(phi: wp.array2d(dtype=float), u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), dx: float,
                   rho: float, n1: int, total: int, slot: wp.array(dtype=wp.int32), area: wp.array(dtype=float),
                   vol_h: wp.array(dtype=float), cells: wp.array(dtype=float), ke: wp.array(dtype=float)):
    """Per dual cell: marching-squares liquid area; per cell: smoothed-Heaviside volume, liquid count and kinetic
    energy (cell-centred velocity); block-reduced into slot ``slot[0]`` of the history buffers."""
    k = wp.tid()
    a = float(0.0)
    h = float(0.0)
    c = float(0.0)
    e = float(0.0)
    if k < total:
        I = k // n1
        J = k - I * n1
        a = dual_cell_liquid_area(phi, I, J, dx)
        if I < phi.shape[0] and J < phi.shape[1]:
            p = phi[I, J]
            h = wp.clamp(0.5 - p / dx, 0.0, 1.0) * dx * dx
            if p < 0.0:
                c = 1.0
                uc = 0.5 * (u[I, J] + u[I + 1, J])
                vc = 0.5 * (v[I, J] + v[I, J + 1])
                e = 0.5 * rho * (uc * uc + vc * vc) * dx * dx
    s = slot[0]
    wp.tile_atomic_add(area, wp.tile_sum(wp.tile(a)), offset=s)
    wp.tile_atomic_add(vol_h, wp.tile_sum(wp.tile(h)), offset=s)
    wp.tile_atomic_add(cells, wp.tile_sum(wp.tile(c)), offset=s)
    wp.tile_atomic_add(ke, wp.tile_sum(wp.tile(e)), offset=s)


@wp.kernel
def record_kernel(speed: wp.array(dtype=float), stats: wp.array(dtype=float), slot: wp.array(dtype=wp.int32),
                  max_speed: wp.array(dtype=float), its: wp.array(dtype=float), res: wp.array(dtype=float)):
    """Per-step scalars into the history slot, then advance the slot."""
    s = slot[0]
    max_speed[s] = speed[0]
    its[s] = stats[0]
    res[s] = stats[1]
    slot[0] = s + 1


@wp.kernel
def cell_speed_kernel(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
    i, j = wp.tid()
    uc = 0.5 * (u[i, j] + u[i + 1, j])
    vc = 0.5 * (v[i, j] + v[i, j + 1])
    out[i, j] = wp.sqrt(uc * uc + vc * vc)


@wp.kernel
def area_sum_kernel(phi: wp.array2d(dtype=float), dx: float, n1: int, total: int, out: wp.array(dtype=float)):
    """Sum of the marching-squares liquid areas of all dual cells (flattened launch, block reduction)."""
    k = wp.tid()
    a = float(0.0)
    if k < total:
        I = k // n1
        a = dual_cell_liquid_area(phi, I, k - I * n1, dx)
    wp.tile_atomic_add(out, wp.tile_sum(wp.tile(a)), offset=0)


class FreeSurfaceSolver2D:
    """2D incompressible free-surface solver (MAC grid, ghost-fluid pressure, PFM-LS interface).

    Args:
        cfg: solver parameters.
        sdf: vectorised signed distance ``sdf(x, y) -> phi`` of the initial liquid (negative inside).
        velocity: optional initial velocity ``velocity(x, y) -> (u, v)`` (vectorised); it is projected.
        device: Warp device.
    """

    #: steps of per-step diagnostics buffered on the device before a host copy
    HISTORY_BATCH = 512
    #: per-step diagnostics: marching-squares liquid area (the 2D "volume"), smoothed-Heaviside volume,
    #: liquid cell count, liquid kinetic energy, CG iterations, CG residual and max speed on liquid faces (plus
    #: ``dt``, kept on the host)
    HIST_KEYS = ("volume", "volume_h", "cells", "ke", "cg_iters", "cg_residual", "max_speed")
    #: bound on the cached CUDA graphs of the step (one per time-step level and step type)
    MAX_GRAPHS = 64

    def __init__(self, cfg: FluidConfig, sdf: Callable, velocity: Callable | None = None, device=None):
        self.cfg = cfg
        self.device = wp.get_device(device)
        self.grid = Grid2D(cfg.nx, cfg.ny, cfg.dx)
        self.nx, self.ny, self.dx = cfg.nx, cfg.ny, cfg.dx
        self.lx, self.ly = cfg.nx * cfg.dx, cfg.ny * cfg.dx
        self.sdf = sdf
        self.t = 0.0
        self.step_count = 0
        self.dt_nominal = cfg.dt
        if cfg.capillary_cfl > 0.0:
            self.dt_nominal = min(cfg.dt, cfg.capillary_cfl * self.capillary_dt())
        dev = self.device
        nx, ny = self.nx, self.ny
        self.u = wp.zeros((nx + 1, ny), dtype=float, device=dev)
        self.v = wp.zeros((nx, ny + 1), dtype=float, device=dev)
        self.valid_u = wp.zeros(self.u.shape, dtype=wp.int32, device=dev)
        self.valid_v = wp.zeros(self.v.shape, dtype=wp.int32, device=dev)
        self.ext = FaceExtrapolator(self.u.shape, self.v.shape, dev)
        self.pressure = PressureSolver(nx, ny, self.dx, cfg.pressure, dev)
        self.vel_scrubs = wp.zeros(1, dtype=wp.int32, device=dev)
        self._block = BLOCK if dev.is_cuda else 1
        self._hist = {k: wp.zeros(self.HISTORY_BATCH, dtype=float, device=dev) for k in self.HIST_KEYS}
        self._slot = wp.zeros(1, dtype=wp.int32, device=dev)
        self._hist_t = np.zeros(self.HISTORY_BATCH)
        self._hist_dt = np.zeros(self.HISTORY_BATCH)
        self._hist_n = 0
        self.history = {k: [] for k in ("step", "t", "dt") + self.HIST_KEYS}
        self._cg_cap_hits = 0
        self._speed = wp.zeros(1, dtype=float, device=dev)
        self._area = wp.zeros(1, dtype=float, device=dev)
        self._speed_host = wp.zeros(2, dtype=float, device="cpu", pinned=dev.is_cuda)
        self._speed_events = [wp.Event(dev), wp.Event(dev)] if dev.is_cuda else None
        self._speed_ready = 2  # first step whose lagged speed is available
        self._graphs = GraphCache(dev, self.MAX_GRAPHS, cfg.use_graph)
        self._cell_speed = None

        self.ls = PFMLevelSetBackend(cfg, sdf, dev, self.dt_nominal)
        if velocity is not None:
            self._set_velocity(velocity)
            self.pressure.project(self.u, self.v, self.phi, 0.0, 0.0, cfg.rho, self.valid_u, self.valid_v,
                                  separation=False)
            self.pressure.stats.zero_()
        else:
            self._mark_valid()
        self._extend()
        self.transport = make_transport(cfg)
        self.transport.initialize(self)
        self.energy_budget = self._energy_budget()
        self.volume0 = None
        self._record_device()
        self._record_host(0.0)
        self._flush_history()
        self.volume0 = self.history["volume"][0]

    # ------------------------------------------------------------------------------------------ setup
    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre coordinates as two ``(nx, ny)`` arrays (``indexing='ij'``)."""
        return self.grid.cell_centers()

    def _set_velocity(self, velocity: Callable) -> None:
        dx = self.dx
        xu, yu = np.meshgrid(np.arange(self.nx + 1) * dx, (np.arange(self.ny) + 0.5) * dx, indexing="ij")
        xv, yv = np.meshgrid((np.arange(self.nx) + 0.5) * dx, np.arange(self.ny + 1) * dx, indexing="ij")
        uu = np.asarray(velocity(xu, yu)[0], dtype=np.float32) * np.ones_like(xu, dtype=np.float32)
        vv = np.asarray(velocity(xv, yv)[1], dtype=np.float32) * np.ones_like(xv, dtype=np.float32)
        uu[0, :] = 0.0
        uu[-1, :] = 0.0
        vv[:, 0] = 0.0
        vv[:, -1] = 0.0
        self.u.assign(uu)
        self.v.assign(vv)

    def _mark_valid(self) -> None:
        phi = self.phi.numpy()
        liq = phi < 0
        vu = np.zeros(self.u.shape, np.int32)
        vu[1:-1, :] = liq[1:, :] | liq[:-1, :]
        vu[0, :], vu[-1, :] = liq[0, :], liq[-1, :]
        vv = np.zeros(self.v.shape, np.int32)
        vv[:, 1:-1] = liq[:, 1:] | liq[:, :-1]
        vv[:, 0], vv[:, -1] = liq[:, 0], liq[:, -1]
        self.valid_u.assign(vu)
        self.valid_v.assign(vv)

    def _energy_budget(self) -> float:
        """Upper bound of the liquid kinetic energy for an inviscid flow: initial kinetic energy + potential energy
        above the flat layer of the same area + surface energy."""
        cfg, dx = self.cfg, self.dx
        phi = self.phi.numpy().astype(np.float64)
        uc, vc = self.cell_velocity()
        heav = np.clip(0.5 - phi / dx, 0.0, 1.0)
        ke = 0.5 * cfg.rho * float(np.sum(((uc ** 2 + vc ** 2) * (phi < 0)))) * dx * dx
        g = np.asarray(cfg.gravity, dtype=np.float64)
        pe = 0.0
        if np.any(g != 0.0):
            axis = int(np.argmax(np.abs(g)))
            n = phi.shape[axis]
            h = (np.arange(n) + 0.5) * dx
            if g[axis] > 0.0:
                h = n * dx - h
            h = h[:, None] if axis == 0 else h[None, :]
            area = float(heav.sum()) * dx * dx
            width = phi.shape[1 - axis] * dx
            pe = cfg.rho * abs(g[axis]) * (float(np.sum(heav * h)) * dx * dx - area * area / (2.0 * width))
        eps = 1.5 * dx
        hs = np.clip(0.5 + phi / (2.0 * eps) + np.sin(np.pi * phi / eps) / (2.0 * np.pi), 0.0, 1.0)
        hs = np.where(phi < -eps, 0.0, np.where(phi > eps, 1.0, hs))
        gx, gy = np.gradient(hs, dx)
        length = float(np.sum(np.hypot(gx, gy))) * dx * dx
        return ke + max(pe, 0.0) + cfg.sigma * length

    # ------------------------------------------------------------------------------------------- step
    @property
    def phi(self) -> wp.array2d:
        """Current cell-centred level set on the simulation grid (negative in the liquid)."""
        return self.ls.phi

    def max_speed(self) -> float:
        """Max |u| over liquid faces (host read)."""
        self._speed.zero_()
        self._launch_max_speed(self._speed)
        return float(self._speed.numpy()[0])

    def _launch_max_speed(self, out: wp.array) -> None:
        n1 = self.ny + 1
        total = (self.nx + 1) * n1
        wp.launch(max_speed_kernel, dim=self._flat_dim(total),
                  inputs=[self.u, self.v, self.valid_u, self.valid_v, n1, total, self._block], outputs=[out],
                  block_dim=self._block, device=self.device)

    def _flat_dim(self, total: int) -> int:
        b = self._block
        return ((total + b - 1) // b) * b

    def capillary_dt(self) -> float:
        """Capillary time-step limit sqrt(rho dx^3 / (2 pi sigma)) (inf without surface tension)."""
        if self.cfg.sigma <= 0.0:
            return math.inf
        return math.sqrt(self.cfg.rho * self.dx ** 3 / (2.0 * math.pi * self.cfg.sigma))

    def _recent_speed(self) -> float:
        """Max liquid-face speed of the step before last (current speed for the first steps)."""
        if self.step_count < self._speed_ready:
            return self.max_speed()
        k = (self.step_count - 1) % 2
        if self._speed_events is not None:
            wp.synchronize_event(self._speed_events[k])
        return float(self._speed_host.numpy()[k])

    def _level_dt(self, k: int) -> float:
        return self.dt_nominal * 2.0 ** (-k / self.cfg.cfl_levels)

    def _level_index(self, dt: float) -> int | None:
        """Index k of the time-step level equal to ``dt`` (None if ``dt`` is not a level)."""
        if dt <= 0.0 or dt > self.dt_nominal:
            return None
        k = int(round(self.cfg.cfl_levels * math.log2(self.dt_nominal / dt)))
        return k if math.isclose(self._level_dt(k), dt, rel_tol=1.0e-12) else None

    def choose_dt(self, dt_max: float | None = None) -> float:
        """Time step for the next step: the upper bound ``dt_max`` (default ``cfg.dt``) limited by the capillary and
        CFL bounds; a CFL-limited nominal step is rounded down to a time-step level."""
        req = self.cfg.dt if dt_max is None else float(dt_max)
        base = min(req, self.dt_nominal)
        if self.cfg.cfl <= 0.0:
            return base
        umax = self._recent_speed()
        if umax <= 1.0e-12:
            return base
        limit = self.cfg.cfl * self.dx / umax
        if limit >= base:
            return base
        if base < self.dt_nominal:
            return limit
        k = max(1, math.ceil(self.cfg.cfl_levels * math.log2(base / limit) - 1.0e-9))
        return self._level_dt(k)

    def step(self, dt_max: float | None = None) -> float:
        """Advance one time step of at most ``dt_max`` (default ``cfg.dt``; the CFL and capillary limits of
        :meth:`choose_dt` apply, so the step can be shorter); returns the dt used."""
        dt = self.choose_dt(dt_max)
        reseed = self.transport.reseed_due()
        level = self._level_index(dt)
        self._graphs.run(("pre",), self._pre_levelset)
        self.ls.step_core(dt)
        post = lambda: self._post_levelset(dt, reseed)  # noqa: E731
        if level is not None:  # a step of another length runs eagerly
            self._graphs.run(("post", level, reseed), post)
        else:
            post()
        self.transport.end_step()
        self.t += dt
        self.step_count += 1
        self._record_host(dt)
        return dt

    def _pre_levelset(self) -> None:
        wp.launch(scrub_velocity_kernel, dim=self.ext.dim, inputs=[self.u, self.v, self.vel_scrubs],
                  device=self.device)
        self.ls.pre_step(self.u, self.v)

    def _post_levelset(self, dt: float, reseed: bool) -> None:
        cfg, dev = self.cfg, self.device
        g = cfg.gravity
        self.ls.post_step(self.u, self.v)
        self.transport.advect(self, dt)
        wp.launch(body_force_kernel, dim=self.ext.dim,
                  inputs=[self.u, self.v, wp.vec2(float(g[0]) * dt, float(g[1]) * dt)], device=dev)
        source = self.ls.curvature_source() if cfg.sigma != 0.0 else None
        self.pressure.project(self.u, self.v, self.phi, dt, cfg.sigma, cfg.rho, self.valid_u, self.valid_v, source,
                              None if self.ls.pad > 0 else False)
        self._extend()
        self.transport.after_projection(self, reseed)
        self._record_device()

    def _extend(self) -> None:
        cfg = self.cfg
        self.ext.extend(self.u, self.v, self.valid_u, self.valid_v, cfg.extension_passes,
                        zero_rest=not cfg.ballistic_air)
        clamp_wall_faces(self.u, self.v, self.device)

    def advance_to(self, t_end: float, callback: Callable | None = None, max_steps: int | None = None) -> int:
        """Step until ``t_end`` (at most ``max_steps`` steps; the last step is shortened to land on it), calling
        ``callback(solver)`` after every step; returns the number of steps."""
        n = 0
        while self.t < t_end - 1.0e-9 and (max_steps is None or n < max_steps):
            self.step(min(self.cfg.dt, t_end - self.t))
            n += 1
            if callback is not None:
                callback(self)
        return n

    # ------------------------------------------------------------------------------------ diagnostics
    def _record_device(self) -> None:
        """Per-step diagnostics into the history slot ``_slot`` (device work only)."""
        h, dev = self._hist, self.device
        self._speed.zero_()
        self._launch_max_speed(self._speed)
        n1 = self.ny + 1
        total = (self.nx + 1) * n1
        wp.launch(measure_kernel, dim=self._flat_dim(total),
                  inputs=[self.phi, self.u, self.v, self.dx, self.cfg.rho, n1, total, self._slot],
                  outputs=[h["volume"], h["volume_h"], h["cells"], h["ke"]], block_dim=self._block, device=dev)
        wp.launch(record_kernel, dim=1, inputs=[self._speed, self.pressure.stats, self._slot],
                  outputs=[h["max_speed"], h["cg_iters"], h["cg_residual"]], device=dev)

    def _record_host(self, dt: float) -> None:
        """Host side of the diagnostics: the lagged-speed copy and the time of the slot just written."""
        if self.cfg.cfl > 0.0:
            k = self.step_count % 2
            wp.copy(self._speed_host, self._speed, dest_offset=k, count=1)
            if self._speed_events is not None:
                wp.get_stream(self.device).record_event(self._speed_events[k])
        self._hist_t[self._hist_n] = self.t
        self._hist_dt[self._hist_n] = dt
        self._hist_n += 1
        if self._hist_n == self.HISTORY_BATCH:
            self._flush_history()

    def _flush_history(self) -> None:
        n = self._hist_n
        if n == 0:
            return
        s0 = self.step_count - n + 1
        self.history["step"].extend(range(s0, s0 + n))
        self.history["t"].extend(self._hist_t[:n].tolist())
        self.history["dt"].extend(self._hist_dt[:n].tolist())
        for k in self.HIST_KEYS:
            vals = self._hist[k].numpy()[:n]
            self.history[k].extend(vals.tolist())
            self._hist[k].zero_()
            if k == "cg_iters":
                self._cg_cap_hits += int(np.count_nonzero(vals >= self.cfg.pressure.max_iters))
        self._slot.zero_()
        self._hist_n = 0

    def volume_series(self) -> dict:
        """Per-step diagnostics as numpy arrays: ``step``, ``t``, ``dt``, ``volume`` and ``area`` (both the
        marching-squares area of ``{phi < 0}`` on the grid), ``volume_h`` (smoothed Heaviside), ``cells``, ``ke``
        (liquid kinetic energy), ``cg_iters``, ``cg_residual``, ``max_speed``."""
        self._flush_history()
        out = {k: np.asarray(v) for k, v in self.history.items()}
        out["area"] = out["volume"]
        return out

    def health(self) -> dict:
        """Hard-failure counters (cheap host reads): ``velocity_nan_scrubs``, ``cg_cap_hits`` and the level-set
        backend's ``nan_scrubs`` / ``nan_particles`` (see :meth:`failure_reason`)."""
        self._flush_history()
        h = {"velocity_nan_scrubs": int(self.vel_scrubs.numpy()[0]), "cg_cap_hits": self._cg_cap_hits}
        h.update(self.ls.health())
        return h

    def failure_reason(self) -> str | None:
        """The first hard failure so far (None if healthy; see :func:`failure_reason`)."""
        return failure_reason(self.health())

    def stats(self) -> dict:
        """Robustness and bookkeeping counters (host reads): the step and time, velocity NaN scrubs, cached CUDA
        graphs, and those of the level-set backend (rejected reinitialisations, NaN repairs, film closure, ...).
        CG iterations are in :meth:`volume_series`."""
        s = {"step": self.step_count, "t": self.t, "velocity_nan_scrubs": int(self.vel_scrubs.numpy()[0]),
             "cuda_graphs": len(self._graphs)}
        s.update(self.ls.stats())
        return s

    def cell_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centred velocity (host arrays)."""
        u = self.u.numpy()
        v = self.v.numpy()
        return 0.5 * (u[1:, :] + u[:-1, :]), 0.5 * (v[:, 1:] + v[:, :-1])

    def cell_speed_device(self) -> wp.array2d:
        """Cell-centred speed |u| (device array, reused)."""
        if self._cell_speed is None:
            self._cell_speed = wp.zeros((self.nx, self.ny), dtype=float, device=self.device)
        wp.launch(cell_speed_kernel, dim=(self.nx, self.ny), inputs=[self.u, self.v], outputs=[self._cell_speed],
                  device=self.device)
        return self._cell_speed

    def fine_phi(self, factor: int = 4) -> np.ndarray:
        """Fine reconstruction of the level set (host array, ``factor`` times the grid resolution)."""
        return self.ls.fine_phi(factor)

    def fine_phi_device(self, factor: int = 4) -> wp.array2d:
        """Fine reconstruction of the level set (device array, reused by the backend)."""
        return self.ls.fine_phi_device(factor)

    def liquid_area(self, phi: wp.array2d, h: float) -> float:
        """Marching-squares area of ``{phi < 0}`` of a cell-centred device field with spacing ``h`` covering the
        box (the level set extrapolated linearly to the walls); one scalar host read."""
        n1 = phi.shape[1] + 1
        total = (phi.shape[0] + 1) * n1
        self._area.zero_()
        wp.launch(area_sum_kernel, dim=self._flat_dim(total), inputs=[phi, h, n1, total], outputs=[self._area],
                  block_dim=self._block, device=self.device)
        return float(self._area.numpy()[0])

    def velocity_particle_positions(self) -> np.ndarray | None:
        """Positions of the APIC particles (host array; None for the semi-Lagrangian transport).  The level-set
        particles are ``solver.ls.ls.particles()``."""
        fn = getattr(self.transport, "particle_positions", None)
        return fn() if fn is not None else None

    # ------------------------------------------------------------------------------------- checkpoint
    def save_checkpoint(self, path: str) -> None:
        """Write the full solver state (grid, level set, transport, history) to an ``.npz`` file."""
        self._flush_history()
        data = {"u": self.u.numpy(), "v": self.v.numpy(), "p": self.pressure.p.numpy(),
                "valid_u": self.valid_u.numpy(), "valid_v": self.valid_v.numpy(),
                "t": np.float64(self.t), "step": np.int64(self.step_count),
                "volume0": np.float64(self.volume0), "vel_scrubs": self.vel_scrubs.numpy(),
                "config": np.bytes_(json.dumps(asdict(self.cfg), default=str))}
        for k, v in self.history.items():
            data["hist_" + k] = np.asarray(v)
        for k, v in self.ls.state_dict().items():
            data["ls_" + k] = np.asarray(v)
        for k, v in self.transport.state_dict().items():
            data["tr_" + k] = np.asarray(v)
        np.savez_compressed(path, **data)

    def load_checkpoint(self, path: str) -> None:
        """Restore a state written by :meth:`save_checkpoint`.

        The checkpoint must come from a solver with the same grid, velocity transport and level-set configuration
        (:data:`CHECKPOINT_FIELDS`); otherwise ``ValueError`` is raised before any state changes.
        """
        d = np.load(path, allow_pickle=False)
        self._check_checkpoint(d, path)
        self.u.assign(d["u"])
        self.v.assign(d["v"])
        self.pressure.p.assign(d["p"])
        self.valid_u.assign(d["valid_u"])
        self.valid_v.assign(d["valid_v"])
        self.t = float(d["t"])
        self.step_count = int(d["step"])
        self._speed_ready = self.step_count + 2
        self.volume0 = float(d["volume0"])
        self.vel_scrubs.assign(d["vel_scrubs"])
        self.history = {k[5:]: d[k].tolist() for k in d.files if k.startswith("hist_")}
        self._cg_cap_hits = int(np.count_nonzero(np.asarray(self.history["cg_iters"]) >= self.cfg.pressure.max_iters))
        for a in self._hist.values():
            a.zero_()
        self._slot.zero_()
        self._hist_n = 0
        self.ls.load_state_dict({k[3:]: d[k] for k in d.files if k.startswith("ls_")})
        self.transport.load_state_dict({k[3:]: d[k] for k in d.files if k.startswith("tr_")}, self)

    def _check_checkpoint(self, d, path: str) -> None:
        missing = [k for k in ("u", "v", "p", "valid_u", "valid_v", "t", "step", "volume0", "vel_scrubs", "config")
                   if k not in d.files]
        if not any(k.startswith("ls_") for k in d.files):
            missing.append("ls_*")
        if missing:
            raise ValueError(f"{path} is not a FreeSurfaceSolver2D checkpoint (missing {missing})")
        saved = json.loads(d["config"].item().decode())
        mine = json.loads(json.dumps(asdict(self.cfg), default=str))
        diff = [f"{k}: checkpoint {saved.get(k)!r}, solver {mine[k]!r}" for k in CHECKPOINT_FIELDS
                if saved.get(k) != mine[k]]
        if diff:
            raise ValueError(f"{path} was written with another configuration: " + "; ".join(diff))
