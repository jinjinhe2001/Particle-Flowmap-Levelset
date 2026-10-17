"""PFMLevelSet2D: the 2D Particle Flow Map Level Set method.

State
-----
* particles on a ``sqrt(ppc) x sqrt(ppc)`` lattice per cell; ``valid`` marks the narrow band
  ``|phi| < band dx`` chosen at each long reinitialisation (other slots are idle until adaptive seeding
  stores a particle in them);
* the background grid level set ``phi_g`` (cell centres).

One step (``i`` = step counter)
-------------------------------
A. long reinitialisation (when due): bin particles, Shepard gradients, particle-to-grid
   reconstruction (cells with a particle within ``p2g_radius dx``), fill the rest from the grid,
   eikonal redistancing + quasi-Newton closest-point projection near the interface + Hermite
   semi-Lagrangian eikonal sweeps outside, reseed every lattice slot from the Hermite grid
   interpolant blended with the old particles, reset the long map;
C. short reinitialisation (when due, and always together with A): Shepard gradients,
   particle-to-grid normals, ``C0 = grad n`` from MAC-averaged normals (quadratic B-spline), reset
   the short map;
D. RK4 advection of positions and of the short-map Jacobian ``dJ/dt = -J grad u``, followed in the
   same kernel by F. ``C = Js^T C0 Js`` and H. the long-map gradient ``g = normalize(J_long^T g0)``;
E. re-bin;
G. semi-Lagrangian (RK2) advection + eikonal redistancing of the background grid;
I. composite: grid cells with a particle within ``p2g_radius dx`` take the particle value;
S. adaptive seeding (default ``seed_min = ppc / 2``): one new particle per band cell holding fewer than
   ``seed_min`` particles (by default these *newborns* only fill the holes of the fine reconstruction, see
   ``composite_newborns``).

``phi_p`` itself is constant between long reinitialisations. All per-step work is asynchronous;
on CUDA the step is captured into a CUDA graph (one per step type and velocity) and replayed.

Reinitialisation schedule
-------------------------
``reinit_long`` / ``reinit_short`` are intervals in steps (``reinit_unit="steps"``: long when
``i % reinit_long == 0``, short every ``reinit_short`` steps after the last long one) or in time
units (``reinit_unit="time"``: a reinitialisation runs at the step that brings the time since the
last one to within ``dt / 2`` of the interval). ``0`` disables the periodic reinitialisation (both
still run once at initialisation); ``0 / 0`` is the pure advection mode. Every long
reinitialisation perturbs the interface slightly (with ``hessian_scale = 1`` convex shapes shrink
by about ``1e-4`` of their area per reinitialisation at a 10-cell radius, stretched filaments gain
area), so the number of long reinitialisations per unit time must not grow as ``dt`` shrinks:
scale step intervals with ``1 / dt`` or use ``reinit_unit="time"``, and prefer
``hessian_scale = 0.5`` when reinitialisations are frequent. The benchmarks use 12 / 6 steps of
``dt = 0.02``, i.e. 0.24 / 0.12 time units (see ``docs/method.md``).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..core import measure
from ..core import options as _opt
from ..core.celllist import CellList2D
from ..core.eikonal import Eikonal2D
from ..core.grid import Grid2D
from . import particles2d as K
from ._base import PFMLevelSetBase, check_sign_change, checked_grid_sdf, checked_sdf
from .newton import QuasiNewtonRedistancer

#: Named protocols of the 2D benchmarks: ``(reinit_long, reinit_short, seed_fraction)``, intervals in steps and the
#: adaptive seeding threshold ``seed_min = seed_fraction * ppc`` (a fraction of ppc, unlike the absolute
#: ``seed_min`` of :data:`pfmls.levelset.pfmls3d.PROTOCOLS`).
PROTOCOLS = {
    "full": (12, 6, 0.5),        # long reinitialisation with redistancing + reseeding every 12 steps, short every 6
    "advection": (0, 0, 0.5),    # pure advection: both reinitialisations only at t = 0 (the paper's "advection
                                 # only")
}                                # both with adaptive seeding at the paper's n_min = ppc / 2

#: Modes of ``composite_newborns`` / ``reconstruct_newborns`` (see :class:`PFMLS2DOptions`).
NEWBORN_MODES = {"mixed": K.NEWBORNS_MIXED, "off": K.NEWBORNS_OFF, "fallback": K.NEWBORNS_FALLBACK}


@dataclass(frozen=True)
class PFMLS2DOptions:
    """Advanced parameters (immutable).

    The defaults are the settings of the 2D benchmarks (Zalesak also uses ``p2g_weights="bspline"``);
    ``docs/method.md``, section 2, explains them. Values are type-checked (``TypeError``);
    :func:`parse_options` converts command-line ``KEY=VALUE`` strings.
    """

    #: radius (in dx) of every particle -> point reconstruction: the cell-centre transfer at
    #: reinitialisation, the per-step composite, the fine reconstruction and the KNN interpolation
    #: used by reseeding and the Newton projection; must lie in (0, 0.5)
    p2g_radius: float = 0.25
    #: reconstruction weights: "uniform" (mean of the particles within the radius) or "bspline"
    #: (N2(d / h) with h the target grid spacing; the composite uses h = dx / composite_factor)
    p2g_weights: str = "uniform"
    #: scale s of the normal gradient C0 = s grad n built at short reinitialisation, which enters the
    #: particle extrapolation phi_p + normalize(g_p + C_p d) . d: 1 = ``C0 = grad n`` (twice the
    #: second-order Taylor term of a signed distance; convex shapes shrink slightly at every long
    #: reinitialisation), 0.5 = Taylor-consistent (recommended with frequent reinitialisation,
    #: see docs/method.md section 3)
    hessian_scale: float = 1.0
    #: reseed blend: particle value if n_H . n_P > e1, linear blend down to e2, grid value below
    blend_e1: float = 0.9
    blend_e2: float = 0.7
    #: particle value only if |phi_P| < blend_gate dx and |phi_H| <= blend_grid_gate dx
    blend_gate: float = 0.25
    blend_grid_gate: float = 1.0
    #: quasi-Newton band dilations
    newton_dilations: int = 3
    #: quasi-Newton iteration cap (the iteration starts at x_s - phi n; raise the cap for bands wider
    #: than 4 dx)
    newton_max_iter: int = 64
    #: quasi-Newton step clamp (dx)
    newton_step_clamp: float = 0.02
    #: quasi-Newton tolerance (dx)
    newton_tol: float = 1.0e-4
    #: semi-Lagrangian eikonal sweeps before (bilinear) and after (Hermite) the Newton step
    sl_linear_sweeps: int = 5
    sl_hermite_sweeps: int = 30
    sl_step: float = 0.9
    #: Jacobi iterations of the eikonal initial guess at reinitialisation. Cells within about
    #: 0.7 * iterations cells of the interface get their converged distance, the far field keeps its
    #: sign-correct input; None = nx + ny
    eikonal_iterations: int | None = 48
    #: Jacobi iterations of the per-step background-grid redistancing (same rule; cells further
    #: away keep their sign-correct semi-Lagrangian values; see docs/method.md section 1)
    grid_eikonal_iterations: int | None = 16

    # --- adaptive seeding ---
    #: every step, each band cell (|phi_g| < band dx) holding fewer than seed_min particles gets one candidate
    #: particle (None = ppc / 2, the paper's n_min; 0 = off). Candidates whose neighbours do not support them are
    #: rejected (see seed_coherence and docs/method.md section 1); the accepted ones take idle slots and live until
    #: the next long reinitialisation
    seed_min: int | None = None
    #: a new particle is rejected if the Taylor predictions of its neighbours disagree in sign,
    #: |sum w v| < seed_coherence sum w |v| (0 = no check)
    seed_coherence: float = 0.9
    #: how the newborns enter the per-step composite and the fine reconstruction: "off" = not at all,
    #: "fallback" = only at points that no lattice particle covers, "mixed" = averaged with the lattice
    #: particles. With the composite default "off" the newborns never enter the dynamics (the evolved state
    #: equals that of an unseeded run); otherwise the reinitialisations read them like lattice particles
    composite_newborns: str = "off"
    reconstruct_newborns: str = "fallback"

    # --- alternative variants (the defaults are the recommended settings) ---
    #: evaluate the velocity at the step's start time in every RK stage of the particles and the
    #: grid (first order in time)
    frozen_time: bool = False
    #: recompute particle gradients by Shepard interpolation at long reinitialisation (False: keep
    #: the transported gradients)
    shepard_at_long_reinit: bool = True
    #: run the grid redistancing at the initial long reinitialisation
    redistance_at_init: bool = True
    #: refinement of the composite grid of the "bspline" weights (h = dx / composite_factor)
    composite_factor: int = 5

    def __post_init__(self):
        _opt.check_field_types(self)
        if self.p2g_weights not in ("uniform", "bspline"):
            raise ValueError(f"p2g_weights must be 'uniform' or 'bspline', got {self.p2g_weights!r}")
        if not 0.0 < self.p2g_radius < 0.5:
            raise ValueError(f"p2g_radius must lie in (0, 0.5) (dx), got {self.p2g_radius}")
        if self.hessian_scale < 0.0:
            raise ValueError(f"hessian_scale must be >= 0, got {self.hessian_scale}")
        if not self.blend_e1 > self.blend_e2:
            raise ValueError(f"need blend_e1 > blend_e2, got {self.blend_e1} <= {self.blend_e2}")
        if self.composite_factor < 1:
            raise ValueError(f"composite_factor must be >= 1, got {self.composite_factor}")
        for name in ("composite_newborns", "reconstruct_newborns"):
            if getattr(self, name) not in NEWBORN_MODES:
                raise ValueError(f"{name} must be one of {sorted(NEWBORN_MODES)}, got {getattr(self, name)!r}")
        if not 0.0 <= self.seed_coherence <= 1.0:
            raise ValueError(f"seed_coherence must lie in [0, 1], got {self.seed_coherence}")
        for name in ("newton_dilations", "newton_max_iter", "sl_linear_sweeps", "sl_hermite_sweeps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.seed_min is not None and self.seed_min < 0:
            raise ValueError("seed_min must be >= 0 or None")
        for name in ("eikonal_iterations", "grid_eikonal_iterations"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"{name} must be >= 0 or None")


def parse_options(pairs) -> dict:
    """Typed :class:`PFMLS2DOptions` keyword arguments from ``"KEY=VALUE"`` strings (command lines).

    Values are converted by the field's annotation: bools accept ``1/0, true/false, yes/no, on/off``,
    ``none`` gives None where allowed. Raises ``ValueError`` for an unknown key or an unconvertible value.
    """
    return _opt.parse_options(PFMLS2DOptions, pairs)


@wp.kernel
def _fill_scalar(m: wp.array2d(dtype=wp.int32), src: wp.array2d(dtype=float), dst: wp.array2d(dtype=float)):
    i, j = wp.tid()
    if m[i, j] == 0:
        dst[i, j] = src[i, j]


@wp.kernel
def _fill_vector(m: wp.array2d(dtype=wp.int32), src: wp.array2d(dtype=wp.vec2), dst: wp.array2d(dtype=wp.vec2)):
    i, j = wp.tid()
    if m[i, j] == 0:
        dst[i, j] = src[i, j]


@wp.kernel
def _one_sided_gradient(phi: wp.array2d(dtype=float), g: wp.array2d(dtype=wp.vec2), inv_dx: float):
    """Forward differences (backward on the last row/column), unnormalised."""
    i, j = wp.tid()
    nx = phi.shape[0]
    ny = phi.shape[1]
    gx = float(0.0)
    gy = float(0.0)
    if i < nx - 1:
        gx = (phi[i + 1, j] - phi[i, j]) * inv_dx
    else:
        gx = (phi[i, j] - phi[i - 1, j]) * inv_dx
    if j < ny - 1:
        gy = (phi[i, j + 1] - phi[i, j]) * inv_dx
    else:
        gy = (phi[i, j] - phi[i, j - 1]) * inv_dx
    g[i, j] = wp.vec2(gx, gy)


#: Device arrays that define the level-set state (see :meth:`PFMLevelSet2D.snapshot`); with adaptive seeding also
#: SEEDING_ARRAYS (the free-slot list of the newborns).
STATE_ARRAYS = ("x", "phi", "g", "g0", "valid", "Js", "Jinit", "C", "C0", "phi_g", "_phi_s")
SEEDING_ARRAYS = ("_free", "_free_count", "_n_new")


class PFMLevelSet2D(PFMLevelSetBase):
    """Particle flow map level set on a 2D cell-centred grid.

    Parameters
    ----------
    res:
        Grid resolution ``n`` or ``(nx, ny)``. The domain is ``[0, nx dx] x [0, ny dx]``.
    ppc:
        Particles per cell: a perfect square >= 9 such that a lattice point lies within
        ``p2g_radius dx`` of every cell centre (9, 16, 25, 36, ...).
    band:
        Half-width (in dx) of the particle band chosen at each long reinitialisation. 1-2 is the
        validated range for deforming flows; wider bands lose accuracy in stretching flows (see
        ``docs/method.md``), rigid motion is insensitive to it. Bands wider than ``WIDE_BAND`` (4 dx)
        trigger a ``RuntimeWarning``.
    reinit_long, reinit_short:
        Long / short reinitialisation intervals (0 = never after initialisation).
    reinit_unit:
        ``"steps"`` (intervals are step counts, the default) or ``"time"`` (time units).
    dx:
        Cell size (default ``1 / nx``).
    device:
        Warp device (``"cuda:0"``, ``"cpu"``, ...).
    use_graph:
        Capture each step type into a CUDA graph (ignored on CPU). Graphs are keyed on the step type
        and the velocity's face arrays, so a MAC velocity should be updated in place (or cycle through
        a few buffers) rather than reallocated every step.
    options:
        A :class:`PFMLS2DOptions`; keyword ``overrides`` replace individual fields.

    Attributes
    ----------
    phi_g:
        Background grid level set ``(nx, ny)`` on the device (the composite of particle and grid
        values; ``phi < 0`` inside; never reallocated). It may be edited between steps (e.g. ghost
        values at an open boundary): the edit enters the next step's semi-Lagrangian advection and the
        long reinitialisation's fill of cells without particles; the particles, the redistanced grid
        behind the composite and cached fine reconstructions do not see it.
    x, phi, g, valid:
        Particle slot arrays on the device (``ppc * nx * ny`` slots, never reallocated; see
        :meth:`particles` and :meth:`scrub_nonfinite`). ``valid`` is 0 for an idle slot, 1 for a lattice
        particle and 2 for a particle created by adaptive seeding. A coupled solver may deactivate particles
        (``valid = 0``) or change their values between steps; every slot is reseeded at the next
        long reinitialisation.
    t, step_index:
        Simulation time and number of steps taken.
    reinit_long, reinit_short:
        Reinitialisation intervals; may be changed between steps.

    Example
    -------
    >>> ls = PFMLevelSet2D(res=200, ppc=16, band=2.0, reinit_long=12, reinit_short=6)
    >>> ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.75), 0.15))
    >>> for _ in range(100):
    ...     ls.step(0.02, pfmls.flows.LeVequeVortex2D())
    >>> area = ls.area()
    """

    def __init__(self, res: int | tuple[int, int] = 200, ppc: int = 16, band: float = 2.0,
                 reinit_long: float = 12, reinit_short: float = 6, *, reinit_unit: str = "steps",
                 dx: float | None = None, device=None, use_graph: bool = True,
                 options: PFMLS2DOptions | None = None, **overrides):
        nx, ny = (res, res) if isinstance(res, int) else (int(res[0]), int(res[1]))
        self.grid = Grid2D(nx, ny, float(dx) if dx is not None else 1.0 / nx)
        unknown = _opt.unknown_keys(PFMLS2DOptions, overrides)
        if unknown:
            raise TypeError(f"unknown option(s) {unknown} (see PFMLS2DOptions)")
        self._options = dataclasses.replace(options or PFMLS2DOptions(), **overrides)
        if self._options.seed_min is None:  # default: the paper's n_min = ppc / 2
            self._options = dataclasses.replace(self._options, seed_min=int(ppc) // 2)
        o = self._options
        a = int(round(math.sqrt(ppc)))
        if a * a != ppc or a < 1:
            raise ValueError(f"ppc must be a perfect square, got {ppc}")
        if a < 3:
            raise ValueError(f"ppc={ppc}: need ppc >= 9; with fewer particles per cell the deformed band stops "
                             f"covering the cell centres (LeVeque full method at ppc=1 loses about 22% of the area)")
        nearest = 0.0 if a % 2 == 1 else math.sqrt(2.0) * 0.5 / a  # lattice point nearest a cell centre (dx)
        if nearest > o.p2g_radius:
            raise ValueError(f"ppc={ppc}: no lattice point lies within p2g_radius={o.p2g_radius} dx of a cell "
                             f"centre (nearest {nearest:.3f} dx), so particles would never reach the grid; "
                             f"use an odd sqrt(ppc) (9, 25, ...) or a larger p2g_radius")
        self._init_base(band, reinit_long, reinit_short, reinit_unit, device, use_graph,
                        f"bands wider than {self.WIDE_BAND:g} dx lose area in deforming flows (LeVeque full "
                        f"method: 99.65% at 2 dx, 96.8% at 6 dx); 1-3 dx is the validated range (see docs/method.md)")
        self.ppc, self.a = int(ppc), a
        self._stage = 0.0 if o.frozen_time else 1.0
        self._grid_eik_iters = nx + ny if o.grid_eikonal_iterations is None else int(o.grid_eikonal_iterations)
        self._far = float(nx + ny) * self.grid.dx  # "outside" value for a non-finite grid cell
        self._cells_stale = False  # the cell lists miss the newborns of the last step (see _fresh_cells)
        self._allocate()

    @property
    def lattice_side(self) -> int:
        """Particles per cell along each axis (``ppc = lattice_side^2``; also available as ``a``)."""
        return self.a

    # ------------------------------------------------------------------ setup

    def _allocate(self) -> None:
        g, dev, o = self.grid, self.device, self._options
        self.nax, self.nay = self.a * g.nx, self.a * g.ny
        self.num_slots = self.nax * self.nay
        n = self.num_slots
        z = lambda dtype: wp.zeros(n, dtype=dtype, device=dev)  # noqa: E731
        self.x, self.phi, self.g, self.g0 = z(wp.vec2), z(float), z(wp.vec2), z(wp.vec2)
        self.valid = z(wp.int32)
        self.Js, self.Jinit, self.C, self.C0 = z(wp.mat22), z(wp.mat22), z(wp.mat22), z(wp.mat22)
        self._x_old, self._phi_old, self._g_old = z(wp.vec2), z(float), z(wp.vec2)

        shape = g.shape
        zg = lambda dtype, s=shape: wp.zeros(s, dtype=dtype, device=dev)  # noqa: E731
        self.phi_g = zg(float)        # background grid level set (composite)
        self._phi_s = zg(float)       # advected + redistanced grid before the particle composite
        self._phi_tmp = zg(float)
        self._phi_r, self._g_r, self._mask = zg(float), zg(wp.vec2), zg(wp.int32)
        self._phi_eik, self._g_eik = zg(float), zg(wp.vec2)
        self._face_x = zg(float, (g.nx + 1, g.ny))
        self._face_y = zg(float, (g.nx, g.ny + 1))
        self._clock = wp.zeros(2, dtype=float, device=dev)
        self._scrub_counter = wp.zeros(2, dtype=wp.int32, device=dev)
        self._params = wp.zeros(1, dtype=wp.vec4, device=dev)
        self._istep = wp.zeros(1, dtype=wp.int32, device=dev)
        if self._seeding:
            # free-slot list, the lattice particles' valid (the reinitialisations' view) and per-cell candidates
            self._free, self._slot_flags, self._slot_offsets, self._valid_lattice = (z(wp.int32) for _ in range(4))
            self._free_count, self._n_new, self._dropped = (wp.zeros(1, dtype=wp.int32, device=dev) for _ in range(3))
            zc = lambda dtype: wp.zeros(g.nx * g.ny, dtype=dtype, device=dev)  # noqa: E731
            self._seed_accept, self._seed_offsets = zc(wp.int32), zc(wp.int32)
            self._cand_x, self._cand_g, self._cand_phi, self._cand_C = zc(wp.vec2), zc(wp.vec2), zc(float), zc(wp.mat22)

        self.cells = CellList2D(g.nx, g.ny, g.dx, n, device=dev)
        self._eikonal = Eikonal2D(g.nx, g.ny, g.dx, device=dev, iterations=o.eikonal_iterations)
        self._redistancer = QuasiNewtonRedistancer(
            g.nx, g.ny, g.dx, device=dev, dilations=o.newton_dilations, linear_sweeps=o.sl_linear_sweeps,
            hermite_sweeps=o.sl_hermite_sweeps, sl_step=o.sl_step, max_iter=o.newton_max_iter,
            step_clamp=o.newton_step_clamp, tol=o.newton_tol)

        self._parts = self._view(self.x, self.phi, self.g)  # reinitialisations (they read the cell lists)
        self._parts_old = self._view(self._x_old, self._phi_old, self._g_old)
        self._parts_comp = self._view(self.x, self.phi, self.g, o.composite_newborns)
        self._parts_fine = self._view(self.x, self.phi, self.g, o.reconstruct_newborns)

    def _view(self, x, phi, g, newborns: str = "mixed") -> K.ParticleView:
        """Particle view for the reconstructions; without seeding every view is "mixed" (``valid`` never read)."""
        v = K.ParticleView()
        v.x, v.phi, v.g, v.C, v.valid = x, phi, g, self.C, self.valid
        v.radius = self._options.p2g_radius * self.grid.dx
        v.newborns = NEWBORN_MODES[newborns if self._seeding else "mixed"]
        return v

    @property
    def _seeding(self) -> bool:
        return self._options.seed_min > 0

    def init_from_sdf(self, sdf) -> None:
        """Initialise from a signed distance (``phi < 0`` inside).

        ``sdf`` is either a callable ``phi(x, y)`` on numpy arrays (e.g. from :mod:`pfmls.levelset.sdf`),
        evaluated exactly at cell centres and particles, or an ``(nx, ny)`` array of cell-centre
        values (particles are then sampled bilinearly). The values must be finite, the callable must
        return an array of its inputs' shape and the level set must change sign inside the grid
        (``ValueError`` otherwise; the state is unchanged).
        """
        g = self.grid
        lattice = self._x_old  # scratch: the state is only written once the input has been checked
        self._launch(K.seed_lattice, self.num_slots, [lattice, self.nax, g.dx / self.a])
        if callable(sdf):
            phi_cells = checked_sdf(sdf, g.cell_centers(), "cell centres")
            check_sign_change(phi_cells)
            xp = lattice.numpy().astype(np.float64)
            phi_parts = checked_sdf(sdf, (xp[:, 0], xp[:, 1]), "particle positions")
            self.phi_g.assign(phi_cells)
            self.phi.assign(phi_parts)
        else:
            arr = checked_grid_sdf(sdf, g.shape)
            check_sign_change(arr)
            self.phi_g.assign(arr)
            self._launch(K.sample_grid_to_particles, self.num_slots, [self.phi_g, lattice, 1.0 / g.dx, self.phi])
        wp.copy(self.x, lattice)
        self.valid.fill_(1)
        self._launch(K.set_identity, self.num_slots, [self.Js])
        self._launch(K.set_identity, self.num_slots, [self.Jinit])
        self.C.zero_()
        self.C0.zero_()
        self.cells.build(self.x, self.valid)
        self._cells_stale = False
        self._launch(K.shepard_gradient, self.num_slots, [self.x, self.phi, self.valid, self.cells.view, g.dx, self.g])
        wp.copy(self.g0, self.g)
        if self._seeding:
            self._dropped.zero_()

        self._reset_clock()
        self._launch(K.set_step_constants, 1, [self._clock, 0.0, 0.0, self._params, wp.vec4(), self._istep, 0])
        self._step_body(True, True, None, redistance=self._options.redistance_at_init)
        self._initialized = True

    # ------------------------------------------------------------------ stepping

    def step(self, dt: float, velocity, skip_long_reinit: bool = False) -> None:
        """Advance by ``dt`` in ``velocity`` (a :class:`pfmls.flows.Flow2D`).

        ``skip_long_reinit`` drops a long reinitialisation that is due at this step (the schedule
        continues as if it had run; a short reinitialisation still runs when due).

        Raises ``ValueError`` (before any work is launched) if ``dt`` is not a finite number >= 0, or
        if the velocity does not match the grid or lives on another device.
        """
        long, short, i, t1 = self._begin_step(dt, velocity, skip_long_reinit)
        self._launch(K.set_step_constants, 1, [self._clock, float(self.t), float(dt), self._params,
                                               wp.vec4(*velocity.params()), self._istep, int(i)])
        ux, uy = velocity.mac_arrays(self.device)
        self._graphs.run((long, short, velocity.func, ux.ptr, uy.ptr), lambda: self._step_body(long, short, velocity))
        self._cells_stale = self._seeding  # also after a graph replay (no host code runs)
        self.t = t1
        self.step_index = i

    def _fresh_cells(self) -> None:
        """Bin the newborns of the last step for the output (seeding ends a step without a rebuild; the next
        step's advection rebuilds the lists anyway)."""
        if self._cells_stale:
            self.cells.build(self.x, self.valid)
            self._cells_stale = False

    def _reinit_valid(self) -> wp.array:
        """The particles the reinitialisations read, binned into the cell lists: with seeding and the composite
        default "off" only the lattice particles (the newborns stay out of the dynamics), otherwise all."""
        if not self._seeding:
            return self.valid  # (the lists are current: every step ends with a rebuild)
        valid = self.valid
        if self._options.composite_newborns == "off":
            self._launch(K.lattice_only, self.num_slots, [self.valid, self._valid_lattice])
            valid = self._valid_lattice
        self.cells.build(self.x, valid)
        return valid

    def _step_body(self, long: bool, short: bool, velocity, redistance: bool = True) -> None:
        g, n = self.grid, self.num_slots
        valid = self._reinit_valid() if long or short else self.valid
        if long:
            self._long_reinit(redistance, valid)
            valid = self.valid  # (the reseed frees the newborns)
        if short:
            # after a long reinitialisation with Shepard gradients, g is already the Shepard gradient
            # of the current particles
            self._short_reinit(valid, shepard=not (long and self._options.shepard_at_long_reinit))
        maps = [self.Jinit, self.C0, self.g0, self.C, self.g]
        if velocity is not None:  # RK4 of x and Js, then C = Js^T C0 Js and g = normalize(J_long^T g0)
            rk4, sl = K.flow_kernels(velocity.func)
            ux, uy = velocity.mac_arrays(self.device)
            self._launch(rk4, n, [self.x, self.Js, self.valid, self._clock, self._params, ux, uy, 1.0 / g.dx,
                                  self._stage, *maps])
        else:
            self._launch(K.update_particle_maps, n, [self.valid, self.Js, *maps])
        self.cells.build(self.x, self.valid)
        if velocity is not None:
            self._launch(sl, g.shape, [self.phi_g, self._phi_tmp, self._clock, self._params, ux, uy, g.dx,
                                       self._stage])
        else:
            wp.copy(self._phi_tmp, self.phi_g)
        self._eikonal.redistance(self._phi_tmp, self._phi_s, self._grid_eik_iters)
        self._launch(K.composite_grid, g.shape,
                     [self._parts_comp, self.cells.view, g.dx, self._w_inv_h(g.dx / self._options.composite_factor),
                      self._phi_s, self._far, self.phi_g])
        if self._seeding and velocity is not None:
            self._adaptive_seed()

    def _adaptive_seed(self) -> None:
        """One candidate per under-populated band cell (:func:`particles2d.seed_evaluate`), prefix sum of the
        accepted candidates and storage in free slots (deterministic; the cell lists become stale)."""
        g, o = self.grid, self._options
        self._launch(K.seed_evaluate, g.shape,
                     [self._parts, self.cells.view, self.Jinit, self.Js, self.g0, self.phi_g, self._istep, g.dx,
                      self._band * g.dx, o.seed_min, o.seed_coherence, self._seed_accept, self._cand_x,
                      self._cand_phi, self._cand_g, self._cand_C])
        wp.utils.array_scan(self._seed_accept, self._seed_offsets, inclusive=False)
        self._launch(K.seed_store, g.nx * g.ny,
                     [self._seed_accept, self._seed_offsets, self._cand_x, self._cand_phi, self._cand_g, self._cand_C,
                      self._free, self._free_count, self._n_new, self._dropped, self.x, self.phi, self.g, self.g0,
                      self.C, self.C0, self.Js, self.Jinit, self.valid])
        self._launch(K.seed_advance, 1, [self._seed_accept, self._seed_offsets, self._n_new, self._free_count])
        self._cells_stale = True

    def _list_free_slots(self) -> None:
        """List the idle slots that seeding fills (at every long reinitialisation, which frees the newborns)."""
        self._launch(K.mark_idle, self.num_slots, [self.valid, self._slot_flags])
        wp.utils.array_scan(self._slot_flags, self._slot_offsets, inclusive=False)
        self._launch(K.build_free_list, self.num_slots, [self._slot_flags, self._slot_offsets, self._free,
                                                         self._free_count, self._n_new])

    def _particle_to_grid(self) -> None:
        g = self.grid
        self._launch(K.p2g_cells, g.shape, [self._parts, self.cells.view, g.dx, self._w_inv_h(g.dx),
                                            self._phi_r, self._g_r, self._mask])
        self._launch(_fill_scalar, g.shape, [self._mask, self.phi_g, self._phi_r])

    def _eikonal_guess(self) -> None:
        g = self.grid
        self._eikonal.redistance(self._phi_r, self._phi_eik)
        self._launch(_one_sided_gradient, g.shape, [self._phi_eik, self._g_eik, 1.0 / g.dx])

    def _long_reinit(self, redistance: bool, valid: wp.array) -> None:
        """Long reinitialisation of the particles ``valid`` (binned in the cell lists, see :meth:`_reinit_valid`).
        The reseed overwrites every slot, newborns included."""
        g, n, o = self.grid, self.num_slots, self._options
        cells = self.cells.view
        if o.shepard_at_long_reinit:
            self._launch(K.shepard_gradient, n, [self.x, self.phi, valid, cells, g.dx, self.g])
        self._particle_to_grid()
        if redistance:
            self._eikonal_guess()
            self._launch(_fill_scalar, g.shape, [self._mask, self._phi_eik, self._phi_r])
            self._launch(_fill_vector, g.shape, [self._mask, self._g_eik, self._g_r])
            self._redistancer.run(self._phi_r, self._g_r, self._mask, self._phi_eik, self._parts, cells)
        wp.copy(self._x_old, self.x)
        wp.copy(self._phi_old, self.phi)
        wp.copy(self._g_old, self.g)
        self._launch(K.reseed, n, [self._parts_old, cells, self._phi_r, self._g_r, self.nax, g.dx, self.a,
                                   self._band * g.dx, o.blend_e1, o.blend_e2, o.blend_gate, o.blend_grid_gate,
                                   self.x, self.phi, self.g, self.g0, self.valid, self.Js, self.Jinit])
        self.C.zero_()
        self.cells.build(self.x, self.valid)
        if o.shepard_at_long_reinit:
            self._launch(K.shepard_gradient, n, [self.x, self.phi, self.valid, cells, g.dx, self.g])
            wp.copy(self.g0, self.g)
        if self._seeding:
            self._list_free_slots()

    def _short_reinit(self, valid: wp.array, shepard: bool = True) -> None:
        """Short reinitialisation: ``C0`` of the particles ``valid``, then the short-map reset of every particle.
        Newborns not among ``valid`` take their transported ``C`` as ``C0`` (it continues across the reset)."""
        g, n = self.grid, self.num_slots
        if shepard:
            self._launch(K.shepard_gradient, n, [self.x, self.phi, valid, self.cells.view, g.dx, self.g])
        self._particle_to_grid()
        self._eikonal_guess()
        self._launch(_fill_vector, g.shape, [self._mask, self._g_eik, self._g_r])
        self._launch(K.split_to_faces, (g.nx + 1, g.ny + 1), [self._g_r, self._face_x, self._face_y])
        self._launch(K.normal_gradient_from_faces, n, [self.x, valid, self._face_x, self._face_y, 1.0 / g.dx,
                                                       self._options.hessian_scale, self.C0])
        if valid is not self.valid:
            self._launch(K.keep_newborn_curvature, n, [self.valid, valid, self.C, self.C0])
        self._launch(K.short_map_reset, n, [self.valid, self.Js, self.Jinit])

    def scrub_nonfinite(self, counter: wp.array | None = None) -> None:
        """Repair non-finite state (e.g. after a float overflow upstream), without host synchronisation.

        Particles with a non-finite position, value or gradient are deactivated until the next long
        reinitialisation; non-finite grid values become "outside" (``(nx + ny) dx``). If given, the
        int32 array ``counter`` (at least 2 entries) accumulates the number of repaired grid cells
        (entry 0) and particles (entry 1). A no-op for a finite state.
        """
        counter = self._scrub_target(counter)
        g = self.grid
        self._launch(K.scrub_nonfinite_grid, g.shape, [self.phi_g, self._far, counter, 0])
        self._launch(K.scrub_nonfinite_grid, g.shape, [self._phi_s, self._far, counter, 0])
        self._launch(K.scrub_nonfinite_particles, self.num_slots, [self.x, self.phi, self.g, self.valid, counter, 1])

    # ------------------------------------------------------------------ state

    def _state_names(self) -> tuple[str, ...]:
        return STATE_ARRAYS + (SEEDING_ARRAYS if self._seeding else ())

    def _check_state(self, arrays: dict, what: str) -> None:
        """Raise ``ValueError`` unless ``arrays`` (a snapshot or a state dict) match this instance's
        resolution, ppc, dx and seeding setting (a seeded state holds the free-slot list, an unseeded one not)."""
        seeded = [n for n in SEEDING_ARRAYS if (n if what == "snapshot" else n.lstrip("_")) in arrays]
        if self._seeding != bool(seeded):
            raise ValueError(f"{what} comes from a level set {'with' if seeded else 'without'} adaptive seeding, "
                             f"this one has seed_min={self._options.seed_min}")
        for name in self._state_names():
            mine = getattr(self, name)
            key = name if what == "snapshot" else name.lstrip("_")
            if key not in arrays:
                raise ValueError(f"{what} has no entry {key!r}")
            a = arrays[key]
            if what == "snapshot":
                ok = tuple(a.shape) == tuple(mine.shape) and a.dtype == mine.dtype
                want = tuple(mine.shape)
            else:
                want = tuple(mine.shape) + tuple(getattr(mine.dtype, "_shape_", ()))
                ok = tuple(np.shape(a)) == want
            if not ok:
                raise ValueError(f"{what} entry {key!r} has shape {tuple(np.shape(a))}, expected {want} "
                                 f"(res {self.grid.nx}x{self.grid.ny}, ppc {self.ppc})")
        dx = arrays.get("dx")
        if dx is not None and abs(float(dx) - self.grid.dx) > 1.0e-6 * self.grid.dx:
            raise ValueError(f"{what} has dx={float(dx)}, this level set has dx={self.grid.dx}")

    def snapshot(self, out: dict | None = None) -> dict:
        """Device-side copy of the full state (particles, maps, grids, clock) for a later :meth:`restore`.

        Passing a previous snapshot as ``out`` reuses its buffers (no allocation).
        """
        snap = out if out is not None else {}
        for name in self._state_names():
            src = getattr(self, name)
            dst = snap.get(name)
            if dst is None or dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device:
                dst = snap[name] = wp.empty_like(src)
            wp.copy(dst, src)
        snap.update(self._clock_scalars())
        return snap

    def restore(self, snap: dict) -> None:
        """Return to the state captured by :meth:`snapshot` (captured CUDA graphs stay valid).

        Raises ``ValueError`` if the snapshot comes from a level set with another resolution, ppc or seeding
        setting (``seed_min`` zero or not).
        """
        self._check_state(snap, "snapshot")
        for name in self._state_names():
            wp.copy(getattr(self, name), snap[name])
        self._set_clock(snap)
        self.cells.build(self.x, self.valid)
        self._cells_stale = False

    def state_dict(self) -> dict:
        """Host copy of the full state (numpy arrays and scalars), e.g. for checkpoints."""
        d = {name.lstrip("_"): self._host(getattr(self, name)) for name in self._state_names()}
        d.update(self._clock_scalars())
        return d

    def load_state_dict(self, state: dict) -> None:
        """Load a state written by :meth:`state_dict` (same resolution, ppc, dx and seeding setting, else
        ``ValueError``)."""
        self._check_state(state, "state_dict")
        for name in self._state_names():
            getattr(self, name).assign(np.asarray(state[name.lstrip("_")]))
        self._set_clock(state)
        self.cells.build(self.x, self.valid)
        self._cells_stale = False

    # ------------------------------------------------------------------ output

    def reconstruct_device(self, factor: int = 5, newborns: str | None = None) -> wp.array:
        """Fine ``(factor nx, factor ny)`` reconstruction on the device (particles where a particle lies
        within ``p2g_radius dx`` of a fine cell centre, bilinear grid elsewhere). ``newborns`` ("off",
        "fallback", "mixed"; default ``reconstruct_newborns``) sets how the particles created by adaptive
        seeding count. The buffer is cached per factor."""
        g = self.grid
        parts = self._parts_fine
        if newborns is not None:
            if newborns not in NEWBORN_MODES:
                raise ValueError(f"newborns must be one of {sorted(NEWBORN_MODES)}, got {newborns!r}")
            parts = self._view(self.x, self.phi, self.g, newborns)
        out = self._fine.get(factor)
        if out is None:
            out = self._fine[factor] = wp.zeros((g.nx * factor, g.ny * factor), dtype=float, device=self.device)
        self._fresh_cells()
        self._launch(K.reconstruct_fine, out.shape, [parts, self.cells.view, g.dx / factor, 1.0 / g.dx,
                                                     self._w_inv_h(g.dx / factor), self._phi_s, out])
        return out

    def reconstruct(self, factor: int = 5, newborns: str | None = None) -> np.ndarray:
        """Fine reconstruction as a numpy array (a copy; see :meth:`reconstruct_device`)."""
        return self._host(self.reconstruct_device(factor, newborns))

    def release_fine(self) -> None:
        """Free the cached fine-reconstruction buffers."""
        self._fine.clear()

    def area(self, factor: int = 5, newborns: str | None = None) -> float:
        """Area of ``{phi < 0}`` from the zero contours of the fine reconstruction.

        Contours are closed along the domain walls, so liquid touching the walls is measured too.
        """
        return measure.contour_area(self.reconstruct(factor, newborns), self.grid.dx / factor)

    def particles(self) -> dict:
        """Valid particles as numpy arrays: ``x`` (n, 2), ``phi`` (n,), ``g`` (n, 2) and ``newborn`` (n,), true
        for particles created by adaptive seeding."""
        valid = self.valid.numpy()
        v = valid != 0
        return {"x": self.x.numpy()[v], "phi": self.phi.numpy()[v], "g": self.g.numpy()[v],
                "newborn": valid[v] == 2}

    def num_valid(self) -> int:
        """Number of active particles (band particles and newborns)."""
        return int(np.count_nonzero(self.valid.numpy()))

    @property
    def dropped_newborns(self) -> int:
        """Newborns that adaptive seeding could not store since :meth:`init_from_sdf` because no slot was idle
        (a host read)."""
        return int(self._dropped.numpy()[0]) if self._seeding else 0
