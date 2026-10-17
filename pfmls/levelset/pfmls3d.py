"""PFMLevelSet3D: the 3D Particle Flow Map Level Set method.

Same pipeline as :class:`pfmls.levelset.PFMLevelSet2D`, with narrow-band particle storage.

State
-----
* particles on an ``a x a x a`` lattice (``ppc = a^3``) in the *seed cells* (cells within
  ``band + lattice radius`` of the interface) chosen at each long reinitialisation; ``valid`` marks the
  narrow band ``|phi| < band dx``. Storage is sized by the band (``capacity`` slots, grown on demand),
  never by ``res^3 ppc``;
* the background grid level set ``phi_g`` (cell centres).

One step (``i`` = step counter)
-------------------------------
A. long reinitialisation (when due): Shepard gradients of the particles it reads, particle-to-grid
   reconstruction (cells with a particle within ``p2g_radius dx``), fill the rest from the grid, eikonal
   redistancing + quasi-Newton
   closest-point projection in a band around the interface (:mod:`.newton3d`), compact the seed cells
   (prefix sum; one host read of their count, used to grow the storage), then reseed every lattice slot
   of the seed cells from the tricubic Hermite grid interpolant blended with the old particles, and reset
   the long map;
C. short reinitialisation (when due, and always together with A): ``C0 = hessian_scale grad n`` from a
   least-squares fit of the particle normals (``normal_gradient="particles"``; ``"faces"`` = the 2D
   path: Shepard gradients, particle-to-grid normals, quadratic B-spline derivative of the
   MAC-averaged normals), reset the short map;
D. RK4 advection of positions and of the short-map Jacobian ``dJ/dt = -J grad u``, followed in the same
   kernel by F. ``C = Js^T C0 Js`` and ``g = normalize(J_long^T g0)``; E. re-bin;
G. semi-Lagrangian (RK2) advection + eikonal redistancing of the background grid;
I. composite: grid cells with a lattice particle within ``composite_radius dx`` take the particle value;
S. optional adaptive seeding: one new particle per band cell holding fewer than ``seed_min`` particles (by
   default these *newborns* only fill the holes of the output reconstruction, see ``composite_newborns``);
K. every ``compact_every`` steps (not on long steps): slot compaction, which reorders the live particles
   into cell order (memory locality), drops the newborns far from the interface and frees every other slot.

``phi_p`` is constant between long reinitialisations. On CUDA each step type is captured into a CUDA
graph (a long step is two graphs around the storage check) and replayed.

Protocols
---------
``reinit_long = 0`` runs the long reinitialisation only at ``t = 0``: the long map spans the whole run and
only the short map is reset. With ``reinit_short = 5`` and ``seed_min = 16`` this is the ``"research"``
protocol of the 3D benchmark. ``reinit_long = reinit_short = 0`` is
pure advection. The full method (``12 / 6``) redistances and reseeds every ``reinit_long`` steps.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..core import measure3d
from ..core import options as _opt
from ..core.celllist3d import CellList3D
from ..core.eikonal3d import INIT_MODES, Eikonal3D
from ..core.grid3d import Grid3D
from ..core.kernels3d import hermite_cross_terms, safe_normalize3
from ..flows.analytic3d import FlowParams3D
from . import particles3d as K
from ._base import PFMLevelSetBase, check_sign_change, checked_grid_sdf, checked_sdf
from .newton3d import QuasiNewtonRedistancer3D

#: Ways of computing the normal gradient ``C0`` at short reinitialisation (see :class:`PFMLS3DOptions`).
NORMAL_GRADIENTS = ("faces", "particles")

#: Named reinitialisation protocols of the 3D benchmarks: ``(reinit_long, reinit_short, seed_min)``.
PROTOCOLS = {
    "full": (12, 6, 0),       # long reinitialisation with redistancing + reseeding every 12 steps
    "research": (0, 5, 16),   # long map from t = 0, short reinit every 5 steps, adaptive seeding
    "advection": (0, 0, 0),   # pure advection: both reinitialisations only at t = 0
}

#: Modes of ``composite_newborns`` / ``reconstruct_newborns`` (see :class:`PFMLS3DOptions`).
NEWBORN_MODES = {"mixed": K.NEWBORNS_MIXED, "off": K.NEWBORNS_OFF, "fallback": K.NEWBORNS_FALLBACK}

#: Storage constants: slot counts are multiples of SLOT_BLOCK ppc; with adaptive seeding the storage grows when more
#: than GROWTH_FILL of it is used; the seeding buffers hold at least MIN_SEED_CANDIDATES candidates; the Shepard
#: gradients of a long reinitialisation cover the particles within (blend_gate + p2g_radius + SHEPARD_MARGIN) dx
#: of the interface (those the reseed and the Newton interpolation read)
SLOT_BLOCK = 64
GROWTH_FILL = 0.8
MIN_SEED_CANDIDATES = 256
SHEPARD_MARGIN = 0.25


@dataclass(frozen=True)
class PFMLS3DOptions:
    """Advanced parameters (immutable).

    The defaults follow the 2D implementation (:class:`pfmls.levelset.PFMLS2DOptions`); 3D-specific
    fields are marked. Values are type-checked (``TypeError``); :func:`parse_options` converts command-line
    ``KEY=VALUE`` strings. The protocols and the 3D options are described in
    ``docs/method.md``, section 4.
    """

    # --- reconstruction and extrapolation ---
    #: radius (in dx) of the particle -> point reconstructions of the reinitialisations (cell-centre P2G,
    #: reseed and Newton KNN), in (0, 0.5). None = the lattice spacing 1 / a (0.25 for 64 ppc, 1/3 for 27 ppc,
    #: 0.2 for 125 ppc). Neighbours at exactly the radius are excluded (1e-3 relative margin)
    p2g_radius: float | None = None
    #: radius (in dx) of the per-step composite (grid cells with a particle within it take the particle value)
    #: and of the fine reconstruction (mesh, volume, rendering; other points take the trilinear background
    #: grid); 3D; in (0, 0.5) and not below the p2g radius
    composite_radius: float = 0.4
    reconstruct_radius: float = 0.4
    #: how the particles created by adaptive seeding (newborns) enter the per-step composite and the fine
    #: reconstruction (3D): "off" = not at all, "fallback" = only at points that no lattice particle covers,
    #: "mixed" = averaged with the lattice particles. With the composite default "off" the
    #: newborns never enter the dynamics (the evolved state equals that of an unseeded run); otherwise the
    #: reinitialisations read them like lattice particles
    composite_newborns: str = "off"
    reconstruct_newborns: str = "fallback"
    #: reconstruction weights: "uniform" (mean of the particles within the radius) or "bspline"
    p2g_weights: str = "uniform"
    #: scale s of the normal gradient C0 = s grad n in the extrapolation phi_p + normalize(g_p + C_p d) . d:
    #: 0.5 = Taylor-consistent (3D default), 1 = ``C0 = grad n`` (2D default; biases convex shapes inwards at every long
    #: reinitialisation)
    hessian_scale: float = 0.5
    #: how C0 = grad n is computed at short reinitialisation (3D): "particles" = least-squares gradient of the
    #: particle normals within normal_gradient_radius dx (unbiased for small radii, noisy for large ones);
    #: "faces" = quadratic B-spline derivative of the MAC-averaged particle-to-grid normals (as in
    #: 2D; underestimates the curvature of radii below about 3 dx)
    normal_gradient: str = "particles"
    normal_gradient_radius: float = 0.5
    #: reseed blend: particle value if n_H . n_P > e1, linear blend down to e2, grid value below
    blend_e1: float = 0.9
    blend_e2: float = 0.7
    #: particle value only if |phi_P| < blend_gate dx and |phi_H| <= blend_grid_gate dx
    blend_gate: float = 0.25
    blend_grid_gate: float = 1.0
    #: quasi-Newton band: interface cells dilated newton_dilations times plus every cell with eikonal
    #: distance below newton_band dx (3D; must cover the Hermite stencils of the reseeded band)
    newton_dilations: int = 3
    newton_band: float = 4.0
    #: quasi-Newton iteration cap, trust radius (dx: initial and minimum newton_step_clamp, doubled while the
    #: steps keep their direction, at most newton_max_step; 0 = every step clamped to newton_step_clamp)
    #: and tolerance (dx). The cap sets the cost
    newton_max_iter: int = 16
    newton_step_clamp: float = 0.02
    newton_max_step: float = 1.0
    newton_tol: float = 1.0e-4
    #: trilinear semi-Lagrangian eikonal sweeps before the Newton step, limited to sweep_width dx (3D)
    sl_linear_sweeps: int = 5
    sl_step: float = 0.9
    sweep_width: float = 6.0
    #: Jacobi iterations of the eikonal initial guess at reinitialisation (None = whole grid)
    eikonal_iterations: int | None = 48
    #: Jacobi iterations of the per-step background-grid redistancing (None = whole grid)
    grid_eikonal_iterations: int | None = 16
    #: interface initialisation of the eikonal solver: "gradient" (phi / |grad phi|, 3D default) or
    #: "crossing" (smallest axis crossing, as in 2D; overestimates oblique distances)
    eikonal_init: str = "gradient"
    #: compute C0 from the initial particles before the initial long reinitialisation, so that the initial
    #: reconstruction and reseeding use the second-order extrapolation (False: C = 0 at t = 0)
    init_normal_gradient: bool = True

    # --- adaptive seeding and storage (3D) ---
    #: adaptive seeding: every step, one particle is added to each band cell (|phi_g| < band dx) holding
    #: fewer than seed_min particles (0 = off; the research protocol uses 16 for 64 ppc)
    seed_min: int = 0
    #: a new particle is rejected if the weighted mean of its neighbours' unit normals is shorter than
    #: seed_coherence, or if its sign differs from the background grid (0 = no checks)
    seed_coherence: float = 0.9
    #: slot compaction every compact_every steps (0 = never; skipped on long-reinitialisation steps, which
    #: rebuild the storage): the live particles are reordered into cell order and the other slots freed
    compact_every: int = 20
    #: at compaction, drop the newborns further than cull_distance dx from the background-grid interface
    #: (None = band + 2, 0 = keep all); lattice particles are never culled
    cull_distance: float | None = None
    #: slot capacity = capacity_factor x the slots needed at initialisation
    capacity_factor: float = 2.0
    #: when the slots needed exceed the capacity (checked at each long reinitialisation, and every
    #: capacity_check_every steps while seeding), grow to growth_factor x the need
    growth_factor: float = 1.5
    capacity_check_every: int = 10
    #: extra margin (dx) of the seed-cell test |phi(cell centre)| < band + lattice radius + margin
    seed_margin: float = 0.25

    # --- alternative variants (the defaults are the recommended settings) ---
    #: velocity frozen at the step's start time in every RK stage (first order in time)
    frozen_time: bool = False
    #: recompute particle gradients by Shepard interpolation at long reinitialisation
    shepard_at_long_reinit: bool = True
    #: run the grid redistancing at the initial long reinitialisation
    redistance_at_init: bool = True
    #: refinement of the composite grid of the "bspline" weights (h = dx / composite_factor)
    composite_factor: int = 4

    def __post_init__(self):
        _opt.check_field_types(self)
        for name in ("composite_newborns", "reconstruct_newborns"):
            if getattr(self, name) not in NEWBORN_MODES:
                raise ValueError(f"{name} must be one of {sorted(NEWBORN_MODES)}, got {getattr(self, name)!r}")
        if self.p2g_weights not in ("uniform", "bspline"):
            raise ValueError(f"p2g_weights must be 'uniform' or 'bspline', got {self.p2g_weights!r}")
        if self.p2g_radius is not None and not 0.0 < self.p2g_radius < 0.5:
            raise ValueError(f"p2g_radius must lie in (0, 0.5) (dx) or be None, got {self.p2g_radius}")
        for name in ("composite_radius", "reconstruct_radius"):
            if not 0.0 < getattr(self, name) < 0.5:
                raise ValueError(f"{name} must lie in (0, 0.5) (dx), got {getattr(self, name)}")
        if self.hessian_scale < 0.0:
            raise ValueError(f"hessian_scale must be >= 0, got {self.hessian_scale}")
        if self.normal_gradient not in NORMAL_GRADIENTS:
            raise ValueError(f"normal_gradient must be one of {NORMAL_GRADIENTS}, got {self.normal_gradient!r}")
        if not 0.0 < self.normal_gradient_radius <= 1.0:
            raise ValueError(f"normal_gradient_radius must lie in (0, 1] (dx), got {self.normal_gradient_radius}")
        if not self.blend_e1 > self.blend_e2:
            raise ValueError(f"need blend_e1 > blend_e2, got {self.blend_e1} <= {self.blend_e2}")
        if self.composite_factor < 1:
            raise ValueError(f"composite_factor must be >= 1, got {self.composite_factor}")
        if self.eikonal_init not in INIT_MODES:
            raise ValueError(f"eikonal_init must be one of {sorted(INIT_MODES)}, got {self.eikonal_init!r}")
        for name in ("newton_dilations", "newton_max_iter", "sl_linear_sweeps", "seed_min", "compact_every"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.cull_distance is not None and self.cull_distance < 0.0:
            raise ValueError("cull_distance must be >= 0 or None")
        if self.newton_step_clamp <= 0.0 or self.newton_max_step < 0.0:
            raise ValueError("need newton_step_clamp > 0 and newton_max_step >= 0")
        for name in ("eikonal_iterations", "grid_eikonal_iterations"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"{name} must be >= 0 or None")
        if self.capacity_factor < 1.0 or self.growth_factor < 1.0:
            raise ValueError("capacity_factor and growth_factor must be >= 1")
        if self.capacity_check_every < 1:
            raise ValueError("capacity_check_every must be >= 1")


def parse_options(pairs) -> dict:
    """Typed :class:`PFMLS3DOptions` keyword arguments from ``"KEY=VALUE"`` strings (command lines).

    Values are converted by the field's annotation: bools accept ``1/0, true/false, yes/no, on/off``,
    ``none`` gives None where allowed. Raises ``ValueError`` for an unknown key or an unconvertible value.
    """
    return _opt.parse_options(PFMLS3DOptions, pairs)


@wp.kernel
def _fill_scalar3(m: wp.array3d(dtype=wp.int32), src: wp.array3d(dtype=float), dst: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    if m[i, j, k] == 0:
        dst[i, j, k] = src[i, j, k]


@wp.kernel
def _fill_vector3(m: wp.array3d(dtype=wp.int32), src: wp.array3d(dtype=wp.vec3), dst: wp.array3d(dtype=wp.vec3)):
    i, j, k = wp.tid()
    if m[i, j, k] == 0:
        dst[i, j, k] = src[i, j, k]


@wp.kernel
def _central_normal(phi: wp.array3d(dtype=float), g: wp.array3d(dtype=wp.vec3)):
    """Unit normal from central differences of phi (one-sided on the boundary)."""
    i, j, k = wp.tid()
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]
    ip = wp.min(i + 1, nx - 1)
    im = wp.max(i - 1, 0)
    jp = wp.min(j + 1, ny - 1)
    jm = wp.max(j - 1, 0)
    kp = wp.min(k + 1, nz - 1)
    km = wp.max(k - 1, 0)
    d = wp.vec3(0.0, 0.0, 0.0)
    if ip > im:
        d[0] = (phi[ip, j, k] - phi[im, j, k]) / float(ip - im)
    if jp > jm:
        d[1] = (phi[i, jp, k] - phi[i, jm, k]) / float(jp - jm)
    if kp > km:
        d[2] = (phi[i, j, kp] - phi[i, j, km]) / float(kp - km)
    g[i, j, k] = safe_normalize3(d)


#: Device arrays that define the level-set state (see :meth:`PFMLevelSet3D.snapshot`) and their dtypes.
SLOT_DTYPES = {"x": wp.vec3, "phi": wp.float32, "g": wp.vec3, "g0": wp.vec3, "valid": wp.int32, "Js": wp.mat33,
               "Jinit": wp.mat33, "C": wp.mat33, "C0": wp.mat33}
PARTICLE_ARRAYS = tuple(SLOT_DTYPES)
GRID_ARRAYS = ("phi_g", "_phi_s", "_n_used")
#: Per-slot scratch arrays (old particles at reseeding, compaction buffers, particle lists) and their dtypes.
SCRATCH_DTYPES = {"_x_old": wp.vec3, "_phi_old": wp.float32, "_g_old": wp.vec3, "_valid_old": wp.int32,
                  "_v_tmp": wp.vec3, "_flags": wp.int32, "_offsets": wp.int32, "_list": wp.int32}
SCRATCH_ARRAYS = tuple(SCRATCH_DTYPES)
#: Adaptive-seeding buffers (sized by the candidates per step, see ``_allocate_seeding``).
SEEDING_ARRAYS = ("_seed_list", "_seed_accept", "_seed_offsets", "_seed_pw", "_seed_pv", "_seed_pC")


class PFMLevelSet3D(PFMLevelSetBase):
    """Particle flow map level set on a 3D cell-centred grid.

    Parameters
    ----------
    res:
        Grid resolution ``n`` or ``(nx, ny, nz)``. The domain is ``[0, nx dx] x [0, ny dx] x [0, nz dx]``.
    ppc:
        Particles per cell, a perfect cube >= 27 such that a lattice point lies within ``p2g_radius dx``
        of every cell centre (27, 64, 125, ...). 64 is the benchmark value; 27 is coarser in strongly
        deforming flows (see ``docs/method.md``, section 4).
    band:
        Half-width (in dx) of the particle band chosen at each long reinitialisation. Bands wider than
        ``WIDE_BAND`` (4 dx) trigger a ``RuntimeWarning``.
    reinit_long, reinit_short:
        Long / short reinitialisation intervals (0 = never after initialisation).
    reinit_unit:
        ``"steps"`` (intervals are step counts) or ``"time"`` (time units).
    dx:
        Cell size (default ``1 / nx``).
    device:
        Warp device (``"cuda:0"``, ``"cpu"``, ...).
    use_graph:
        Capture each step type into a CUDA graph (ignored on CPU). Graphs are keyed on the step type and
        the velocity's face arrays, so a MAC velocity should be updated in place (or cycle through a few
        buffers) rather than reallocated every step.
    capacity:
        Initial number of particle slots (default: ``capacity_factor`` x the slots needed at
        initialisation). The storage grows automatically when the band needs more.
    options:
        A :class:`PFMLS3DOptions`; keyword ``overrides`` replace individual fields.

    Attributes
    ----------
    phi_g:
        Background grid level set ``(nx, ny, nz)`` on the device (``phi < 0`` inside; never reallocated). It
        may be edited between steps (e.g. ghost values at an open boundary): the edit enters the next step's
        semi-Lagrangian advection and the long reinitialisation's fill of cells without particles; the
        particles, the redistanced grid behind the composite and cached fine reconstructions do not see it.
    x, phi, g, valid:
        Particle slot arrays on the device (read-only; reallocated when the storage grows, so re-read them
        after :meth:`step`). ``valid`` is 0 for a free slot, 1 for a lattice particle and 2 for a particle
        created by adaptive seeding.
    t, step_index:
        Simulation time and number of steps taken.
    reinit_long, reinit_short:
        Reinitialisation intervals; may be changed between steps.

    Example
    -------
    >>> ls = PFMLevelSet3D(res=50, ppc=64, reinit_long=12, reinit_short=6)
    >>> ls.init_from_sdf(sdf3d.sphere((0.35, 0.35, 0.35), 0.15))
    >>> for _ in range(250):
    ...     ls.step(0.02, Enright3D())
    >>> vol = ls.volume()
    """

    #: cells per axis that the default fine reconstruction (mesh, volume) does not exceed
    FINE_CELLS = 512

    def __init__(self, res: int | tuple[int, int, int] = 50, ppc: int = 64, band: float = 2.0,
                 reinit_long: float = 12, reinit_short: float = 6, *, reinit_unit: str = "steps",
                 dx: float | None = None, device=None, use_graph: bool = True, capacity: int | None = None,
                 options: PFMLS3DOptions | None = None, **overrides):
        nx, ny, nz = (res, res, res) if isinstance(res, int) else (int(res[0]), int(res[1]), int(res[2]))
        self.grid = Grid3D(nx, ny, nz, float(dx) if dx is not None else 1.0 / nx)
        unknown = _opt.unknown_keys(PFMLS3DOptions, overrides)
        if unknown:
            raise TypeError(f"unknown option(s) {unknown} (see PFMLS3DOptions)")
        self._options = dataclasses.replace(options or PFMLS3DOptions(), **overrides)
        o = self._options
        a = int(round(ppc ** (1.0 / 3.0)))
        if a < 1 or a ** 3 != ppc:
            raise ValueError(f"ppc must be a perfect cube, got {ppc}")
        if a < 3:
            raise ValueError(f"ppc={ppc}: need ppc >= 27; with fewer particles per cell the deformed band stops "
                             f"covering the cell centres")
        #: radius (dx) of the reinitialisation reconstructions (``p2g_radius`` resolved)
        self.p2g_radius = 1.0 / a if o.p2g_radius is None else float(o.p2g_radius)
        for name in ("composite_radius", "reconstruct_radius"):
            if getattr(o, name) < self.p2g_radius:
                raise ValueError(f"{name}={getattr(o, name)} must be >= p2g_radius={self.p2g_radius:.3f}")
        nearest = 0.0 if a % 2 == 1 else math.sqrt(3.0) * 0.5 / a  # lattice point nearest a cell centre (dx)
        if nearest > self.p2g_radius:
            raise ValueError(f"ppc={ppc}: no lattice point lies within p2g_radius={self.p2g_radius} dx of a cell "
                             f"centre (nearest {nearest:.3f} dx), so particles would never reach the grid")
        self._init_base(band, reinit_long, reinit_short, reinit_unit, device, use_graph,
                        f"bands wider than {self.WIDE_BAND:g} dx are not validated in 3D (1-3 dx is the tested "
                        f"range, see docs/method.md, section 4)")
        if o.newton_band < band + 1.0:
            raise ValueError(f"newton_band={o.newton_band} must be >= band + 1 = {band + 1.0} (dx)")
        self.ppc, self.a = int(ppc), a
        self._stage = 0.0 if o.frozen_time else 1.0
        n_sum = nx + ny + nz
        self._grid_eik_iters = n_sum if o.grid_eikonal_iterations is None else int(o.grid_eikonal_iterations)
        self._far = float(n_sum) * self.grid.dx  # "outside" value for a non-finite grid cell
        self._x_max = wp.vec3(*((n - 1.0e-3) * self.grid.dx for n in (nx, ny, nz)))  # particle clamp
        # seed cells: every lattice point with |phi| < band lies in a cell whose centre is within
        # band + (lattice radius) + margin of the interface
        self._seed_threshold = (self._band + math.sqrt(3.0) * (0.5 - 0.5 / a) + o.seed_margin) * self.grid.dx

        self.capacity = 0
        self._requested_capacity = capacity
        self.grow_events = 0
        #: newborns that adaptive seeding could not store because the storage was full (they are retried)
        self.dropped_newborns = 0
        self._seed_capacity = 0  # candidates per step that the seeding buffers hold
        self._cells_stale = False  # the cell lists miss the newborns of the last step (see _fresh_cells)
        self._allocate_grid()

    @property
    def lattice_side(self) -> int:
        """Particles per cell along each axis (``ppc = lattice_side^3``; also available as ``a``)."""
        return self.a

    @property
    def num_slots(self) -> int:
        """Number of particle slots (``capacity``; the name of the 2D class)."""
        return self.capacity

    # ------------------------------------------------------------------ storage

    def _allocate_grid(self) -> None:
        g, dev, o = self.grid, self.device, self._options
        shape = g.shape
        zg = lambda dtype, s=shape: wp.zeros(s, dtype=dtype, device=dev)  # noqa: E731
        self.phi_g = zg(float)        # background grid level set (composite)
        self._phi_s = zg(float)       # advected + redistanced grid before the particle composite
        self._phi_tmp = zg(float)
        self._phi_r, self._g_r, self._mask = zg(float), zg(wp.vec3), zg(wp.int32)
        self._X_r = zg(wp.vec4)
        self._phi_eik, self._g_eik = zg(float), zg(wp.vec3)
        self._face_x = zg(float, (g.nx + 1, g.ny, g.nz))
        self._face_y = zg(float, (g.nx, g.ny + 1, g.nz))
        self._face_z = zg(float, (g.nx, g.ny, g.nz + 1))
        n = g.num_cells
        self._cell_flags = wp.zeros(n, dtype=wp.int32, device=dev)
        self._cell_offsets = wp.zeros(n, dtype=wp.int32, device=dev)
        self._seed_cells = wp.zeros(n, dtype=wp.int32, device=dev)
        self._seed_count = wp.zeros(1, dtype=wp.int32, device=dev)
        self._list_count = wp.zeros(1, dtype=wp.int32, device=dev)
        self._n_used = wp.zeros(1, dtype=wp.int32, device=dev)
        self._dropped = wp.zeros(1, dtype=wp.int32, device=dev)
        self._seed_overflow = wp.zeros(1, dtype=wp.int32, device=dev)
        self._counters = wp.zeros(3, dtype=wp.int32, device=dev)
        self._scrub_counter = wp.zeros(2, dtype=wp.int32, device=dev)
        self._clock = wp.zeros(2, dtype=float, device=dev)
        self._params = wp.zeros(1, dtype=FlowParams3D, device=dev)
        self._istep = wp.zeros(1, dtype=wp.int32, device=dev)

        self.cells = CellList3D(g.nx, g.ny, g.nz, g.dx, 1, device=dev)
        self._eikonal = Eikonal3D(g.nx, g.ny, g.nz, g.dx, device=dev, iterations=o.eikonal_iterations,
                                  init=o.eikonal_init)
        self._redistancer = QuasiNewtonRedistancer3D(
            g.nx, g.ny, g.nz, g.dx, device=dev, dilations=o.newton_dilations, band_width=o.newton_band,
            linear_sweeps=o.sl_linear_sweeps, sweep_width=o.sweep_width, sl_step=o.sl_step,
            max_iter=o.newton_max_iter, step_clamp=o.newton_step_clamp, max_step=o.newton_max_step,
            tol=o.newton_tol)

    def _round_capacity(self, capacity: int) -> int:
        """Slot counts are multiples of ``SLOT_BLOCK`` (64) ppc."""
        block = SLOT_BLOCK * self.ppc
        return int(math.ceil(max(int(capacity), 1) / block)) * block

    def _allocate_particles(self, capacity: int, keep: bool = True) -> None:
        """(Re)allocate every slot array for ``capacity`` slots, keeping the current particles (``keep``).

        A growth (``keep`` and more slots, :meth:`_grow_particles`) replaces the arrays one at a time, so that it
        needs about one slot array more than the new storage instead of the old and the new storage together.
        Otherwise the new arrays are allocated and filled before any attribute changes.  Both are exception-safe:
        a failed allocation (out of memory) leaves the level set as it was."""
        cap = self._round_capacity(capacity)
        if keep and 0 < self.capacity < cap:
            self._grow_particles(cap)
        else:
            dev = self.device
            new = {name: wp.zeros(cap, dtype=dtype, device=dev) for name, dtype in SLOT_DTYPES.items()}
            new.update({name: wp.zeros(cap, dtype=dtype, device=dev) for name, dtype in SCRATCH_DTYPES.items()})
            ids, scratch = self.cells.allocate(cap)
            if keep and self.capacity > 0:
                n = min(self.capacity, cap)
                for name in PARTICLE_ARRAYS:
                    wp.copy(new[name], getattr(self, name), count=n)
            for name, arr in new.items():
                setattr(self, name, arr)
            self.capacity = cap
            self.cells.set_buffers(ids, scratch)
        self._make_views()

    def _grow_particles(self, cap: int) -> None:
        """Grow the slot storage to ``cap`` slots one array at a time.

        The scratch arrays hold no state between steps and are released first (with the particle views and the
        captured graphs, which reference the old buffers); each state array is then copied into its new buffer
        and the old one released.  If an allocation fails, the state arrays moved so far are cut back to views of
        their first ``capacity`` slots (same contents) and the scratch arrays are reallocated before the error
        propagates."""
        old, dev = self.capacity, self.device
        self._graphs.clear()
        self._fine.clear()
        self._parts = self._parts_old = self._parts_comp = self._parts_fine = None
        for name in SCRATCH_ARRAYS:
            setattr(self, name, None)
        moved = []
        try:
            for name, dtype in SLOT_DTYPES.items():
                arr = wp.zeros(cap, dtype=dtype, device=dev)
                wp.copy(arr, getattr(self, name), count=old)
                setattr(self, name, arr)
                moved.append(name)
            scratch = {name: wp.zeros(cap, dtype=dtype, device=dev) for name, dtype in SCRATCH_DTYPES.items()}
            cell_buffers = self.cells.allocate(cap)
        except BaseException:
            for name in moved:
                setattr(self, name, getattr(self, name)[:old])
            for name, dtype in SCRATCH_DTYPES.items():
                setattr(self, name, wp.zeros(old, dtype=dtype, device=dev))
            self._make_views()
            raise
        for name, arr in scratch.items():
            setattr(self, name, arr)
        self.capacity = cap
        self.cells.set_buffers(*cell_buffers)

    def _make_views(self) -> None:
        """Particle views of the reinitialisations, the composite and the fine reconstruction (after the slot
        arrays changed; captured graphs and cached reconstructions are dropped)."""
        o = self._options
        # reinitialisation reads: newborns skipped with the composite default "off", included otherwise
        dyn, r = ("off" if o.composite_newborns == "off" else "mixed"), self.p2g_radius * (1.0 - 1.0e-3)
        self._parts = self._view(self.x, self.phi, self.g, r, dyn)
        self._parts_old = self._view(self._x_old, self._phi_old, self._g_old, r, dyn, self._valid_old)
        self._parts_comp = self._view(self.x, self.phi, self.g, o.composite_radius, o.composite_newborns)
        self._parts_fine = self._view(self.x, self.phi, self.g, o.reconstruct_radius, o.reconstruct_newborns)
        self._graphs.clear()
        self._fine.clear()

    def _ensure_capacity(self, needed: int, factor: float) -> None:
        if needed > self.capacity:
            grow = self.capacity > 0
            self._allocate_particles(max(int(needed * factor), self.ppc))
            if grow:
                self.grow_events += 1
            self.cells.build(self.x, self.valid)
            self._cells_stale = False

    def _fresh_cells(self) -> None:
        """Bin the newborns of the last step if they are not binned yet. Adaptive seeding ends a step without
        a rebuild: the next step's advection rebuilds the lists anyway, and only the reinitialisations, the
        compaction and the output read them before that. Whether the lists are stale is part of the CUDA
        graph keys, so a captured step always matches the state it is replayed on."""
        if self._cells_stale:
            self.cells.build(self.x, self.valid)
            self._cells_stale = False

    def _allocate_seeding(self, candidates: int) -> None:
        """Buffers for ``candidates`` seeding candidates per step (27 partial sums each)."""
        n, dev = max(int(candidates), MIN_SEED_CANDIDATES), self.device
        bufs = {"_seed_list": (n, wp.int32), "_seed_accept": (n, wp.int32), "_seed_offsets": (n, wp.int32),
                "_seed_pw": (27 * n, float), "_seed_pv": (27 * n, wp.vec4), "_seed_pC": (27 * n, wp.mat33)}
        new = {name: wp.zeros(size, dtype=dtype, device=dev) for name, (size, dtype) in bufs.items()}
        for name, arr in new.items():
            setattr(self, name, arr)
        self._seed_capacity = n
        self._graphs.clear()

    def _view(self, x, phi, g, radius: float, newborns: str = "mixed", valid=None) -> K.ParticleView3D:
        v = K.ParticleView3D()
        v.x, v.phi, v.g, v.C, v.valid = x, phi, g, self.C, self.valid if valid is None else valid
        v.radius = radius * self.grid.dx
        v.newborns = NEWBORN_MODES[newborns]
        return v

    def _count_cells_below(self, phi: wp.array, threshold: float) -> None:
        """Compact the cells with |phi| < threshold into ``_seed_cells`` / ``_seed_count`` (async)."""
        g = self.grid
        self._launch(K.mark_cells_below, g.shape, [phi, threshold, self._cell_flags])
        wp.utils.array_scan(self._cell_flags, self._cell_offsets, inclusive=False)
        self._launch(K.compact_cells, g.num_cells, [self._cell_flags, self._cell_offsets, self._seed_cells,
                                                    self._seed_count, g.num_cells])

    def _seed_slots_needed(self) -> int:
        """Host read of the slots needed by the last seed-cell compaction (synchronises)."""
        return int(self._seed_count.numpy()[0]) * self.ppc

    # ------------------------------------------------------------------ initialisation

    def init_from_sdf(self, sdf) -> None:
        """Initialise from a signed distance (``phi < 0`` inside).

        ``sdf`` is either a callable ``phi(x, y, z)`` on numpy arrays (e.g. from
        :mod:`pfmls.levelset.sdf3d`), evaluated exactly at cell centres and particles, or an
        ``(nx, ny, nz)`` array of cell-centre values (particles are then sampled trilinearly). The values
        must be finite, the callable must return an array of its inputs' shape and the level set must
        change sign inside the grid (``ValueError`` otherwise).
        """
        g = self.grid
        arr = checked_sdf(sdf, g.cell_centers(), "cell centres") if callable(sdf) else checked_grid_sdf(sdf, g.shape)
        check_sign_change(arr)
        self.phi_g.assign(arr)
        wp.copy(self._phi_s, self.phi_g)
        self._count_cells_below(self.phi_g, self._seed_threshold)
        needed = self._seed_slots_needed()
        cap = self._requested_capacity or int(needed * self._options.capacity_factor)
        self._allocate_particles(max(cap, needed), keep=False)  # a re-initialisation starts afresh
        self._launch(K.seed_lattice3, self.capacity, [self._seed_cells, self._seed_count, self.ppc, self.a, g.ny,
                                                      g.nz, g.dx, self.x, self.valid, self._n_used])
        if callable(sdf):
            xp = self.x.numpy()[:needed].astype(np.float64)
            vals = np.zeros(self.capacity, dtype=np.float32)
            vals[:needed] = checked_sdf(sdf, (xp[:, 0], xp[:, 1], xp[:, 2]), "particle positions")
            self.phi.assign(vals)
        else:
            self._launch(K.sample_grid_to_particles3, self.capacity, [self.phi_g, self.x, self.valid, 1.0 / g.dx,
                                                                      self.phi])
        self._launch(K.reset_maps3, self.capacity, [self._n_used, self.Js, self.Jinit, self.C])
        self.C0.zero_()
        self._dropped.zero_()
        self._seed_overflow.zero_()
        if self._options.seed_min > 0:
            self._allocate_seeding(max(needed // self.ppc, self._seed_capacity))
        self.cells.build(self.x, self.valid)
        self._cells_stale = False
        self._launch(K.shepard_gradient3, self.capacity, [self.x, self.phi, self.valid, self.cells.view, g.dx, 0,
                                                          self.g])
        wp.copy(self.g0, self.g)
        if self._options.init_normal_gradient:
            self._short_reinit(shepard=False)
            self._launch(K.update_particle_maps3, self.capacity,
                         [self.valid, self.Js, self.Jinit, self.C0, self.g0, self.C, self.g])

        self._reset_clock()
        self._set_constants(0.0, 0.0, (0.0,) * 8, 0)
        self._long_prepare(self._options.redistance_at_init)
        self._ensure_capacity(self._seed_slots_needed(), self._options.growth_factor)
        self._long_finish()
        self._short_reinit(shepard=not self._options.shepard_at_long_reinit)
        self._advance(None)
        self._initialized = True

    # ------------------------------------------------------------------ stepping

    def _set_constants(self, t: float, dt: float, params, istep: int) -> None:
        self._launch(K.set_step_constants3, 1, [self._clock, float(t), float(dt), self._params,
                                                FlowParams3D(*params), self._istep, int(istep)])

    def step(self, dt: float, velocity, skip_long_reinit: bool = False) -> None:
        """Advance by ``dt`` in ``velocity`` (a :class:`pfmls.flows.analytic3d.Flow3D`).

        ``skip_long_reinit`` drops a long reinitialisation that is due at this step (the schedule
        continues as if it had run; a short reinitialisation still runs when due).

        Raises ``ValueError`` (before any work is launched) if ``dt`` is not a finite number >= 0, or if
        the velocity does not match the grid or lives on another device.
        """
        o = self._options
        long, short, i, t1 = self._begin_step(dt, velocity, skip_long_reinit)
        compact = not long and o.compact_every > 0 and i % o.compact_every == 0
        self._set_constants(self.t, dt, velocity.params(), i)
        ux, uy, uz = velocity.mac_arrays(self.device)
        vkey = (velocity.func, ux.ptr, uy.ptr, uz.ptr)
        stale = self._cells_stale
        if long:
            self._graphs.run(("long_prepare", stale), lambda: self._long_prepare(True))
            self._cells_stale = False
            self._ensure_capacity(self._seed_slots_needed(), o.growth_factor)
            self._graphs.run(("long_finish", short) + vkey, lambda: self._finish_step(True, short, False, velocity))
        else:
            self._graphs.run(("step", short, compact, stale) + vkey,
                             lambda: self._finish_step(False, short, compact, velocity))
        self._cells_stale = o.seed_min > 0 and not compact  # also after a graph replay (no host code runs)
        if o.seed_min > 0 and i % o.capacity_check_every == 0:
            self._check_seeding_storage()
        self.t = t1
        self.step_index = i

    def _check_seeding_storage(self) -> None:
        """Grow the slot storage when seeding has used most of it (or had to drop newborns), and the candidate
        buffers when more cells were flagged than they hold; one host read (synchronises)."""
        self._launch(K.gather_counters, 1, [self._n_used, self._dropped, self._seed_overflow, self._counters])
        used, dropped, overflow = (int(v) for v in self._counters.numpy())
        if overflow > 0:  # those cells were not seeded at that step (they are flagged again at the next one)
            self._seed_overflow.zero_()
            self._allocate_seeding(int((self._seed_capacity + overflow) * self._options.growth_factor))
        if dropped:
            self.dropped_newborns += dropped
            self._dropped.zero_()
            warnings.warn(f"adaptive seeding dropped {dropped} new particles because the storage ({self.capacity} "
                          f"slots) was full; growing it (raise capacity_factor / growth_factor or lower "
                          f"capacity_check_every to avoid this)", RuntimeWarning, stacklevel=3)
        if dropped or used > GROWTH_FILL * self.capacity:
            self._ensure_capacity(max(used, self.capacity) + 1, self._options.growth_factor)

    def _finish_step(self, long: bool, short: bool, compact: bool, velocity) -> None:
        if long:
            self._long_finish()
        if short:
            # after a long reinitialisation with Shepard gradients, g already is the Shepard gradient of the
            # current particles
            self._short_reinit(shepard=not (long and self._options.shepard_at_long_reinit))
        self._advance(velocity)
        if compact:
            self._compact()

    def _advance(self, velocity) -> None:
        g, n, o = self.grid, self.capacity, self._options
        maps = [self.Jinit, self.C0, self.g0, self.C, self.g]
        if velocity is not None:  # RK4 of x and Js, then C = Js^T C0 Js and g = normalize(J_long^T g0)
            rk4, sl = K.flow_kernels3(velocity.func)
            ux, uy, uz = velocity.mac_arrays(self.device)
            self._launch(rk4, n, [self.x, self.Js, self.valid, self._clock, self._params, ux, uy, uz, 1.0 / g.dx,
                                  self._stage, self._x_max, *maps])
        else:
            self._launch(K.update_particle_maps3, n, [self.valid, self.Js, *maps])
        self.cells.build(self.x, self.valid)
        self._cells_stale = False
        if velocity is not None:
            self._launch(sl, g.shape, [self.phi_g, self._phi_tmp, self._clock, self._params, ux, uy, uz, g.dx,
                                       self._stage])
        else:
            wp.copy(self._phi_tmp, self.phi_g)
        self._eikonal.redistance(self._phi_tmp, self._phi_s, self._grid_eik_iters)
        self._launch(K.composite_grid3, g.shape,
                     [self._parts_comp, self.cells.view, g.dx, self._w_inv_h(g.dx / o.composite_factor),
                      self._phi_s, self.phi_g])
        if o.seed_min > 0 and velocity is not None:
            self._adaptive_seed()

    def _adaptive_seed(self) -> None:
        """One candidate per flagged cell (compacted into a list), partial neighbour sums per (candidate,
        neighbour cell), acceptance, prefix sum of the accepted candidates and storage (deterministic)."""
        g, o = self.grid, self._options
        band = self._band * g.dx
        nc, cells = self._seed_capacity, self.cells.view
        self._launch(K.mark_deficient_cells, g.shape, [self.phi_g, cells, band, o.seed_min, self._cell_flags])
        wp.utils.array_scan(self._cell_flags, self._cell_offsets, inclusive=False)
        self._launch(K.compact_cells, g.num_cells, [self._cell_flags, self._cell_offsets, self._seed_list,
                                                    self._list_count, g.num_cells])
        sums = [self._seed_pw, self._seed_pv, self._seed_pC]
        self._launch(K.seed_partial_sums3, 27 * nc, [self._seed_list, self._list_count, self._parts, cells,
                                                     self._istep, g.dx, *sums])
        self._launch(K.seed_accept3, nc, [self._seed_list, self._list_count, cells, self._phi_s, self._istep, g.dx,
                                          band, o.seed_coherence, *sums, self._seed_accept])
        wp.utils.array_scan(self._seed_accept, self._seed_offsets, inclusive=False)
        self._launch(K.seed_store3, nc, [self._seed_list, self._list_count, self._seed_accept, self._seed_offsets,
                                         cells, self._istep, g.dx, *sums, self._n_used, self._dropped, self.x,
                                         self.phi, self.g, self.g0, self.C, self.C0, self.Js, self.Jinit,
                                         self.valid])
        self._launch(K.seed_advance3, 1, [self._list_count, self._seed_accept, self._seed_offsets, self._n_used,
                                          self.capacity, self._seed_overflow])
        self._cells_stale = True

    def _compact(self) -> None:
        """Reorder the binned particles into cell order (keeping their order within a cell, so neighbour sums
        are unchanged up to rounding), drop the newborns far from the interface and free every other slot."""
        g, n, o = self.grid, self.capacity, self._options
        self._fresh_cells()
        cells = self.cells
        far = self._band + 2.0 if o.cull_distance is None else o.cull_distance
        self._launch(K.mark_kept_slots, n, [cells.ids, cells.start, g.num_cells, self.x, self.valid, self._phi_s, g.dx,
                                            far * g.dx, self._flags])
        wp.utils.array_scan(self._flags, self._offsets, inclusive=False)
        sel = [cells.ids, self._flags, self._offsets]
        for kernel, arr, tmp in ((K.gather_vec3, self.x, self._x_old), (K.gather_vec3, self.g, self._x_old),
                                 (K.gather_vec3, self.g0, self._x_old), (K.gather_float, self.phi, self._phi_old),
                                 (K.gather_int, self.valid, self._list)):
            self._launch(kernel, n, [arr, *sel, tmp])
            wp.copy(arr, tmp)
        for arr in (self.Js, self.Jinit, self.C, self.C0):
            self._launch(K.gather_matrix_rows, n, [arr, *sel, self._x_old, self._g_old, self._v_tmp])
            self._launch(K.set_matrix_rows, n, [self._flags, self._offsets, self._x_old, self._g_old, self._v_tmp, arr])
        self._launch(K.finish_compaction, n, [self._flags, self._offsets, self.valid, self._n_used])
        self.cells.build(self.x, self.valid)
        self._cells_stale = False

    def _particle_to_grid(self) -> None:
        g = self.grid
        self._launch(K.p2g_cells3, g.shape, [self._parts, self.cells.view, g.dx, self._w_inv_h(g.dx),
                                             self._phi_r, self._g_r, self._mask])
        self._launch(_fill_scalar3, g.shape, [self._mask, self.phi_g, self._phi_r])

    def _eikonal_guess(self) -> None:
        self._eikonal.redistance(self._phi_r, self._phi_eik)
        self._launch(_central_normal, self.grid.shape, [self._phi_eik, self._g_eik])

    def _long_prepare(self, redistance: bool) -> None:
        """Long reinitialisation up to the seed-cell compaction (before the storage check)."""
        g, o = self.grid, self._options
        self._fresh_cells()
        cells = self.cells.view
        if o.shepard_at_long_reinit:
            # Shepard gradients of the particles whose gradient this step reads: those of the cell-centre P2G
            # and the near-interface particles of the Newton / reseed interpolations (the reseed uses particle
            # values only where |phi_P| < blend_gate dx, interpolated from particles within p2g_radius)
            self._shepard_subset((o.blend_gate + self.p2g_radius + SHEPARD_MARGIN) * g.dx)
        self._particle_to_grid()
        self._eikonal_guess()
        if redistance:
            self._launch(_fill_scalar3, g.shape, [self._mask, self._phi_eik, self._phi_r])
            self._launch(_fill_vector3, g.shape, [self._mask, self._g_eik, self._g_r])
            self._redistancer.run(self._phi_r, self._g_r, self._X_r, self._mask, self._phi_eik, self._parts, cells)
        else:
            self._launch(_fill_vector3, g.shape, [self._mask, self._g_eik, self._g_r])
            self._launch(hermite_cross_terms, g.shape, [self._g_r, self._X_r, g.dx])
        self._count_cells_below(self._phi_r, self._seed_threshold)

    def _long_finish(self) -> None:
        """Reseed the seed cells from the redistanced grid and the old particles; reset the long map."""
        g, n, o = self.grid, self.capacity, self._options
        cells = self.cells.view
        wp.copy(self._x_old, self.x)
        wp.copy(self._phi_old, self.phi)
        wp.copy(self._g_old, self.g)
        wp.copy(self._valid_old, self.valid)
        self._launch(K.reseed3, n, [self._parts_old, cells, self._phi_r, self._g_r, self._X_r, self._seed_cells,
                                    self._seed_count, self.ppc, self.a, g.dx, self._band * g.dx, o.blend_e1,
                                    o.blend_e2, o.blend_gate, o.blend_grid_gate, self.x, self.phi, self.g, self.g0,
                                    self.valid, self._n_used])
        self._launch(K.reset_maps3, n, [self._n_used, self.Js, self.Jinit, self.C])
        self.cells.build(self.x, self.valid)
        self._cells_stale = False
        if o.shepard_at_long_reinit:  # (the reseeded slots hold no newborns)
            self._launch(K.shepard_gradient3, n, [self.x, self.phi, self.valid, cells, g.dx, 0, self.g])
            wp.copy(self.g0, self.g)

    def _shepard_subset(self, phi_below: float) -> None:
        """Shepard gradients of the particles that the cell-centre P2G reads and of those with
        ``|phi_p| < phi_below`` (the other particles keep their transported gradients)."""
        g, n, skip = self.grid, self.capacity, self._skip_newborns()
        self._launch(K.mark_p2g_particles, n, [self.x, self.phi, self.valid, self.cells.view, g.dx,
                                               self.p2g_radius * g.dx * (1.0 + 1.0e-4), phi_below, skip, self._flags])
        wp.utils.array_scan(self._flags, self._offsets, inclusive=False)
        self._launch(K.compact_cells, n, [self._flags, self._offsets, self._list, self._list_count, n])
        self._launch(K.shepard_gradient3_listed, n, [self._list, self._list_count, self.x, self.phi, self.valid,
                                                     self.cells.view, g.dx, skip, self.g])

    def _skip_newborns(self) -> int:
        """1 if the reinitialisations ignore the newborns of adaptive seeding (``composite_newborns="off"``)."""
        return int(self._options.composite_newborns == "off")

    def _short_reinit(self, shepard: bool = True) -> None:
        g, n, o = self.grid, self.capacity, self._options
        self._fresh_cells()
        if o.normal_gradient == "particles":
            self._launch(K.normal_gradient_particles3, n, [self.x, self.g, self.valid, self.cells.view,
                                                           o.normal_gradient_radius * g.dx, o.hessian_scale,
                                                           self._skip_newborns(), self.C0])
        else:
            if shepard:  # Shepard gradients of the particles that the cell-centre reconstruction reads
                self._shepard_subset(0.0)
            self._particle_to_grid()
            self._eikonal_guess()
            self._launch(_fill_vector3, g.shape, [self._mask, self._g_eik, self._g_r])
            self._launch(K.split_to_faces3, (g.nx + 1, g.ny + 1, g.nz + 1),
                         [self._g_r, self._face_x, self._face_y, self._face_z])
            self._launch(K.normal_gradient_from_faces3, n,
                         [self.x, self.valid, self._face_x, self._face_y, self._face_z, 1.0 / g.dx,
                          o.hessian_scale, self.C0])
        self._launch(K.short_map_reset3, n, [self.valid, self.Js, self.Jinit])

    def scrub_nonfinite(self, counter: wp.array | None = None) -> None:
        """Repair non-finite state (e.g. after a float overflow upstream), without host synchronisation.

        Particles with a non-finite position, value or gradient are deactivated (until the next long
        reinitialisation or compaction frees their slot); non-finite grid values become "outside"
        (``(nx + ny + nz) dx``). If given, the int32 array ``counter`` (at least 2 entries) accumulates the
        number of repaired grid cells (entry 0) and particles (entry 1). A no-op for a finite state.
        """
        counter = self._scrub_target(counter)
        g = self.grid
        self._launch(K.scrub_nonfinite_grid3, g.shape, [self.phi_g, self._far, counter, 0])
        self._launch(K.scrub_nonfinite_grid3, g.shape, [self._phi_s, self._far, counter, 0])
        if self.capacity > 0:
            self._launch(K.scrub_nonfinite_particles3, self.capacity, [self.x, self.phi, self.g, self.valid,
                                                                       counter, 1])

    # ------------------------------------------------------------------ state

    def _scalars(self) -> dict:
        return {**self._clock_scalars(), "capacity": self.capacity, "ppc": self.ppc, "res": tuple(self.grid.shape)}

    def _check_state(self, state: dict, what: str) -> None:
        """Raise ``ValueError`` unless ``state`` (a snapshot or a state dict) matches this instance's
        resolution, ppc and dx and its arrays match its slot count."""
        snap = what == "snapshot"

        def key(name):
            return name if snap else name.lstrip("_")

        missing = [key(n) for n in PARTICLE_ARRAYS + GRID_ARRAYS if key(n) not in state]
        missing += [n for n in ("capacity", "t", "step_index", "i_long", "t_long", "t_short") if n not in state]
        if missing:
            raise ValueError(f"{what} has no entries {missing}")
        here = f"(this level set: res {'x'.join(map(str, self.grid.shape))}, ppc {self.ppc}, dx {self.grid.dx})"
        if "ppc" in state and int(state["ppc"]) != self.ppc:
            raise ValueError(f"{what} has ppc={int(state['ppc'])} {here}")
        if "res" in state and tuple(int(v) for v in np.atleast_1d(state["res"])) != tuple(self.grid.shape):
            raise ValueError(f"{what} has res={tuple(int(v) for v in np.atleast_1d(state['res']))} {here}")
        dx = state.get("dx")
        if dx is not None and abs(float(dx) - self.grid.dx) > 1.0e-6 * self.grid.dx:
            raise ValueError(f"{what} has dx={float(dx)} {here}")
        cap = int(state["capacity"])
        if cap <= 0 or self._round_capacity(cap) != cap:
            raise ValueError(f"{what} has capacity={cap}, not a multiple of {SLOT_BLOCK} ppc = "
                             f"{SLOT_BLOCK * self.ppc} {here}")
        for name in PARTICLE_ARRAYS + GRID_ARRAYS:
            if name in SLOT_DTYPES:
                dtype, want = SLOT_DTYPES[name], (cap,)
            else:
                mine = getattr(self, name)
                dtype, want = mine.dtype, tuple(mine.shape)
            a = state[key(name)]
            if snap:
                ok = isinstance(a, wp.array) and tuple(a.shape) == want and a.dtype == dtype
            else:
                want = want + tuple(getattr(dtype, "_shape_", ()))
                ok = tuple(np.shape(a)) == want
            if not ok:
                raise ValueError(f"{what} entry {key(name)!r} has shape {tuple(np.shape(a))}, expected {want} {here}")

    def _set_scalars(self, s: dict) -> None:
        self._set_clock(s)
        if self._options.seed_min > 0 and self._seed_capacity == 0:  # loaded without init_from_sdf
            self._allocate_seeding(self.capacity // (2 * self.ppc))

    def _match_capacity(self, capacity: int) -> None:
        if capacity != self.capacity:
            self._allocate_particles(capacity, keep=False)  # the arrays are overwritten next

    def snapshot(self, out: dict | None = None) -> dict:
        """Device-side copy of the full state (particles, maps, grids, clock) for a later :meth:`restore`.

        Passing a previous snapshot of the same capacity as ``out`` reuses its buffers.
        """
        snap = out if out is not None and out.get("capacity") == self.capacity else {}
        for name in PARTICLE_ARRAYS + GRID_ARRAYS:
            src = getattr(self, name)
            dst = snap.get(name)
            if dst is None or dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device:
                dst = snap[name] = wp.empty_like(src)
            wp.copy(dst, src)
        snap.update(self._scalars())
        return snap

    def restore(self, snap: dict) -> None:
        """Return to the state captured by :meth:`snapshot` (graphs stay valid unless the capacity changed).

        Raises ``ValueError`` (before anything is changed) if the snapshot comes from a level set with another
        resolution, ppc or dx.
        """
        self._check_state(snap, "snapshot")
        self._match_capacity(int(snap["capacity"]))
        for name in PARTICLE_ARRAYS + GRID_ARRAYS:
            wp.copy(getattr(self, name), snap[name])
        self._set_scalars(snap)
        self.cells.build(self.x, self.valid)
        self._cells_stale = False

    def state_dict(self) -> dict:
        """Host copy of the full state (numpy arrays and scalars), e.g. for checkpoints."""
        d = {name.lstrip("_"): self._host(getattr(self, name)) for name in PARTICLE_ARRAYS + GRID_ARRAYS}
        d.update(self._scalars())
        return d

    def load_state_dict(self, state: dict) -> None:
        """Load a state written by :meth:`state_dict` (same resolution, ppc and dx, else ``ValueError``
        before anything is changed)."""
        self._check_state(state, "state_dict")
        self._match_capacity(int(state["capacity"]))
        for name in PARTICLE_ARRAYS + GRID_ARRAYS:
            getattr(self, name).assign(np.asarray(state[name.lstrip("_")]))
        self._set_scalars(state)
        self.cells.build(self.x, self.valid)
        self._cells_stale = False

    # ------------------------------------------------------------------ output

    def fine_shape(self, factor: int | None = None, res: int | None = None) -> tuple[tuple[int, int, int], float]:
        """Shape and spacing of a fine reconstruction grid: ``factor`` x the grid, or ``res`` cells along x.

        The default (both None) is the largest factor <= 4 that keeps every axis within ``FINE_CELLS`` (512)
        cells: factor 4 up to 128^3, 2 at 192^3 and 256^3 (the buffer stays cached until :meth:`release_fine`)."""
        g = self.grid
        if res is None:
            f = max(1, min(4, self.FINE_CELLS // max(g.shape))) if factor is None else int(factor)
            return (g.nx * f, g.ny * f, g.nz * f), g.dx / f
        h = g.nx * g.dx / int(res)
        return (int(res), int(round(g.ny * g.dx / h)), int(round(g.nz * g.dx / h))), h

    def reconstruct_device(self, factor: int | None = None, res: int | None = None,
                           newborns: str | None = None) -> wp.array:
        """Fine reconstruction on the device (see :meth:`fine_shape`): particles where a particle lies within
        ``reconstruct_radius dx`` of a fine cell centre, trilinear grid elsewhere. ``newborns`` ("off",
        "fallback", "mixed"; default ``reconstruct_newborns``) sets how the particles created by adaptive
        seeding count. The buffer is cached per shape."""
        g = self.grid
        parts = self._parts_fine
        if newborns is not None:
            if newborns not in NEWBORN_MODES:
                raise ValueError(f"newborns must be one of {sorted(NEWBORN_MODES)}, got {newborns!r}")
            parts = self._view(self.x, self.phi, self.g, self._options.reconstruct_radius, newborns)
        shape, h = self.fine_shape(factor, res)
        out = self._fine.get(shape)
        if out is None:
            out = self._fine[shape] = wp.zeros(shape, dtype=float, device=self.device)
        self._fresh_cells()
        self._launch(K.reconstruct_fine3, shape, [parts, self.cells.view, h, 1.0 / g.dx, self._w_inv_h(h),
                                                   self._phi_s, out])
        return out

    def reconstruct(self, factor: int | None = None, res: int | None = None,
                    newborns: str | None = None) -> np.ndarray:
        """Fine reconstruction as a numpy array (a copy; see :meth:`reconstruct_device`)."""
        return self._host(self.reconstruct_device(factor, res, newborns))

    def release_fine(self) -> None:
        """Free the cached fine-reconstruction buffers."""
        self._fine.clear()

    def mesh(self, factor: int | None = None, res: int | None = None, newborns: str | None = None):
        """Marching-cubes mesh ``(verts, faces)`` of the zero level set of the fine reconstruction
        (physical coordinates, outward-oriented triangles; closed along the domain walls)."""
        shape, h = self.fine_shape(factor, res)
        return measure3d.marching_cubes(self.reconstruct(factor, res, newborns), h)

    def volume(self, factor: int | None = None, res: int | None = None, newborns: str | None = None) -> float:
        """Volume of ``{phi < 0}`` from the marching-cubes mesh of the fine reconstruction."""
        return measure3d.mesh_volume(*self.mesh(factor, res, newborns))

    def particles(self) -> dict:
        """Valid particles as numpy arrays: ``x`` (n, 3), ``phi`` (n,), ``g`` (n, 3) and ``newborn`` (n,), true
        for particles created by adaptive seeding."""
        valid = self.valid.numpy()
        v = valid != 0
        return {"x": self.x.numpy()[v], "phi": self.phi.numpy()[v], "g": self.g.numpy()[v],
                "newborn": valid[v] == 2}

    def num_valid(self) -> int:
        """Number of active (band) particles."""
        return int(np.count_nonzero(self.valid.numpy()))

    def num_used(self) -> int:
        """Number of slots in use (valid particles and the free slots between them)."""
        return int(self._n_used.numpy()[0])

    def memory_bytes(self) -> dict:
        """Device memory held by the level set, in bytes: ``particles`` (slot arrays incl. scratch and the
        cell-list id lists), ``seeding`` (adaptive-seeding candidate buffers), ``grid`` (grid arrays, solvers
        and cell-list ranges) and ``fine`` (cached reconstructions)."""
        def size(a):
            return a.capacity if a is not None else 0

        cell_ids, cell_ranges = self.cells.memory_bytes()
        part = sum(size(getattr(self, n, None)) for n in PARTICLE_ARRAYS + SCRATCH_ARRAYS) + cell_ids
        seeding = sum(size(getattr(self, n, None)) for n in SEEDING_ARRAYS)
        grid = sum(size(v) for v in vars(self).values() if isinstance(v, wp.array) and v.ndim == 3)
        grid += sum(size(a) for a in (self._cell_flags, self._cell_offsets, self._seed_cells)) + cell_ranges
        for solver in (self._eikonal, self._redistancer):
            grid += sum(size(v) for v in vars(solver).values() if isinstance(v, wp.array))
        fine = sum(size(a) for a in self._fine.values())
        return {"particles": part, "seeding": seeding, "grid": grid, "fine": fine,
                "total": part + seeding + grid + fine}
