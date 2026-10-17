"""PFM-LS level-set backend for the free-surface solver.

Wraps :class:`pfmls.levelset.PFMLevelSet2D`: its particles and background grid are advected with the
extrapolated MAC velocity, and its composite grid level set ``phi_g`` drives the pressure mask and the
ghost-fluid fractions.  The surface-tension curvature is taken from a closest-point redistanced copy of the level
set (:meth:`PFMLevelSetBackend.curvature_source`).  The backend is configured by :class:`PFMBackendConfig`
(``FluidConfig.pfm_levelset``); its defaults are the free-surface settings, :meth:`PFMBackendConfig.plain` gives the
plain PFM-LS behaviour (closed walls, B-spline velocity, no film closure).

Ghost layer and wall separation
-------------------------------
With ``pad > 0`` the level set lives on the fluid grid padded by ``pad`` ghost cells beyond every wall.  Before
each step the padded velocity is filled from the fluid faces (constant extension across the walls), particles in
the ghost layer are deactivated (as without padding, only particles inside the box carry the interface), and the
ghost cells are rebuilt from the cells they mirror: evenly where the wall face is closed (``u.n = 0``,
the wall is a symmetry plane) and oddly where the projection opened it (the liquid moves away from the wall, which
is then an interface with air behind it; the values of liquid cells and particles next to the open face are capped
by their distance to the wall).  Semi-Lagrangian departure points beyond an open wall therefore see air,
and liquid can detach from the lid or a side wall without the level set keeping a film there (see
:class:`pfmls.fluid.pressure.PressureSolver` for the unilateral wall condition).  ``pad = 0`` keeps the walls closed.

Velocity sampling (``velocity_sampling``): ``"linear"`` (default) samples the MAC velocity bilinearly (interpolating);
``"bspline"`` is the level set's own quadratic B-spline sampler (:class:`pfmls.flows.MACVelocity2D`).
The B-spline is an approximating smoother: at the surface of a thin sheet it mixes the liquid
velocity with its extrapolation into the air, which slows down the surfaces of converging or thinning liquid.

Robustness layers on top of the level set:

* **NaN scrub** before every step (:meth:`PFMLevelSet2D.scrub_nonfinite`): non-finite grid values become air,
  particles with a non-finite position, value or gradient are deactivated;
* **volume guard** on long-reinitialisation steps (the steps that redistance and reseed): the level-set state is
  snapshotted on the device, and if the liquid area (marching squares on the grid) changes by more than
  ``volume_guard`` of the current liquid area, the snapshot is restored and the step is redone without the long
  reinitialisation.  This costs one host read per long reinitialisation;
* **squeezed-film closure** (``close_films``): when two liquid fronts meet, PFM-LS transports the air between them
  exactly, and a film thinner than a cell survives indefinitely (the liquid-only solver puts no pressure in it)
  while the liquid behind it keeps flowing in, so liquid is lost at the stalled fronts.  An air cell between two
  liquid neighbours whose marching-squares film thickness is below ``film_width`` cells and that is being compressed
  (``div u < 0`` across it) is turned into liquid: ``phi_g`` is set to ``-film_value dx`` and the air-side particles
  in the cell are deactivated (as by the NaN scrub, until the next long reinitialisation).  A one-cell notch of a
  wavy surface passes the same test; ``film_deep`` (in cells, 0 = off) tells the two apart: a film is bounded
  across its thickness by thin air or liquid, while a notch opens onto deeper air, so a cell squeezed along one axis
  is only closed if neither of its neighbours along the other axis lies more than ``film_deep`` cells into the air.

The adapter relies on these :class:`PFMLevelSet2D` members.  Public API: ``scrub_nonfinite``, ``reinit_due``,
``snapshot``/``restore``, ``step(..., skip_long_reinit=...)``, ``reconstruct_device``, ``num_valid``, ``num_slots``,
``state_dict``/``load_state_dict``.  State arrays it edits between steps, as the level set's docstring allows:
``phi_g`` (ghost rebuild, open-wall capping, film closure), the particle ``phi`` (capped next to open walls) and
``valid`` (cleared in the ghost layer and in closed films), read together with ``x``.  ``tests/test_fluid.py``
exercises all of them through the default solver.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from ..core import options as _opt
from ..flows import MACVelocity2D
from ..levelset import PFMLevelSet2D, PFMLS2DOptions
from ._kernels import dual_cell_liquid_area, mac_velocity_linear
from .redistance import redistance_closest_point
from .pressure import smooth_band_kernel

#: Curvature source: closest-point redistancing of a copy of the level set within CURVATURE_BAND dx of the interface
#: (CURVATURE_NEWTON_ITERS iterations), CURVATURE_JACOBI_PASSES eikonal passes beyond, then ``curvature_passes``
#: binomial passes within CURVATURE_SMOOTH_BAND dx of the interface.
CURVATURE_BAND = 2.0
CURVATURE_NEWTON_ITERS = 8
CURVATURE_JACOBI_PASSES = 4
CURVATURE_SMOOTH_BAND = 4.0
#: Liquid cells and particles within OPEN_WALL_LAYERS cells of an open wall face are capped by their distance to it.
OPEN_WALL_LAYERS = 3

VELOCITY_SAMPLINGS = ("linear", "bspline")
REINIT_UNITS = ("nominal_steps", "steps", "time")


@dataclass(frozen=True)
class PFMBackendConfig:
    """PFM-LS interface of the free-surface solver (``FluidConfig.pfm_levelset``).

    The defaults are the settings of the water examples; :meth:`plain` returns the plain PFM-LS behaviour.
    :meth:`updated` changes fields of this class and of :attr:`options` by name, :func:`parse_options` converts
    command-line ``KEY=VALUE`` strings (the ``--levelset-opt`` flag of the water examples).
    """

    #: particles per cell (a perfect square >= 9)
    ppc: int = 16
    #: particle band half-width (dx)
    band: float = 2.0
    #: long / short reinitialisation intervals, in the unit given by ``reinit_unit``
    reinit_long: float = 12
    reinit_short: float = 6
    #: "nominal_steps": nominal time steps of the solver, converted to time units, so a CFL-limited step does not
    #: add reinitialisations (each perturbs the interface slightly); "steps" or "time": passed on as given
    reinit_unit: str = "nominal_steps"
    #: advanced level-set options; the free surface uses the Taylor-consistent normal-gradient scale 0.5
    #: (docs/method.md, section 3) and no adaptive seeding (the settings the water examples were validated with)
    options: PFMLS2DOptions = field(default_factory=lambda: PFMLS2DOptions(hessian_scale=0.5, seed_min=0))
    #: capture the level-set steps into CUDA graphs
    use_graph: bool = True
    #: ghost cells beyond every wall (0 = closed walls only; > 0 lets liquid detach from walls the projection opens)
    pad: int = 4
    #: undo a long reinitialisation that changes the liquid area by more than this fraction of the current liquid
    #: area (0 = off)
    volume_guard: float = 0.002
    #: 3x3 binomial passes on the redistanced copy of the level set that the curvature is taken from (0 = none;
    #: smoothing the curvature itself is ``PressureConfig.curvature_smoothing``)
    curvature_passes: int = 1
    #: MAC velocity sampler of the level set: "linear" (bilinear) or "bspline" (quadratic B-spline)
    velocity_sampling: str = "linear"
    #: close squeezed sub-cell air films between liquid fronts (see the module docstring)
    close_films: bool = True
    #: film closure: maximum film thickness (cells), value given to a closed cell (-film_value dx), and the air depth
    #: (cells) across the film beyond which a squeezed cell counts as a notch and stays open (0 = no notch test)
    film_width: float = 0.5
    film_value: float = 0.25
    film_deep: float = 0.75

    def __post_init__(self):
        _opt.check_field_types(self)
        if not isinstance(self.options, PFMLS2DOptions):
            raise TypeError(f"options must be a PFMLS2DOptions, got {type(self.options).__name__}")
        if self.reinit_unit not in REINIT_UNITS:
            raise ValueError(f"reinit_unit must be one of {REINIT_UNITS}, got {self.reinit_unit!r}")
        if self.velocity_sampling not in VELOCITY_SAMPLINGS:
            raise ValueError(f"velocity_sampling must be one of {VELOCITY_SAMPLINGS}, got {self.velocity_sampling!r}")
        for name in ("pad", "curvature_passes"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("volume_guard", "film_width", "film_value", "film_deep"):
            if not getattr(self, name) >= 0.0:
                raise ValueError(f"{name} must be >= 0")

    @classmethod
    def plain(cls, **overrides) -> "PFMBackendConfig":
        """Plain PFM-LS: closed walls, the level set's B-spline velocity, no film closure, no curvature smoothing,
        intervals in steps and ``hessian_scale = 1``; ``overrides`` as in :meth:`updated`."""
        base = cls(reinit_unit="steps", options=PFMLS2DOptions(seed_min=0), pad=0, curvature_passes=0,
                   velocity_sampling="bspline", close_films=False)
        return base.updated(**overrides)

    def updated(self, **overrides) -> "PFMBackendConfig":
        """A copy with fields of this class or of :class:`pfmls.PFMLS2DOptions` replaced by name.

        Raises ``TypeError`` for a name that is in neither class.
        """
        own = {f.name for f in dataclasses.fields(self)} - {"options"}
        mine = {k: v for k, v in overrides.items() if k in own}
        theirs = {k: v for k, v in overrides.items() if k not in own and k != "options"}
        unknown = _opt.unknown_keys(PFMLS2DOptions, theirs)
        if unknown:
            raise TypeError(f"unknown option(s) {unknown}: not a field of PFMBackendConfig or PFMLS2DOptions")
        opts = overrides.get("options", self.options)
        if theirs:
            opts = dataclasses.replace(opts, **theirs)
        return dataclasses.replace(self, options=opts, **mine)

    def levelset_kwargs(self, dt_nominal: float) -> dict:
        """Constructor arguments of :class:`PFMLevelSet2D` (``reinit_unit="nominal_steps"`` becomes time units of
        ``dt_nominal``, or steps if ``dt_nominal`` is not a positive number)."""
        rl, rs, unit = self.reinit_long, self.reinit_short, self.reinit_unit
        if unit == "nominal_steps":
            if math.isfinite(dt_nominal) and dt_nominal > 0.0:
                rl, rs, unit = float(rl) * dt_nominal, float(rs) * dt_nominal, "time"
            else:
                unit = "steps"
        return dict(ppc=self.ppc, band=self.band, reinit_long=rl, reinit_short=rs, reinit_unit=unit,
                    options=self.options, use_graph=self.use_graph)


def parse_options(pairs) -> dict:
    """Typed keyword arguments for :meth:`PFMBackendConfig.updated` from ``"KEY=VALUE"`` strings.

    A key is a field of :class:`PFMBackendConfig` (except ``options``) or of :class:`pfmls.PFMLS2DOptions`; values
    are converted by the field's annotation (bools: ``true/false``, ``1/0``, ...).  Raises ``ValueError`` for an
    unknown key or an unconvertible value.
    """
    own = {f.name for f in dataclasses.fields(PFMBackendConfig)} - {"options"}
    out = {}
    for pair in pairs:
        key, value = _opt.split_pair(pair)
        if key in own:
            out[key] = _opt.convert_value(PFMBackendConfig, key, value)
        elif key in {f.name for f in dataclasses.fields(PFMLS2DOptions)}:
            out[key] = _opt.convert_value(PFMLS2DOptions, key, value)
        else:
            names = sorted(own | {f.name for f in dataclasses.fields(PFMLS2DOptions)})
            raise ValueError(f"unknown level-set option {key!r}; options: {', '.join(names)}")
    return out


class LinearMACVelocity2D(MACVelocity2D):
    """MAC velocity for the level set, sampled bilinearly (interpolating; ``u.n = 0`` on closed walls)."""

    func = mac_velocity_linear


@wp.kernel
def _area_kernel(phi: wp.array2d(dtype=float), dx: float, slot: int, out: wp.array(dtype=float)):
    I, J = wp.tid()
    wp.atomic_add(out, slot, dual_cell_liquid_area(phi, I, J, dx))


@wp.kernel
def _pad_faces_kernel(src: wp.array2d(dtype=float), pad: int, dst: wp.array2d(dtype=float)):
    """dst[I, J] = src[clamp(I - pad), clamp(J - pad)]: constant extension of a face grid across the walls."""
    I, J = wp.tid()
    i = wp.clamp(I - pad, 0, src.shape[0] - 1)
    j = wp.clamp(J - pad, 0, src.shape[1] - 1)
    dst[I, J] = src[i, j]


@wp.func
def _mirror(I: int, pad: int, n: int) -> int:
    """Interior cell mirrored to padded cell index I (cells 0..n-1 of the box sit at pad..pad+n-1)."""
    i = I - pad
    if i < 0:
        i = -1 - i
    elif i > n - 1:
        i = 2 * n - 1 - i
    return wp.clamp(i, 0, n - 1)


@wp.kernel
def _ghost_kernel(phi: wp.array2d(dtype=float), u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), pad: int):
    """Rebuild the ghost cells: even reflection across closed walls, air (|phi|, the odd reflection of liquid)
    behind wall faces with a non-zero velocity (opened by the projection, or next to air)."""
    I, J = wp.tid()
    nx = v.shape[0]
    ny = u.shape[1]
    i = I - pad
    j = J - pad
    if i >= 0 and i < nx and j >= 0 and j < ny:
        return
    ci = _mirror(I, pad, nx)
    cj = _mirror(J, pad, ny)
    open_wall = int(0)
    if i < 0 and u[0, cj] != 0.0:
        open_wall = 1
    if i > nx - 1 and u[nx, cj] != 0.0:
        open_wall = 1
    if j < 0 and v[ci, 0] != 0.0:
        open_wall = 1
    if j > ny - 1 and v[ci, ny] != 0.0:
        open_wall = 1
    val = phi[ci + pad, cj + pad]
    if open_wall == 1:
        val = wp.abs(val)  # odd reflection of liquid: the open wall is the interface, air behind it
    phi[I, J] = val


@wp.func
def _opened(phi: wp.array2d(dtype=float), face: float, i: int, j: int, pad: int) -> bool:
    """A wall face is open when the projection gave it a velocity although its wall cell (i, j) is liquid."""
    return face != 0.0 and phi[i + pad, j + pad] < 0.0


@wp.kernel
def _open_wall_distance_kernel(phi: wp.array2d(dtype=float), u: wp.array2d(dtype=float), v: wp.array2d(dtype=float),
                               pad: int, layers: int, dx: float):
    """Liquid cells within ``layers`` cells of an open wall face: phi = max(phi, -distance to the wall)."""
    i, j = wp.tid()
    nx = v.shape[0]
    ny = u.shape[1]
    c = phi[i + pad, j + pad]
    if c >= 0.0:
        return
    d = c
    if i < layers and _opened(phi, u[0, j], 0, j, pad):
        d = wp.max(d, -(float(i) + 0.5) * dx)
    if i >= nx - layers and _opened(phi, u[nx, j], nx - 1, j, pad):
        d = wp.max(d, -(float(nx - i) - 0.5) * dx)
    if j < layers and _opened(phi, v[i, 0], i, 0, pad):
        d = wp.max(d, -(float(j) + 0.5) * dx)
    if j >= ny - layers and _opened(phi, v[i, ny], i, ny - 1, pad):
        d = wp.max(d, -(float(ny - j) - 0.5) * dx)
    if d != c:
        phi[i + pad, j + pad] = d


@wp.kernel
def _open_wall_particles_kernel(x: wp.array(dtype=wp.vec2), phi: wp.array(dtype=float), phi_g: wp.array2d(dtype=float),
                                u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), pad: int, dx: float,
                                reach: float, valid: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32)):
    """Liquid particles within ``reach`` of an open wall face: phi_p = max(phi_p, signed distance to the wall)
    (negative inside the box; particles in the ghost layer behind the open face become air)."""
    k = wp.tid()
    if valid[k] == 0 or phi[k] >= 0.0:
        return
    nx = v.shape[0]
    ny = u.shape[1]
    q = x[k] - wp.vec2(float(pad) * dx, float(pad) * dx)
    i = wp.clamp(int(wp.floor(q[0] / dx)), 0, nx - 1)
    j = wp.clamp(int(wp.floor(q[1] / dx)), 0, ny - 1)
    lx = float(nx) * dx
    ly = float(ny) * dx
    c = phi[k]
    d = c
    if q[0] < reach and _opened(phi_g, u[0, j], 0, j, pad):
        d = wp.max(d, -q[0])
    if q[0] > lx - reach and _opened(phi_g, u[nx, j], nx - 1, j, pad):
        d = wp.max(d, q[0] - lx)
    if q[1] < reach and _opened(phi_g, v[i, 0], i, 0, pad):
        d = wp.max(d, -q[1])
    if q[1] > ly - reach and _opened(phi_g, v[i, ny], i, ny - 1, pad):
        d = wp.max(d, q[1] - ly)
    if d > c:
        inside = q[0] >= 0.0 and q[1] >= 0.0 and q[0] <= lx and q[1] <= ly
        if inside:
            d = wp.min(d, -1.0e-6 * dx)  # liquid inside the box stays liquid
        phi[k] = d  # ghost-region particles behind an open wall become air
        wp.atomic_add(count, 0, 1)


@wp.kernel
def _drop_outside_kernel(x: wp.array(dtype=wp.vec2), pad: int, dx: float, lx: float, ly: float,
                         valid: wp.array(dtype=wp.int32)):
    """Deactivate particles in the ghost layer (the ghost cells are rebuilt from the box every step)."""
    k = wp.tid()
    if valid[k] == 0:
        return
    q = x[k] - wp.vec2(float(pad) * dx, float(pad) * dx)
    if q[0] < 0.0 or q[1] < 0.0 or q[0] > lx or q[1] > ly:
        valid[k] = 0


@wp.kernel
def _interior_kernel(src: wp.array2d(dtype=float), pad: int, dst: wp.array2d(dtype=float)):
    i, j = wp.tid()
    dst[i, j] = src[i + pad, j + pad]


@wp.func
def _at(phi: wp.array2d(dtype=float), I: int, J: int) -> float:
    return phi[wp.clamp(I, 0, phi.shape[0] - 1), wp.clamp(J, 0, phi.shape[1] - 1)]


@wp.kernel
def _mark_films_kernel(phi: wp.array2d(dtype=float), u: wp.array2d(dtype=float), v: wp.array2d(dtype=float),
                       pad: int, max_width: float, deep: float, mark: wp.array2d(dtype=wp.int32)):
    """Mark air cells holding a squeezed sub-cell air film between two liquid neighbours (see the module docstring).

    ``deep > 0``: a cell squeezed along one axis whose neighbours along the other axis (across the film) include
    air deeper than ``deep`` is a notch open to the air, not a film, and is left alone."""
    i, j = wp.tid()
    nx = mark.shape[0]
    ny = mark.shape[1]
    I = i + pad
    J = j + pad
    c = phi[I, J]
    m = int(0)
    if c >= 0.0 and i > 0 and i < nx - 1:
        a = phi[I - 1, J]
        b = phi[I + 1, J]
        if a < 0.0 and b < 0.0:
            if c / (c - a) + c / (c - b) < max_width and u[i + 1, j] - u[i, j] < 0.0:
                if deep <= 0.0 or wp.max(_at(phi, I, J - 1), _at(phi, I, J + 1)) <= deep:
                    m = 1
    if c >= 0.0 and j > 0 and j < ny - 1:
        a = phi[I, J - 1]
        b = phi[I, J + 1]
        if a < 0.0 and b < 0.0:
            if c / (c - a) + c / (c - b) < max_width and v[i, j + 1] - v[i, j] < 0.0:
                if deep <= 0.0 or wp.max(_at(phi, I - 1, J), _at(phi, I + 1, J)) <= deep:
                    m = 1
    mark[i, j] = m


@wp.kernel
def _fill_films_kernel(mark: wp.array2d(dtype=wp.int32), pad: int, value: float, phi: wp.array2d(dtype=float)):
    i, j = wp.tid()
    if mark[i, j] == 1:
        phi[i + pad, j + pad] = -value


@wp.kernel
def _drop_film_particles_kernel(x: wp.array(dtype=wp.vec2), phi: wp.array(dtype=float),
                                mark: wp.array2d(dtype=wp.int32), inv_dx: float, pad: int,
                                valid: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32)):
    k = wp.tid()
    if valid[k] == 0 or phi[k] <= 0.0:
        return
    p = x[k]
    i = int(wp.floor(p[0] * inv_dx)) - pad
    j = int(wp.floor(p[1] * inv_dx)) - pad
    if i < 0 or j < 0 or i >= mark.shape[0] or j >= mark.shape[1]:
        return
    if mark[i, j] == 1:
        valid[k] = 0
        wp.atomic_add(count, 0, 1)


def _reflect(x: np.ndarray, length: float) -> np.ndarray:
    """Even reflection of coordinates outside [0, length] back into the box."""
    x = np.where(x < 0.0, -x, x)
    return np.where(x > length, 2.0 * length - x, x)


class PFMLevelSetBackend:
    """Free-surface adapter around :class:`pfmls.levelset.PFMLevelSet2D`, configured by ``cfg.pfm_levelset``
    (a :class:`PFMBackendConfig`).

    Args:
        cfg: the solver's :class:`~pfmls.fluid.FluidConfig` (grid and ``pfm_levelset``).
        sdf: vectorised signed distance ``sdf(x, y)`` of the initial liquid (negative inside).
        device: Warp device.
        dt_nominal: nominal time step of the solver (converts ``reinit_unit="nominal_steps"`` to time units).
    """

    def __init__(self, cfg, sdf, device=None, dt_nominal: float = 0.0):
        pc: PFMBackendConfig = cfg.pfm_levelset
        if not isinstance(pc, PFMBackendConfig):
            raise TypeError(f"FluidConfig.pfm_levelset must be a PFMBackendConfig, got {type(pc).__name__}")
        self.config = pc
        self.pad = pc.pad
        self.volume_guard = pc.volume_guard
        self.curvature_passes = pc.curvature_passes
        self.close_films = pc.close_films
        self.film_width, self.film_value, self.film_deep = pc.film_width, pc.film_value, pc.film_deep
        self.nx, self.ny, self.dx = cfg.nx, cfg.ny, cfg.dx
        self.device = wp.get_device(device)
        dev, p, dx = self.device, self.pad, cfg.dx
        lx, ly = cfg.nx * dx, cfg.ny * dx
        self.ls = PFMLevelSet2D(res=(cfg.nx + 2 * p, cfg.ny + 2 * p), dx=dx, device=dev,
                                **pc.levelset_kwargs(dt_nominal))
        if p > 0:
            self.ls.init_from_sdf(lambda x, y: sdf(_reflect(x - p * dx, lx), _reflect(y - p * dx, ly)))
        else:
            self.ls.init_from_sdf(sdf)
        self.rejects = 0
        self._scrubs = wp.zeros(2, dtype=wp.int32, device=dev)  # repaired grid cells, particles
        self._area = wp.zeros(2, dtype=float, device=dev)
        self._films = wp.zeros(1, dtype=wp.int32, device=dev)
        self._wall_caps = wp.zeros(1, dtype=wp.int32, device=dev)
        self._snap: dict | None = None
        self._fine: dict[int, wp.array2d] = {}
        shp = (cfg.nx, cfg.ny)
        self._phi = wp.zeros(shp, dtype=float, device=dev) if p > 0 else self.ls.phi_g
        self._ka, self._kb = (wp.zeros(shp, dtype=float, device=dev) for _ in range(2))
        self._kfixed = wp.zeros(shp, dtype=wp.int32, device=dev)
        self._fmark = wp.zeros(shp, dtype=wp.int32, device=dev)
        self._ux = wp.zeros((cfg.nx + 2 * p + 1, cfg.ny + 2 * p), dtype=float, device=dev)
        self._uy = wp.zeros((cfg.nx + 2 * p, cfg.ny + 2 * p + 1), dtype=float, device=dev)
        flow_cls = LinearMACVelocity2D if pc.velocity_sampling == "linear" else MACVelocity2D
        self._flow = flow_cls(self._ux, self._uy, dx)
        self._sync_interior()

    # ------------------------------------------------------------------------------------------- API
    @property
    def phi(self) -> wp.array2d:
        """Level set on the fluid grid (the interior of the padded level set)."""
        return self._phi

    def curvature_source(self) -> wp.array2d:
        """Field whose curvature enters the Laplace pressure.

        The composite ``phi_g`` mixes particle and grid values cell by cell, so its second differences are noisy
        at the sub-cell level (capillary oscillations then gain energy).  The curvature is therefore taken from a
        copy of the level set redistanced by closest-point projection onto its bicubic interpolant
        (:func:`pfmls.fluid.redistance.redistance_closest_point`), followed by ``curvature_passes`` 3x3
        binomial passes near the interface (they damp sub-cell wiggles of the zero set, which otherwise lengthen it
        and bias the mean curvature low).
        """
        dev, dim, dx = self.device, self._phi.shape, self.dx
        a = redistance_closest_point(self._phi, dx, CURVATURE_BAND * dx, CURVATURE_NEWTON_ITERS,
                                     CURVATURE_JACOBI_PASSES, self._ka, self._kb, self._kfixed, dev)
        b = self._kb if a is self._ka else self._ka
        for _ in range(self.curvature_passes):
            wp.launch(smooth_band_kernel, dim=dim, inputs=[a, a, CURVATURE_SMOOTH_BAND * dx], outputs=[b], device=dev)
            a, b = b, a
        return a

    def pre_step(self, u: wp.array2d, v: wp.array2d) -> None:
        """Device part before the level-set step: NaN scrub, padded velocity, ghost cells."""
        ls, dev = self.ls, self.device
        ls.scrub_nonfinite(self._scrubs)
        wp.launch(_pad_faces_kernel, dim=self._ux.shape, inputs=[u, self.pad, self._ux], device=dev)
        wp.launch(_pad_faces_kernel, dim=self._uy.shape, inputs=[v, self.pad, self._uy], device=dev)
        self._rebuild_ghosts(u, v)

    def step_core(self, dt: float) -> None:
        """The PFM-LS step in the padded velocity of the last :meth:`pre_step` (its own CUDA graphs), with the volume
        guard on long-reinitialisation steps (one host read per long reinitialisation)."""
        ls = self.ls
        if self.volume_guard <= 0.0 or not ls.reinit_due(dt)[0]:
            ls.step(dt, self._flow)
            return
        self._snap = ls.snapshot(self._snap)
        self._measure(0)
        ls.step(dt, self._flow)
        self._sync_interior()
        self._measure(1)
        a0, a1 = (float(x) for x in self._area.numpy())
        if not np.isfinite(a1) or abs(a1 - a0) > self.volume_guard * max(a0, self.dx * self.dx):
            self.rejects += 1
            ls.restore(self._snap)
            ls.step(dt, self._flow, skip_long_reinit=True)

    def post_step(self, u: wp.array2d, v: wp.array2d) -> None:
        """Device part after the level-set step: squeezed-film closure, open-wall capping, interior copy."""
        ls, dev = self.ls, self.device
        if self.close_films:
            wp.launch(_mark_films_kernel, dim=(self.nx, self.ny),
                      inputs=[ls.phi_g, u, v, self.pad, self.film_width, self.film_deep * self.dx],
                      outputs=[self._fmark], device=dev)
            wp.launch(_fill_films_kernel, dim=(self.nx, self.ny),
                      inputs=[self._fmark, self.pad, self.film_value * self.dx, ls.phi_g], device=dev)
            wp.launch(_drop_film_particles_kernel, dim=ls.num_slots,
                      inputs=[ls.x, ls.phi, self._fmark, 1.0 / self.dx, self.pad, ls.valid, self._films], device=dev)
        if self.pad > 0:  # the composite's Taylor extension uses the particles' old normals next to open walls
            wp.launch(_open_wall_distance_kernel, dim=(self.nx, self.ny),
                      inputs=[ls.phi_g, u, v, self.pad, OPEN_WALL_LAYERS, self.dx], device=dev)
        self._sync_interior()

    def advance(self, u: wp.array2d, v: wp.array2d, dt: float) -> None:
        """One interface step in the MAC velocity ``(u, v)`` of the fluid grid (:meth:`pre_step`,
        :meth:`step_core`, :meth:`post_step`; the solver calls the three separately, this is the standalone form)."""
        self.pre_step(u, v)
        self.step_core(dt)
        self.post_step(u, v)

    def fine_phi_device(self, factor: int = 4) -> wp.array2d:
        """Fine reconstruction of the box (the ghost layer cropped; device array, reused)."""
        fine = self.ls.reconstruct_device(factor)
        if self.pad == 0:
            return fine
        out = self._fine.get(factor)
        if out is None:
            out = self._fine[factor] = wp.zeros((factor * self.nx, factor * self.ny), dtype=float, device=self.device)
        wp.launch(_interior_kernel, dim=out.shape, inputs=[fine, factor * self.pad, out], device=self.device)
        return out

    def fine_phi(self, factor: int = 4) -> np.ndarray:
        """Host copy of :meth:`fine_phi_device`."""
        return self.fine_phi_device(factor).numpy().copy()

    def health(self) -> dict:
        """Hard-failure counters: repaired non-finite grid cells and particles."""
        scrubs = self._scrubs.numpy()
        return {"nan_scrubs": int(scrubs[0]), "nan_particles": int(scrubs[1])}

    def stats(self) -> dict:
        """Robustness counters: rejected reinitialisations, NaN repairs, particles dropped by the film closure and
        capped next to open walls, active level-set particles."""
        return {"redistance_rejects": self.rejects, **self.health(),
                "film_particles_dropped": int(self._films.numpy()[0]),
                "wall_particles_capped": int(self._wall_caps.numpy()[0]), "ls_particles": self.ls.num_valid()}

    def state_dict(self) -> dict:
        d = self.ls.state_dict()
        d.update(rejects=self.rejects, scrubs=self._scrubs.numpy().copy(), films=self._films.numpy().copy(),
                 wall_caps=self._wall_caps.numpy().copy())
        return d

    def load_state_dict(self, state: dict) -> None:
        self.ls.load_state_dict(state)
        self.rejects = int(state["rejects"])
        self._scrubs.assign(np.asarray(state["scrubs"], dtype=np.int32))
        self._films.assign(np.asarray(state["films"], dtype=np.int32))
        self._wall_caps.assign(np.asarray(state["wall_caps"], dtype=np.int32))
        self._sync_interior()

    # ------------------------------------------------------------------------------------------ helpers
    def _rebuild_ghosts(self, u: wp.array2d, v: wp.array2d) -> None:
        """Ghost cells for the velocity about to be used (wall faces open or closed in ``u, v``).

        Next to an open face the wall becomes an interface: the values of liquid cells and liquid particles within
        ``OPEN_WALL_LAYERS`` cells are capped by their distance to the wall, so the odd ghost values put the
        interface on the wall and it then moves with the liquid.
        """
        if self.pad == 0:
            return
        ls, dev, dx = self.ls, self.device, self.dx
        wp.launch(_drop_outside_kernel, dim=ls.num_slots,
                  inputs=[ls.x, self.pad, dx, self.nx * dx, self.ny * dx, ls.valid], device=dev)
        wp.launch(_open_wall_distance_kernel, dim=(self.nx, self.ny),
                  inputs=[ls.phi_g, u, v, self.pad, OPEN_WALL_LAYERS, dx], device=dev)
        wp.launch(_ghost_kernel, dim=ls.phi_g.shape, inputs=[ls.phi_g, u, v, self.pad], device=dev)
        wp.launch(_open_wall_particles_kernel, dim=ls.num_slots,
                  inputs=[ls.x, ls.phi, ls.phi_g, u, v, self.pad, dx, float(OPEN_WALL_LAYERS) * dx, ls.valid,
                          self._wall_caps], device=dev)

    def _sync_interior(self) -> None:
        """Copy the interior of the padded level set to :attr:`phi`."""
        if self.pad > 0:
            wp.launch(_interior_kernel, dim=(self.nx, self.ny), inputs=[self.ls.phi_g, self.pad, self._phi],
                      device=self.device)

    def _measure(self, slot: int) -> None:
        if slot == 0:
            self._area.zero_()
        wp.launch(_area_kernel, dim=(self.nx + 1, self.ny + 1), inputs=[self._phi, self.dx, slot, self._area],
                  device=self.device)
