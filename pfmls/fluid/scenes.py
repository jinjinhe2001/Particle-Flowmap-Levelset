"""Scene presets for the 2D free-surface solver.

A preset returns a :class:`Scene`: solver overrides, the initial signed distance (negative in the liquid, built from
:mod:`pfmls.levelset.sdf`), a default duration, the time between video frames and the probes recorded at every
frame.  Lengths are in units of the paper's unit box, ``rho = 1`` and ``g = 1``.
``scene.make_solver()`` builds a :class:`~pfmls.fluid.FreeSurfaceSolver2D` for a preset.

=====================  ================================================================================================
droplet_splash         drop r=0.08 at (0.5, 0.75) falling into a pool of depth 0.3525 (the paper's setup), 200^2
dam_break              column a x 2a (a = 0.25) collapsing in the unit box (Martin & Moyce, n^2 = 2), sigma=1e-3, 400^2
=====================  ================================================================================================

``res`` is the number of cells per unit length; the scenes step with ``dt = 0.1 / res`` (:func:`gravity_dt`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from ..levelset import sdf
from ..levelset.sdf import SDF
from .freesurface2d import FluidConfig, FreeSurfaceSolver2D


def sdf_below(h: float) -> SDF:
    """Liquid below the line y = h (a pool of depth h)."""
    return sdf.half_plane((0.0, h), (0.0, 1.0))


@dataclass
class Scene:
    """A preset: initial liquid, solver overrides, duration, frame interval and probes.

    ``overrides`` are :class:`~pfmls.fluid.FluidConfig` fields (grid, sigma, gravity, dt, transport, ...);
    ``probes`` map a name to ``fn(phi, dx) -> float``, evaluated by the example runner on the host copy of the grid
    level set at every frame and written to ``frames.csv``.
    """

    #: name (output directory and file names)
    name: str
    #: initial signed distance ``sdf(x, y)``, negative in the liquid
    sdf: Callable
    #: default end time and time between frames
    t_end: float
    frame_dt: float
    overrides: dict = field(default_factory=dict)
    #: one-line description printed by the runner
    description: str = ""
    #: probes ``name -> fn(phi, dx)`` recorded at every frame
    probes: dict[str, Callable] = field(default_factory=dict)
    #: renderer: the speed drawn with the full foam tint
    speed_scale: float = 1.0

    def make_config(self, **overrides) -> FluidConfig:
        """The scene's :class:`~pfmls.fluid.FluidConfig`; ``overrides`` replace single fields."""
        return replace(FluidConfig(**self.overrides), **overrides)

    def make_solver(self, device=None, **overrides) -> FreeSurfaceSolver2D:
        """A solver for this scene (liquid at rest) on ``device``; ``overrides`` as in :meth:`make_config`."""
        return FreeSurfaceSolver2D(self.make_config(**overrides), self.sdf, device=device)


def front_position(phi: np.ndarray, dx: float, row: int = 2) -> float:
    """x of the rightmost liquid/air crossing on cell row ``row`` of the grid level set (sub-cell, linear; the
    right wall if the liquid touches it, NaN for a dry row)."""
    r = phi[:, row]
    liq = np.nonzero(r < 0)[0]
    if len(liq) == 0:
        return float("nan")
    i = liq[-1]
    if i + 1 >= len(r):
        return len(r) * dx
    return (i + 0.5 + r[i] / (r[i] - r[i + 1])) * dx


def gravity_dt(res: int) -> float:
    """Nominal time step ``0.1 dx`` (``dx = 1 / res``; 5e-4 at 200 cells per unit, the paper's step).

    Keeping ``dt / dx`` fixed keeps the number of steps per cell crossed fixed, and with it the rate at which the
    squeezed-film closure (every step) and the PFM-LS reinitialisations (every 12 / 6 nominal steps) act on a moving
    interface."""
    return 0.1 / res


#: Droplet splash: drop radius, release height of its centre and pool depth (the paper's setup).
SPLASH_RADIUS, SPLASH_HEIGHT, SPLASH_POOL = 0.08, 0.75, 0.3525


def splash_weber(sigma: float, rho: float = 1.0, g: float = 1.0) -> float:
    """Weber number rho U^2 D / sigma of the droplet splash: impact speed U^2 = 2 g (fall height of the drop's
    lowest point), diameter D = 0.16 (We ~ 508 at sigma = 2e-4)."""
    u2 = 2.0 * g * (SPLASH_HEIGHT - SPLASH_RADIUS - SPLASH_POOL)
    return rho * u2 * 2.0 * SPLASH_RADIUS / sigma if sigma > 0 else float("inf")


def droplet_splash(sigma: float = 2.0e-4, res: int = 200, t_end: float = 3.75) -> Scene:
    """The paper's splash: drop and pool in the unit box (``res`` x ``res`` cells)."""
    return Scene(
        name="droplet_splash",
        sdf=sdf_below(SPLASH_POOL) | sdf.circle((0.5, SPLASH_HEIGHT), SPLASH_RADIUS),
        t_end=t_end, frame_dt=0.01,
        overrides=dict(nx=res, ny=res, sigma=sigma, gravity=(0.0, -1.0), dt=gravity_dt(res)),
        description=(f"Drop r={SPLASH_RADIUS:g} at (0.5,{SPLASH_HEIGHT:g}) falls (g=1) into a pool of depth "
                     f"{SPLASH_POOL:g}; sigma={sigma:g}, We~{splash_weber(sigma):.0f}."),
    )


def dam_break(res: int = 400, t_end: float = 6.0, sigma: float = 1.0e-3, a: float = 0.25) -> Scene:
    """Martin & Moyce column ``a x 2a`` at the left wall of the unit box (``res`` x ``res`` cells).

    ``sigma = 1e-3`` (Bond number rho g L^2 / sigma = 1000, water in a tank about 9 cm wide) rounds the tips of the
    thin run-up jets into rims and drops.  The default duration (t = 6) runs past the plunging breaker on the right
    wall and the splash of the returning wave on the left wall to the first low pool.  The probe ``front`` is the
    position of the liquid front on the third cell row above the floor."""
    return Scene(
        name="dam_break",
        sdf=sdf.rectangle((-1.0, -1.0), (a, 2.0 * a)),
        t_end=t_end, frame_dt=0.01,
        overrides=dict(nx=res, ny=res, sigma=sigma, gravity=(0.0, -1.0), dt=gravity_dt(res)),
        probes={"front": lambda phi, dx: front_position(phi, dx, row=2)},
        description=f"Water column a x 2a (a={a:g}) released in the unit box (Martin & Moyce n^2=2), sigma={sigma:g}.",
        speed_scale=1.2,
    )


SCENES: dict[str, Callable[..., Scene]] = {
    "droplet_splash": droplet_splash,
    "dam_break": dam_break,
}


def get_scene(name: str, **kwargs) -> Scene:
    """The preset ``name`` (a key of :data:`SCENES`) built with ``kwargs``."""
    if name not in SCENES:
        raise KeyError(f"unknown scene '{name}'; available: {', '.join(SCENES)}")
    return SCENES[name](**kwargs)
