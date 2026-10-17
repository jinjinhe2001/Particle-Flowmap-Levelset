"""PFM-LS: the Particle Flow Map Level Set method in NVIDIA Warp.

Public API::

    import pfmls
    ls = pfmls.PFMLevelSet2D(res=200, ppc=16, band=2.0, reinit_long=12, reinit_short=6)
    ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.75), 0.15))
    ls.step(0.02, pfmls.flows.LeVequeVortex2D())

    ls3 = pfmls.PFMLevelSet3D(res=64, ppc=27)
    ls3.init_from_sdf(pfmls.sdf3d.sphere((0.35, 0.35, 0.35), 0.15))
    ls3.step(0.01, pfmls.flows.Enright3D())

2D free-surface water with the PFM-LS interface: :mod:`pfmls.fluid` (``FluidConfig`` and ``FreeSurfaceSolver2D``
are also exported here); the 2D renderer and the video writer are in :mod:`pfmls.render`::

    solver = pfmls.fluid.get_scene("droplet_splash").make_solver()
    solver.advance_to(1.0)
"""

try:  # the installed distribution's version (pyproject.toml is the single source)
    from importlib.metadata import PackageNotFoundError, version as _version

    __version__ = _version("pfmls")
except PackageNotFoundError:  # a source checkout that is not installed: the version line of pyproject.toml
    import os as _os
    import re as _re

    _pyproject = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "pyproject.toml")
    try:
        with open(_pyproject) as _f:
            __version__ = _re.search(r'^version\s*=\s*"([^"]+)"', _f.read(), _re.M).group(1)
    except (OSError, AttributeError):
        __version__ = "0+unknown"

from . import flows, fluid, render
from .core import measure, measure3d
from .fluid import FluidConfig, FreeSurfaceSolver2D
from .levelset import PFMLevelSet2D, PFMLevelSet3D, PFMLS2DOptions, PFMLS3DOptions, sdf, sdf3d

__all__ = ["PFMLevelSet2D", "PFMLS2DOptions", "PFMLevelSet3D", "PFMLS3DOptions", "FluidConfig", "FreeSurfaceSolver2D",
           "flows", "fluid", "measure", "measure3d", "render", "sdf", "sdf3d", "__version__"]
