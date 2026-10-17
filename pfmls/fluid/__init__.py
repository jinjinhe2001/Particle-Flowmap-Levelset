"""2D incompressible free-surface flow with the PFM-LS interface (MAC grid, ghost-fluid pressure, surface tension).

The interface is :class:`pfmls.levelset.PFMLevelSet2D`, wrapped by :class:`PFMLevelSetBackend` and configured by
:class:`PFMBackendConfig` (the ``pfm_levelset`` field of :class:`FluidConfig`).  The velocity is transported by
APIC (``velocity="apic"``, the default) or semi-Lagrangian advection (``"sl"``).
"""

from .freesurface2d import FluidConfig, FreeSurfaceSolver2D, failure_reason, make_transport
from .pfm_backend import PFMBackendConfig, PFMLevelSetBackend
from .pressure import PressureConfig, PressureSolver
from .scenes import SCENES, Scene, get_scene

__all__ = [
    "FluidConfig", "FreeSurfaceSolver2D", "PFMBackendConfig", "PFMLevelSetBackend", "PressureConfig", "PressureSolver",
    "SCENES", "Scene", "failure_reason", "get_scene", "make_transport",
]
