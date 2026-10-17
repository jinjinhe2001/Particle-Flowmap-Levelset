"""Velocity fields for level-set transport.

Every 2D flow exposes a Warp function ``func(x, t, params, ux, uy, inv_dx) -> (u, grad_u)`` with
``grad_u[a, b] = d u_a / d x_b``, a parameter vector ``params`` (``wp.vec4``) and the MAC arrays
``ux, uy`` it reads (analytic flows return 1x1 dummies). Kernels are specialised per ``func``.
The 3D flows follow the same pattern with ``(ux, uy, uz)`` and an 8-component parameter vector.
"""

from .analytic2d import Distorted2D, Flow2D, LeVequeVortex2D, Rotation2D
from .analytic3d import Distorted3D, Enright3D, Flow3D, Rotation3D
from .mac import MACVelocity2D
from .mac3d import MACVelocity3D

__all__ = ["Flow2D", "Rotation2D", "LeVequeVortex2D", "Distorted2D", "MACVelocity2D",
           "Flow3D", "Enright3D", "Rotation3D", "Distorted3D", "MACVelocity3D"]
