"""Velocity stored on a staggered (MAC) grid, sampled with the quadratic B-spline.

``ux`` has shape ``(nx + 1, ny)`` (x-faces at ``(i dx, (j + 0.5) dx)``), ``uy`` has shape
``(nx, ny + 1)`` (y-faces at ``((i + 0.5) dx, j dx)``). The level set reads the arrays by reference,
so a fluid solver can update them in place between steps (also inside captured CUDA graphs).
The field is treated as constant during a step (the time argument is ignored).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..core.kernels import mac_velocity_grad
from .analytic2d import Flow2D


def check_face_array(a, name: str) -> None:
    """``TypeError`` unless ``a`` is a float32 Warp array (NumPy arrays must be wrapped with ``wp.array``)."""
    if not isinstance(a, wp.array):
        raise TypeError(f"{name} must be a warp array (e.g. wp.array(u, dtype=float, device=ls.device)), got "
                        f"{type(a).__name__}")
    if a.dtype != wp.float32:
        raise TypeError(f"{name} must have dtype float32, got {getattr(a.dtype, '__name__', a.dtype)}")


@wp.func
def _mac(x: wp.vec2, t: float, p: wp.vec4, ux: wp.array2d(dtype=float), uy: wp.array2d(dtype=float), inv_dx: float):
    return mac_velocity_grad(ux, uy, x, inv_dx)


@wp.kernel
def _sample_mac(ux: wp.array2d(dtype=float), uy: wp.array2d(dtype=float), x: wp.array(dtype=wp.vec2),
                inv_dx: float, out: wp.array(dtype=wp.vec2), grad: wp.array(dtype=wp.mat22)):
    k = wp.tid()
    u, g = mac_velocity_grad(ux, uy, x[k], inv_dx)
    out[k] = u
    grad[k] = g


class MACVelocity2D(Flow2D):
    """Wraps device face arrays ``ux (nx+1, ny)`` and ``uy (nx, ny+1)`` with cell size ``dx``.

    The domain is ``[0, nx dx] x [0, ny dx]`` with cell-centred level-set values; ``ux[i, j]`` sits at
    ``(i dx, (j + 0.5) dx)`` and ``uy[i, j]`` at ``((i + 0.5) dx, j dx)``. The arrays must be float32 Warp arrays
    (``TypeError`` otherwise). They must live on the level set's own grid (same ``nx, ny, dx``) and on its device;
    both are checked on every :meth:`pfmls.PFMLevelSet2D.step` (``ValueError`` otherwise).
    """

    func = _mac

    def __init__(self, ux: wp.array, uy: wp.array, dx: float):
        check_face_array(ux, "ux")
        check_face_array(uy, "uy")
        if ux.shape[0] != uy.shape[0] + 1 or uy.shape[1] != ux.shape[1] + 1:
            raise ValueError(f"inconsistent MAC shapes {ux.shape} and {uy.shape}")
        if ux.device != uy.device:
            raise ValueError(f"MAC arrays on different devices: ux on {ux.device}, uy on {uy.device}")
        self.ux, self.uy, self.dx = ux, uy, float(dx)

    def mac_arrays(self, device):
        """The face arrays; ``ValueError`` if they do not live on ``device``."""
        if device is not None and wp.get_device(device) != self.ux.device:
            raise ValueError(f"MAC velocity arrays live on {self.ux.device}, the level set runs on "
                             f"{wp.get_device(device)}; allocate the faces on the level set's device")
        return self.ux, self.uy

    def check_grid(self, nx: int, ny: int, dx: float) -> None:
        if tuple(self.ux.shape) != (nx + 1, ny) or tuple(self.uy.shape) != (nx, ny + 1):
            raise ValueError(f"MAC arrays {tuple(self.ux.shape)} / {tuple(self.uy.shape)} do not match the "
                             f"level-set grid {nx}x{ny} (expected {(nx + 1, ny)} / {(nx, ny + 1)})")
        if abs(self.dx - dx) > 1.0e-6 * dx:
            raise ValueError(f"MAC velocity dx={self.dx} differs from the level-set dx={dx}")

    def sample(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Velocity ``(n, 2)`` and gradient ``(n, 2, 2)`` (``G[a, b] = d u_a / d x_b``) at points ``(n, 2)``
        (host arrays)."""
        dev = self.ux.device
        xa = wp.array(np.asarray(pts, dtype=np.float32).reshape(-1, 2), dtype=wp.vec2, device=dev)
        out = wp.zeros(len(xa), dtype=wp.vec2, device=dev)
        grad = wp.zeros(len(xa), dtype=wp.mat22, device=dev)
        wp.launch(_sample_mac, dim=len(xa), inputs=[self.ux, self.uy, xa, 1.0 / self.dx, out, grad], device=dev)
        return out.numpy(), grad.numpy()

    def velocity_numpy(self, x, y, t):
        u, _ = self.sample(np.stack([np.ravel(x), np.ravel(y)], axis=1))
        return u[:, 0].reshape(np.shape(x)), u[:, 1].reshape(np.shape(x))

