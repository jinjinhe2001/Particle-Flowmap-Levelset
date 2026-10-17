"""3D velocity stored on a staggered (MAC) grid, sampled with the quadratic B-spline.

``ux`` has shape ``(nx + 1, ny, nz)`` (x-faces at ``(i, j + 0.5, k + 0.5) dx``), ``uy`` ``(nx, ny + 1, nz)``
and ``uz`` ``(nx, ny, nz + 1)``. The level set reads the arrays by reference, so the caller can update
them in place between steps (also inside captured CUDA graphs). The field is treated as constant
during a step (the time argument is ignored).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..core.kernels3d import mac_velocity_grad3
from .analytic3d import Flow3D, FlowParams3D
from .mac import check_face_array


@wp.func
def _mac3(x: wp.vec3, t: float, p: FlowParams3D, ux: wp.array3d(dtype=float), uy: wp.array3d(dtype=float),
          uz: wp.array3d(dtype=float), inv_dx: float):
    return mac_velocity_grad3(ux, uy, uz, x, inv_dx)


@wp.kernel
def _sample_mac3(ux: wp.array3d(dtype=float), uy: wp.array3d(dtype=float), uz: wp.array3d(dtype=float),
                 x: wp.array(dtype=wp.vec3), inv_dx: float, out: wp.array(dtype=wp.vec3),
                 grad: wp.array(dtype=wp.mat33)):
    k = wp.tid()
    u, g = mac_velocity_grad3(ux, uy, uz, x[k], inv_dx)
    out[k] = u
    grad[k] = g


class MACVelocity3D(Flow3D):
    """Wraps device face arrays ``ux (nx+1, ny, nz)``, ``uy (nx, ny+1, nz)``, ``uz (nx, ny, nz+1)``.

    The domain is ``[0, nx dx] x [0, ny dx] x [0, nz dx]`` with cell-centred level-set values. The arrays must be
    float32 Warp arrays (``TypeError`` otherwise). They must live on the level set's own grid (same ``nx, ny, nz,
    dx``) and on its device; both are checked on every :meth:`pfmls.levelset.pfmls3d.PFMLevelSet3D.step`
    (``ValueError`` otherwise).
    """

    func = _mac3

    def __init__(self, ux: wp.array, uy: wp.array, uz: wp.array, dx: float):
        for a, name in ((ux, "ux"), (uy, "uy"), (uz, "uz")):
            check_face_array(a, name)
        nx, ny, nz = uz.shape[0], uz.shape[1], uz.shape[2] - 1
        if tuple(ux.shape) != (nx + 1, ny, nz) or tuple(uy.shape) != (nx, ny + 1, nz):
            raise ValueError(f"inconsistent MAC shapes {ux.shape}, {uy.shape}, {uz.shape}")
        if not ux.device == uy.device == uz.device:
            raise ValueError(f"MAC arrays on different devices: {ux.device}, {uy.device}, {uz.device}")
        self.ux, self.uy, self.uz, self.dx = ux, uy, uz, float(dx)

    def mac_arrays(self, device):
        """The face arrays; ``ValueError`` if they do not live on ``device``."""
        if device is not None and wp.get_device(device) != self.ux.device:
            raise ValueError(f"MAC velocity arrays live on {self.ux.device}, the level set runs on "
                             f"{wp.get_device(device)}; allocate the faces on the level set's device")
        return self.ux, self.uy, self.uz

    def check_grid(self, nx: int, ny: int, nz: int, dx: float) -> None:
        expected = ((nx + 1, ny, nz), (nx, ny + 1, nz), (nx, ny, nz + 1))
        got = (tuple(self.ux.shape), tuple(self.uy.shape), tuple(self.uz.shape))
        if got != expected:
            raise ValueError(f"MAC arrays {got} do not match the level-set grid {nx}x{ny}x{nz} (expected {expected})")
        if abs(self.dx - dx) > 1.0e-6 * dx:
            raise ValueError(f"MAC velocity dx={self.dx} differs from the level-set dx={dx}")

    def sample(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Velocity ``(n, 3)`` and gradient ``(n, 3, 3)`` at points ``(n, 3)`` (host arrays)."""
        dev = self.ux.device
        xa = wp.array(np.asarray(pts, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3, device=dev)
        out = wp.zeros(len(xa), dtype=wp.vec3, device=dev)
        grad = wp.zeros(len(xa), dtype=wp.mat33, device=dev)
        wp.launch(_sample_mac3, dim=len(xa), inputs=[self.ux, self.uy, self.uz, xa, 1.0 / self.dx, out, grad],
                  device=dev)
        return out.numpy(), grad.numpy()

    def velocity_numpy(self, x, y, z, t):
        shape = np.shape(x)
        pts = np.stack([np.ravel(x), np.ravel(y), np.ravel(z)], axis=1)
        u, _ = self.sample(pts)
        return u[:, 0].reshape(shape), u[:, 1].reshape(shape), u[:, 2].reshape(shape)
