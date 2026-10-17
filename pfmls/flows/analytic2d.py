"""Analytic 2D benchmark velocity fields with exact gradients.

* :class:`Rotation2D` - rigid rotation about a centre (Zalesak's disk: ``omega = pi / 3.14``).
* :class:`LeVequeVortex2D` - LeVeque's single vortex, time-reversed with period ``T``:
  ``u = a(t) (cos^2(pi x') sin(2 pi y'), -sin(2 pi x') cos^2(pi y'))``, ``a = u0 cos(pi t / T)``,
  ``x' = x - 0.5``.
* :class:`Distorted2D` - differential rotation ``u = u0 (y', -x') / (r + r0)``, ``r = |x'|``.

Gradients are exact; time-dependent fields are evaluated at the true RK stage times.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp


class Flow2D:
    """Base class: a velocity field usable by :class:`pfmls.PFMLevelSet2D`.

    ``params()`` is uploaded to the device every step, so it may change between steps (e.g. a
    ramped amplitude) without invalidating captured CUDA graphs.
    """

    #: Warp function ``(x, t, params, ux, uy, inv_dx) -> (vec2, mat22)``.
    func = None

    def params(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, 0.0, 0.0)

    def mac_arrays(self, device) -> tuple[wp.array, wp.array]:
        """Face arrays read by ``func``, on ``device`` (1x1 dummies for analytic flows)."""
        key = str(wp.get_device(device))
        cache = Flow2D._dummy_cache
        if key not in cache:
            cache[key] = (wp.zeros((1, 1), dtype=float, device=device), wp.zeros((1, 1), dtype=float, device=device))
        return cache[key]

    def check_grid(self, nx: int, ny: int, dx: float) -> None:
        """Raise ``ValueError`` if the flow cannot be sampled on an ``nx x ny`` grid of spacing ``dx``."""

    def velocity_numpy(self, x: np.ndarray, y: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Velocity on the host (used to build reference solutions)."""
        raise NotImplementedError

    _dummy_cache: dict = {}


@wp.func
def _rotation(x: wp.vec2, t: float, p: wp.vec4, ux: wp.array2d(dtype=float), uy: wp.array2d(dtype=float), inv_dx: float):
    w = p[0]
    u = wp.vec2(-w * (x[1] - p[2]), w * (x[0] - p[1]))
    return u, wp.mat22(0.0, -w, w, 0.0)


@wp.func
def _leveque(x: wp.vec2, t: float, p: wp.vec4, ux: wp.array2d(dtype=float), uy: wp.array2d(dtype=float), inv_dx: float):
    a = p[1] * wp.cos(wp.pi * t / p[0])
    xs = x[0] - 0.5
    ys = x[1] - 0.5
    cx = wp.cos(wp.pi * xs)
    sx = wp.sin(wp.pi * xs)
    cy = wp.cos(wp.pi * ys)
    sy = wp.sin(wp.pi * ys)
    s2x = wp.sin(2.0 * wp.pi * xs)
    c2x = wp.cos(2.0 * wp.pi * xs)
    s2y = wp.sin(2.0 * wp.pi * ys)
    c2y = wp.cos(2.0 * wp.pi * ys)
    u = wp.vec2(a * cx * cx * s2y, -a * s2x * cy * cy)
    g = wp.mat22(
        -2.0 * wp.pi * a * cx * sx * s2y,
        2.0 * wp.pi * a * cx * cx * c2y,
        -2.0 * wp.pi * a * c2x * cy * cy,
        2.0 * wp.pi * a * s2x * cy * sy,
    )
    return u, g


@wp.func
def _distorted(x: wp.vec2, t: float, p: wp.vec4, ux: wp.array2d(dtype=float), uy: wp.array2d(dtype=float), inv_dx: float):
    u0 = p[0]
    xs = x[0] - p[1]
    ys = x[1] - p[2]
    r = wp.sqrt(xs * xs + ys * ys)
    if r < 1.0e-12:
        return wp.vec2(0.0, 0.0), wp.mat22(0.0, 0.0, 0.0, 0.0)
    rr = r + p[3]
    k = u0 / rr
    # d(1/(r+r0))/dx = -(x'/r) / rr^2
    dk_dx = -u0 * xs / (r * rr * rr)
    dk_dy = -u0 * ys / (r * rr * rr)
    u = wp.vec2(k * ys, -k * xs)
    g = wp.mat22(dk_dx * ys, dk_dy * ys + k, -(dk_dx * xs + k), -dk_dy * xs)
    return u, g


class Rotation2D(Flow2D):
    """Rigid counter-clockwise rotation ``u = omega (-(y - cy), x - cx)``.

    The default ``omega = pi / (157 * 0.02)`` is Zalesak's disk: one revolution every 314 steps
    of ``dt = 0.02``.
    """

    func = _rotation

    def __init__(self, omega: float = math.pi / (157 * 0.02), center: tuple[float, float] = (0.5, 0.5)):
        self.omega = float(omega)
        self.center = (float(center[0]), float(center[1]))

    def params(self):
        return (self.omega, self.center[0], self.center[1], 0.0)

    def velocity_numpy(self, x, y, t):
        return -self.omega * (y - self.center[1]), self.omega * (x - self.center[0])


class LeVequeVortex2D(Flow2D):
    """LeVeque's single vortex in ``[0,1]^2``, reversing with period ``period``."""

    func = _leveque

    def __init__(self, period: float = 12.56, u0: float = 1.0):
        self.period = float(period)
        self.u0 = float(u0)

    def params(self):
        return (self.period, self.u0, 0.0, 0.0)

    def velocity_numpy(self, x, y, t):
        a = self.u0 * np.cos(np.pi * t / self.period)
        xs, ys = x - 0.5, y - 0.5
        return (a * np.cos(np.pi * xs) ** 2 * np.sin(2 * np.pi * ys),
                -a * np.sin(2 * np.pi * xs) * np.cos(np.pi * ys) ** 2)


class Distorted2D(Flow2D):
    """Steady differential rotation ``u = u0 (y - cy, -(x - cx)) / (r + r0)`` (clockwise).

    ``r0 = 0`` is the 2D benchmark (the circle never reaches the centre); :class:`pfmls.flows.Distorted3D` defaults to
    ``r0 = 0.01``, which keeps the velocity bounded on its rotation axis.
    """

    func = _distorted

    def __init__(self, u0: float = 0.4, center: tuple[float, float] = (0.5, 0.5), r0: float = 0.0):
        self.u0 = float(u0)
        self.center = (float(center[0]), float(center[1]))
        self.r0 = float(r0)

    def params(self):
        return (self.u0, self.center[0], self.center[1], self.r0)

    def velocity_numpy(self, x, y, t):
        xs, ys = x - self.center[0], y - self.center[1]
        k = self.u0 / np.maximum(np.sqrt(xs**2 + ys**2) + self.r0, 1e-300)
        return k * ys, -k * xs
