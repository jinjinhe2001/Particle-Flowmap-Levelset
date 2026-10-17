"""Analytic 3D benchmark velocity fields with exact gradients.

* :class:`Enright3D` - Enright's divergence-free deformation field, time-reversed with period ``T``
  (the paper's "3D LeVeque sphere": ``u0 = 0.25``, ``T = 10``, sphere ``r = 0.15`` at ``(0.35, 0.35, 0.35)``)::

      u =  2 a(t) sin^2(pi x) sin(2 pi y) sin(2 pi z)
      v =   -a(t) sin(2 pi x) sin^2(pi y) sin(2 pi z)
      w =   -a(t) sin(2 pi x) sin(2 pi y) sin^2(pi z),      a(t) = u0 cos(pi t / T)

  evaluated in the phase form ``x' = x - 1/2`` (``sin^2(pi x) = cos^2(pi x')``,
  ``sin(2 pi y) sin(2 pi z) = sin(2 pi y') sin(2 pi z')``).
* :class:`Rotation3D` - rigid rotation ``u = omega x (x - c)`` about an axis through ``c``.
* :class:`Distorted3D` - differential rotation about the z axis through ``(cx, cy)``,
  ``u = u0 (y', -x', 0) / (r + r0)`` with ``r = |(x', y')|`` (the flow of the paper's 3D distorted-sphere test;
  no benchmark driver for it is included).

Every flow exposes a Warp function ``func(x, t, params, ux, uy, uz, inv_dx) -> (u, grad_u)`` with
``grad_u[a, b] = d u_a / d x_b``, a parameter vector ``params`` (:data:`FlowParams3D`, 8 floats) and the
MAC arrays it reads (analytic flows return 1x1x1 dummies). Kernels are specialised per ``func``;
time-dependent fields are evaluated at the true RK stage times.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

#: Per-step flow parameters uploaded to the device (8 floats).
FlowParams3D = wp.types.vector(length=8, dtype=wp.float32)


class Flow3D:
    """Base class: a velocity field usable by :class:`pfmls.levelset.pfmls3d.PFMLevelSet3D`.

    ``params()`` is uploaded to the device every step, so it may change between steps without
    invalidating captured CUDA graphs.
    """

    #: Warp function ``(x, t, params, ux, uy, uz, inv_dx) -> (vec3, mat33)``.
    func = None

    def params(self) -> tuple[float, ...]:
        return (0.0,) * 8

    def mac_arrays(self, device) -> tuple[wp.array, wp.array, wp.array]:
        """Face arrays read by ``func``, on ``device`` (1x1x1 dummies for analytic flows)."""
        key = str(wp.get_device(device))
        cache = Flow3D._dummy_cache
        if key not in cache:
            cache[key] = tuple(wp.zeros((1, 1, 1), dtype=float, device=device) for _ in range(3))
        return cache[key]

    def check_grid(self, nx: int, ny: int, nz: int, dx: float) -> None:
        """Raise ``ValueError`` if the flow cannot be sampled on the given grid."""

    def velocity_numpy(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float):
        """Velocity ``(u, v, w)`` on the host (used to build reference solutions)."""
        raise NotImplementedError

    _dummy_cache: dict = {}


@wp.func
def _enright(x: wp.vec3, t: float, p: FlowParams3D, ux: wp.array3d(dtype=float), uy: wp.array3d(dtype=float),
             uz: wp.array3d(dtype=float), inv_dx: float):
    a = p[1] * wp.cos(wp.pi * t / p[0])
    xs = x[0] - 0.5
    ys = x[1] - 0.5
    zs = x[2] - 0.5
    cx = wp.cos(wp.pi * xs)
    cy = wp.cos(wp.pi * ys)
    cz = wp.cos(wp.pi * zs)
    s2x = wp.sin(2.0 * wp.pi * xs)
    s2y = wp.sin(2.0 * wp.pi * ys)
    s2z = wp.sin(2.0 * wp.pi * zs)
    c2x = wp.cos(2.0 * wp.pi * xs)
    c2y = wp.cos(2.0 * wp.pi * ys)
    c2z = wp.cos(2.0 * wp.pi * zs)
    pa = wp.pi * a
    u = wp.vec3(2.0 * a * cx * cx * s2y * s2z, -a * s2x * cy * cy * s2z, -a * s2x * s2y * cz * cz)
    sss = s2x * s2y * s2z
    g = wp.mat33(
        -2.0 * pa * sss, 4.0 * pa * cx * cx * c2y * s2z, 4.0 * pa * cx * cx * s2y * c2z,
        -2.0 * pa * c2x * cy * cy * s2z, pa * sss, -2.0 * pa * s2x * cy * cy * c2z,
        -2.0 * pa * c2x * s2y * cz * cz, -2.0 * pa * s2x * c2y * cz * cz, pa * sss,
    )
    return u, g


@wp.func
def _rotation3(x: wp.vec3, t: float, p: FlowParams3D, ux: wp.array3d(dtype=float), uy: wp.array3d(dtype=float),
               uz: wp.array3d(dtype=float), inv_dx: float):
    w = wp.vec3(p[0], p[1], p[2])
    r = x - wp.vec3(p[3], p[4], p[5])
    g = wp.mat33(0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0)
    return wp.cross(w, r), g


@wp.func
def _distorted3(x: wp.vec3, t: float, p: FlowParams3D, ux: wp.array3d(dtype=float), uy: wp.array3d(dtype=float),
                uz: wp.array3d(dtype=float), inv_dx: float):
    u0 = p[0]
    xs = x[0] - p[1]
    ys = x[1] - p[2]
    r = wp.sqrt(xs * xs + ys * ys)
    if r < 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0), wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rr = r + p[3]
    k = u0 / rr
    dk_dx = -u0 * xs / (r * rr * rr)
    dk_dy = -u0 * ys / (r * rr * rr)
    g = wp.mat33(dk_dx * ys, dk_dy * ys + k, 0.0, -(dk_dx * xs + k), -dk_dy * xs, 0.0, 0.0, 0.0, 0.0)
    return wp.vec3(k * ys, -k * xs, 0.0), g


class Enright3D(Flow3D):
    """Enright's deformation field, reversing with period ``period`` (paper: ``u0 = 0.25``, ``T = 10``)."""

    func = _enright

    def __init__(self, period: float = 10.0, u0: float = 0.25):
        self.period = float(period)
        self.u0 = float(u0)

    def params(self):
        return (self.period, self.u0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def velocity_numpy(self, x, y, z, t):
        a = self.u0 * np.cos(np.pi * t / self.period)
        xs, ys, zs = x - 0.5, y - 0.5, z - 0.5
        s2x, s2y, s2z = np.sin(2 * np.pi * xs), np.sin(2 * np.pi * ys), np.sin(2 * np.pi * zs)
        return (2.0 * a * np.cos(np.pi * xs) ** 2 * s2y * s2z, -a * s2x * np.cos(np.pi * ys) ** 2 * s2z,
                -a * s2x * s2y * np.cos(np.pi * zs) ** 2)


class Rotation3D(Flow3D):
    """Rigid rotation with angular speed ``omega`` about ``axis`` through ``center``."""

    func = _rotation3

    def __init__(self, omega: float = math.pi / (157 * 0.02), axis=(0.0, 0.0, 1.0), center=(0.5, 0.5, 0.5)):
        ax = np.asarray(axis, dtype=np.float64)
        self.omega = float(omega)
        self.axis = tuple(float(v) for v in ax / np.linalg.norm(ax))
        self.center = tuple(float(v) for v in center)

    @property
    def omega_vector(self) -> np.ndarray:
        return self.omega * np.asarray(self.axis)

    def params(self):
        w = self.omega_vector
        return (w[0], w[1], w[2], *self.center, 0.0, 0.0)

    def velocity_numpy(self, x, y, z, t):
        w = self.omega_vector
        rx, ry, rz = x - self.center[0], y - self.center[1], z - self.center[2]
        return w[1] * rz - w[2] * ry, w[2] * rx - w[0] * rz, w[0] * ry - w[1] * rx

    def rotate_points(self, pts: np.ndarray, t: float) -> np.ndarray:
        """Exact positions at time ``t`` of points starting at ``pts`` (Rodrigues' formula)."""
        k = np.asarray(self.axis)
        th = self.omega * t
        r = np.asarray(pts, dtype=np.float64) - np.asarray(self.center)
        rot = (r * np.cos(th) + np.cross(k, r) * np.sin(th) + np.outer(r @ k, k) * (1.0 - np.cos(th)))
        return rot + np.asarray(self.center)


class Distorted3D(Flow3D):
    """Steady differential rotation about the z axis through ``center`` (paper: ``u0 = 0.4``, ``r0 = 0.01``; the
    offset keeps the velocity bounded on the axis, which 3D shapes may cross, unlike the 2D benchmark's circle)."""

    func = _distorted3

    def __init__(self, u0: float = 0.4, center=(0.5, 0.5), r0: float = 0.01):
        self.u0 = float(u0)
        self.center = (float(center[0]), float(center[1]))
        self.r0 = float(r0)

    def params(self):
        return (self.u0, self.center[0], self.center[1], self.r0, 0.0, 0.0, 0.0, 0.0)

    def velocity_numpy(self, x, y, z, t):
        xs, ys = x - self.center[0], y - self.center[1]
        k = self.u0 / np.maximum(np.sqrt(xs**2 + ys**2) + self.r0, 1e-300)
        return k * ys, -k * xs, np.zeros_like(np.asarray(z, dtype=np.float64))

    def rotate_points(self, pts: np.ndarray, t: float) -> np.ndarray:
        """Exact positions at time ``t``: each point turns clockwise with ``u0 / (r + r0)``."""
        pts = np.asarray(pts, dtype=np.float64)
        xs, ys = pts[:, 0] - self.center[0], pts[:, 1] - self.center[1]
        a = -self.u0 * t / (np.hypot(xs, ys) + self.r0)
        c, s = np.cos(a), np.sin(a)
        return np.stack([xs * c - ys * s + self.center[0], xs * s + ys * c + self.center[1], pts[:, 2]], axis=1)
