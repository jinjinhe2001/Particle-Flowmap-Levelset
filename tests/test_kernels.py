"""Unit tests of the shared Warp functions (B-splines, sampling, Hermite, MAC velocity)."""

import numpy as np
import pytest
import warp as wp

from pfmls.core.kernels import (N2, bilerp, bspline_sample_grad, dN2, hermite_grad, hermite_grad_raw,
                                hermite_value, mac_velocity_grad, safe_normalize, sgn)
from pfmls.flows import Distorted2D, LeVequeVortex2D, MACVelocity2D, Rotation2D
from pfmls.flows.analytic2d import _distorted, _leveque, _rotation


@wp.kernel
def _eval_bspline(x: wp.array(dtype=float), w: wp.array(dtype=float), dw: wp.array(dtype=float)):
    k = wp.tid()
    w[k] = N2(x[k])
    dw[k] = dN2(x[k])


@wp.kernel
def _eval_misc(v: wp.array(dtype=wp.vec2), s: wp.array(dtype=float), nv: wp.array(dtype=wp.vec2), ns: wp.array(dtype=float)):
    k = wp.tid()
    nv[k] = safe_normalize(v[k])
    ns[k] = sgn(s[k])


@wp.kernel
def _eval_sampling(phi: wp.array2d(dtype=float), g: wp.array2d(dtype=wp.vec2), q: wp.array(dtype=wp.vec2), h: float,
                   lin: wp.array(dtype=float), hv: wp.array(dtype=float), hg: wp.array(dtype=wp.vec2),
                   hn: wp.array(dtype=wp.vec2)):
    k = wp.tid()
    lin[k] = bilerp(phi, q[k])
    hv[k] = hermite_value(phi, g, q[k], h)
    hg[k] = hermite_grad_raw(phi, g, q[k], h)
    hn[k] = hermite_grad(phi, g, q[k], h)


@wp.kernel
def _eval_faces(f: wp.array2d(dtype=float), x: wp.array(dtype=wp.vec2), inv_dx: float, off: wp.vec2,
                val: wp.array(dtype=float), grad: wp.array(dtype=wp.vec2)):
    k = wp.tid()
    v, g = bspline_sample_grad(f, x[k], inv_dx, off)
    val[k] = v
    grad[k] = g


@wp.kernel
def _eval_mac(ux: wp.array2d(dtype=float), uy: wp.array2d(dtype=float), x: wp.array(dtype=wp.vec2), inv_dx: float,
              u: wp.array(dtype=wp.vec2), G: wp.array(dtype=wp.mat22)):
    k = wp.tid()
    a, b = mac_velocity_grad(ux, uy, x[k], inv_dx)
    u[k] = a
    G[k] = b


def test_bspline_partition_and_derivative(device):
    xs = np.linspace(-2.0, 2.0, 401).astype(np.float32)
    x = wp.array(xs, dtype=float, device=device)
    w = wp.zeros_like(x)
    dw = wp.zeros_like(x)
    wp.launch(_eval_bspline, dim=len(xs), inputs=[x, w, dw], device=device)
    w, dw = w.numpy(), dw.numpy()
    assert abs(w[200] - 0.75) < 1e-7 and w[0] == 0.0 and w[-1] == 0.0
    # partition of unity: sum_k N2(s - k) = 1
    s = np.linspace(0.0, 1.0, 11).astype(np.float32)
    shifted = wp.array(np.concatenate([s - k for k in (-2, -1, 0, 1, 2)]).astype(np.float32), dtype=float, device=device)
    ws, dws = wp.zeros_like(shifted), wp.zeros_like(shifted)
    wp.launch(_eval_bspline, dim=shifted.shape[0], inputs=[shifted, ws, dws], device=device)
    np.testing.assert_allclose(ws.numpy().reshape(5, -1).sum(axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(dws.numpy().reshape(5, -1).sum(axis=0), 0.0, atol=1e-5)
    # derivative vs central differences away from the kinks
    fd = np.gradient(w.astype(np.float64), xs.astype(np.float64))
    smooth = np.all(np.abs(np.abs(xs)[:, None] - np.array([0.5, 1.5])[None, :]) > 0.02, axis=1)
    np.testing.assert_allclose(dw[smooth], fd[smooth], atol=2e-3)


def test_safe_normalize_and_sign(device):
    v = wp.array(np.array([[0, 0], [3, 4], [-1e-3, 0], [0, -2]], dtype=np.float32), dtype=wp.vec2, device=device)
    s = wp.array(np.array([0.0, -0.0, -1e-30, 2.0], dtype=np.float32), dtype=float, device=device)
    nv = wp.zeros(4, dtype=wp.vec2, device=device)
    ns = wp.zeros(4, dtype=float, device=device)
    wp.launch(_eval_misc, dim=4, inputs=[v, s, nv, ns], device=device)
    np.testing.assert_allclose(nv.numpy(), [[0, 0], [0.6, 0.8], [-1, 0], [0, -1]], atol=1e-7)
    np.testing.assert_array_equal(ns.numpy(), [1.0, 1.0, -1.0, 1.0])


def test_bilinear_and_hermite_exactness(device):
    """Bilinear reproduces linear fields; Hermite (exact node gradients) reproduces quadratics."""
    n, h = 16, 1.0 / 16
    xs = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    f = lambda x, y: 0.3 + 0.7 * x - 0.4 * y + 1.3 * x * x - 0.9 * x * y + 0.6 * y * y  # noqa: E731
    fx = lambda x, y: 0.7 + 2.6 * x - 0.9 * y  # noqa: E731
    fy = lambda x, y: -0.4 - 0.9 * x + 1.2 * y  # noqa: E731
    phi = wp.array(f(X, Y).astype(np.float32), dtype=float, device=device)
    g = wp.array(np.stack([fx(X, Y), fy(X, Y)], axis=-1).astype(np.float32), dtype=wp.vec2, device=device)
    rng = np.random.default_rng(0)
    pts = rng.uniform(2.0 * h, 1.0 - 2.0 * h, size=(200, 2))  # stay away from the clamped boundary
    q = wp.array((pts / h - 0.5).astype(np.float32), dtype=wp.vec2, device=device)
    m = len(pts)
    lin, hv = wp.zeros(m, dtype=float, device=device), wp.zeros(m, dtype=float, device=device)
    hg, hn = wp.zeros(m, dtype=wp.vec2, device=device), wp.zeros(m, dtype=wp.vec2, device=device)
    wp.launch(_eval_sampling, dim=m, inputs=[phi, g, q, h, lin, hv, hg, hn], device=device)
    exact = f(pts[:, 0], pts[:, 1])
    grad = np.stack([fx(pts[:, 0], pts[:, 1]), fy(pts[:, 0], pts[:, 1])], axis=1)
    np.testing.assert_allclose(hv.numpy(), exact, atol=2e-5)
    np.testing.assert_allclose(hg.numpy(), grad, atol=2e-4)
    np.testing.assert_allclose(hn.numpy(), grad / np.linalg.norm(grad, axis=1, keepdims=True), atol=1e-4)
    # bilinear interpolation error of a quadratic is bounded by h^2 * max|f''| / 8 per direction
    assert np.abs(lin.numpy() - exact).max() < 2.0 * h * h

    # bilinear is exact for a linear field
    lin_field = wp.array((0.2 + 0.5 * X - 0.3 * Y).astype(np.float32), dtype=float, device=device)
    wp.launch(_eval_sampling, dim=m, inputs=[lin_field, g, q, h, lin, hv, hg, hn], device=device)
    np.testing.assert_allclose(lin.numpy(), 0.2 + 0.5 * pts[:, 0] - 0.3 * pts[:, 1], atol=1e-6)


def test_bspline_faces_and_mac_velocity(device):
    """Quadratic B-spline sampling reproduces linear face data (value and gradient) in the interior;
    a MAC-sampled rigid rotation matches the analytic field and its gradient."""
    n, dx = 32, 1.0 / 32
    rot = Rotation2D(omega=1.7, center=(0.5, 0.5))
    xf = np.arange(n + 1) * dx
    yc = (np.arange(n) + 0.5) * dx
    Xu, Yu = np.meshgrid(xf, yc, indexing="ij")
    Xv, Yv = np.meshgrid(yc, xf, indexing="ij")
    ux = wp.array(rot.velocity_numpy(Xu, Yu, 0.0)[0].astype(np.float32), dtype=float, device=device)
    uy = wp.array(rot.velocity_numpy(Xv, Yv, 0.0)[1].astype(np.float32), dtype=float, device=device)
    rng = np.random.default_rng(1)
    pts = rng.uniform(3 * dx, 1 - 3 * dx, size=(100, 2)).astype(np.float32)
    x = wp.array(pts, dtype=wp.vec2, device=device)
    val, grad = wp.zeros(100, dtype=float, device=device), wp.zeros(100, dtype=wp.vec2, device=device)
    wp.launch(_eval_faces, dim=100, inputs=[ux, x, 1.0 / dx, wp.vec2(0.0, 0.5), val, grad], device=device)
    np.testing.assert_allclose(val.numpy(), -1.7 * (pts[:, 1] - 0.5), atol=1e-5)
    np.testing.assert_allclose(grad.numpy(), np.tile([0.0, -1.7], (100, 1)), atol=1e-4)

    u, G = wp.zeros(100, dtype=wp.vec2, device=device), wp.zeros(100, dtype=wp.mat22, device=device)
    wp.launch(_eval_mac, dim=100, inputs=[ux, uy, x, 1.0 / dx, u, G], device=device)
    uu, vv = rot.velocity_numpy(pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64), 0.0)
    np.testing.assert_allclose(u.numpy(), np.stack([uu, vv], axis=1), atol=1e-5)
    np.testing.assert_allclose(G.numpy(), np.tile([[0.0, -1.7], [1.7, 0.0]], (100, 1, 1)), atol=1e-4)

    mac = MACVelocity2D(ux, uy, dx)
    mu, mv = mac.velocity_numpy(pts[:, 0], pts[:, 1], 0.0)
    np.testing.assert_allclose(mu, uu, atol=1e-5)
    np.testing.assert_allclose(mv, vv, atol=1e-5)


@wp.kernel
def _eval_flow2(x: wp.array(dtype=wp.vec2), t: float, p: wp.vec4, dummy: wp.array2d(dtype=float),
                u: wp.array(dtype=wp.vec2), G: wp.array(dtype=wp.mat22), which: int):
    k = wp.tid()
    a = wp.vec2()
    b = wp.mat22()
    if which == 0:
        a, b = _rotation(x[k], t, p, dummy, dummy, 1.0)
    elif which == 1:
        a, b = _leveque(x[k], t, p, dummy, dummy, 1.0)
    else:
        a, b = _distorted(x[k], t, p, dummy, dummy, 1.0)
    u[k] = a
    G[k] = b


def test_analytic_flows_gradients_and_divergence(device):
    """The hand-written Jacobians of the 2D benchmark flows match central differences of their velocities, and the
    flows are divergence-free."""
    rng = np.random.default_rng(4)
    pts = rng.uniform(0.1, 0.9, size=(64, 2))
    pts = pts[np.hypot(pts[:, 0] - 0.5, pts[:, 1] - 0.5) > 0.05]  # away from the Distorted flow's centre
    eps = 1e-5
    flows = (Rotation2D(omega=0.7, center=(0.4, 0.55)), LeVequeVortex2D(period=8.0), Distorted2D(u0=0.4))
    for which, flow in enumerate(flows):
        xa = wp.array(pts.astype(np.float32), dtype=wp.vec2, device=device)
        u = wp.zeros(len(pts), dtype=wp.vec2, device=device)
        G = wp.zeros(len(pts), dtype=wp.mat22, device=device)
        wp.launch(_eval_flow2, dim=len(pts), inputs=[xa, 1.3, wp.vec4(*flow.params()), flow.mac_arrays(device)[0],
                                                     u, G, which], device=device)
        u, G = u.numpy().astype(np.float64), G.numpy().astype(np.float64)
        np.testing.assert_allclose(u, np.stack(flow.velocity_numpy(*pts.T, 1.3), axis=1), atol=1e-5)
        assert np.abs(G[:, 0, 0] + G[:, 1, 1]).max() < 1e-4  # divergence free
        for b in range(2):  # central differences on the host (float64)
            e = np.zeros(2)
            e[b] = eps
            up = np.stack(flow.velocity_numpy(*(pts + e).T, 1.3), axis=1)
            um = np.stack(flow.velocity_numpy(*(pts - e).T, 1.3), axis=1)
            np.testing.assert_allclose(G[:, :, b], (up - um) / (2 * eps), atol=2e-3)


def test_mac_velocity_inputs_and_sample():
    """MACVelocity2D accepts float32 Warp arrays only (a clear TypeError otherwise) and samples on the host."""
    n = 16
    rot = Rotation2D(omega=1.0)
    with pytest.raises(TypeError, match="warp array"):
        MACVelocity2D(np.zeros((n + 1, n), np.float32), np.zeros((n, n + 1), np.float32), 1.0 / n)
    with pytest.raises(TypeError, match="float32"):
        MACVelocity2D(wp.zeros((n + 1, n), dtype=wp.float64, device="cpu"),
                      wp.zeros((n, n + 1), dtype=wp.float64, device="cpu"), 1.0 / n)
    dx = 1.0 / n
    xf, yc = np.arange(n + 1) * dx, (np.arange(n) + 0.5) * dx
    Xu, Yu = np.meshgrid(xf, yc, indexing="ij")
    Xv, Yv = np.meshgrid(yc, xf, indexing="ij")
    mac = MACVelocity2D(wp.array(rot.velocity_numpy(Xu, Yu, 0)[0].astype(np.float32), device="cpu"),
                        wp.array(rot.velocity_numpy(Xv, Yv, 0)[1].astype(np.float32), device="cpu"), dx)
    pts = np.array([[0.5, 0.5], [0.4, 0.6]])
    u, G = mac.sample(pts)
    np.testing.assert_allclose(u, np.stack(rot.velocity_numpy(*pts.T, 0.0), axis=1), atol=1e-5)
    np.testing.assert_allclose(G[0], [[0.0, -1.0], [1.0, 0.0]], atol=1e-4)
