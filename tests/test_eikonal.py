"""Eikonal redistancing: distance accuracy, sign preservation, narrow-band mode."""

import numpy as np
import warp as wp

from pfmls.core.eikonal import Eikonal2D
from pfmls.levelset import sdf


def _grid(n):
    xs = (np.arange(n) + 0.5) / n
    return np.meshgrid(xs, xs, indexing="ij")


def test_redistance_circle(device):
    n, dx = 64, 1.0 / 64
    X, Y = _grid(n)
    exact = sdf.circle((0.45, 0.55), 0.23)(X, Y)
    distorted = exact * (1.0 + 0.8 * X) * (1.2 - 0.5 * np.sin(3 * Y))  # same zero set, |grad| != 1
    src = wp.array(distorted.astype(np.float32), dtype=float, device=device)
    out = wp.zeros_like(src)
    Eikonal2D(n, n, dx, device=device).redistance(src, out)
    phi = out.numpy()
    assert np.all(np.sign(phi) == np.sign(distorted))
    # first-order scheme with a linear (axis-wise) interface initialisation: sub-cell error near the
    # interface, slowly growing first-order error far away
    err = np.abs(phi - exact)
    near = np.abs(exact) < 3 * dx
    assert err[near].max() < 0.35 * dx
    assert err[near].mean() < 0.15 * dx
    assert err.max() < 1.0 * dx
    gx, gy = np.gradient(phi, dx)
    far = (np.abs(exact) > 3 * dx)[1:-1, 1:-1]
    assert np.abs(np.hypot(gx, gy)[1:-1, 1:-1][far] - 1.0).mean() < 0.05


def test_redistance_in_place_and_band(device):
    n, dx = 48, 1.0 / 48
    X, Y = _grid(n)
    exact = sdf.circle((0.5, 0.5), 0.1)(X, Y)
    marker = (3.0 * exact).astype(np.float32)
    full = wp.array(marker, dtype=float, device=device)
    Eikonal2D(n, n, dx, device=device).redistance(full, full)  # in place
    band = wp.array(marker, dtype=float, device=device)
    banded = wp.zeros_like(band)
    Eikonal2D(n, n, dx, device=device, iterations=6).redistance(band, banded)
    f, b = full.numpy(), banded.numpy()
    near = np.abs(exact) < 3 * dx
    np.testing.assert_allclose(b[near], f[near], atol=1e-6)
    # far cells that were never reached keep the input value
    far = np.abs(exact) > 12 * dx
    np.testing.assert_allclose(b[far], marker[far])


def test_no_interface_keeps_input(device):
    n = 16
    src = wp.array(np.full((n, n), 0.3, dtype=np.float32), dtype=float, device=device)
    out = wp.zeros_like(src)
    Eikonal2D(n, n, 1.0 / n, device=device).redistance(src, out)
    np.testing.assert_array_equal(out.numpy(), src.numpy())


def test_two_iterations_per_launch_match_single_launches(device):
    """The fused two-iteration Jacobi kernel equals two single iterations bitwise."""
    from pfmls.core import eikonal as E

    n, dx = 40, 1.0 / 40
    X, Y = _grid(n)
    rng = np.random.default_rng(5)
    phi = sdf.union(sdf.circle((0.4, 0.5), 0.2), sdf.circle((0.7, 0.3), 0.1))(X, Y)
    phi = (phi * (1.0 + 0.5 * rng.uniform(size=phi.shape))).astype(np.float32)
    src = wp.array(phi, dtype=float, device=device)
    T = wp.zeros_like(src)
    frozen = wp.zeros((n, n), dtype=wp.int32, device=device)
    wp.launch(E._init_surface, dim=(n, n), inputs=[src, T, frozen, dx], device=device)
    a, b, c = wp.clone(T), wp.zeros_like(T), wp.zeros_like(T)
    for _ in range(3):  # three rounds: frozen, converged and still-updating cells are all exercised
        wp.launch(E._jacobi, dim=(n, n), inputs=[a, b, frozen, dx], device=device)
        wp.launch(E._jacobi, dim=(n, n), inputs=[b, c, frozen, dx], device=device)
        d = wp.zeros_like(T)
        wp.launch(E._jacobi2, dim=(n, n), inputs=[a, d, frozen, dx], device=device)
        np.testing.assert_array_equal(d.numpy(), c.numpy())
        a = wp.clone(c)
    out_odd = wp.zeros_like(src)
    E.Eikonal2D(n, n, dx, device=device).redistance(src, out_odd, iterations=7)
    single = wp.clone(T)
    tmp = wp.zeros_like(T)
    for _ in range(7):
        wp.launch(E._jacobi, dim=(n, n), inputs=[single, tmp, frozen, dx], device=device)
        single, tmp = tmp, single
    ref = np.where(np.abs(single.numpy()) < 0.5 * E.BIG, single.numpy(), phi)
    np.testing.assert_array_equal(out_odd.numpy(), ref)
