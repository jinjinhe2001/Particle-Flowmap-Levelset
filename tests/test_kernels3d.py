"""Unit tests of the 3D building blocks: sampling, Hermite, B-splines, flows, eikonal, cell lists, measurement."""

import math

import numpy as np
import warp as wp

from pfmls.core import measure3d
from pfmls.core.celllist3d import CellList3D
from pfmls.core.eikonal3d import Eikonal3D
from pfmls.core.grid3d import Grid3D
from pfmls.core.kernels3d import bspline_sample_grad3, hermite3, hermite_cross_terms, safe_normalize3, trilerp
from pfmls.flows.analytic3d import Distorted3D, Enright3D, FlowParams3D, Rotation3D, _distorted3, _enright, _rotation3
from pfmls.flows.mac3d import MACVelocity3D
from pfmls.levelset import sdf3d


@wp.kernel
def _eval_interp(phi: wp.array3d(dtype=float), G: wp.array3d(dtype=wp.vec3), X: wp.array3d(dtype=wp.vec4),
                 q: wp.array(dtype=wp.vec3), h: float, lin: wp.array(dtype=float), hv: wp.array(dtype=float),
                 hg: wp.array(dtype=wp.vec3)):
    k = wp.tid()
    lin[k] = trilerp(phi, q[k])
    v, g = hermite3(phi, G, X, q[k], h)
    hv[k] = v
    hg[k] = g


@wp.kernel
def _eval_bspline3(f: wp.array3d(dtype=float), x: wp.array(dtype=wp.vec3), inv_dx: float, off: wp.vec3,
                   val: wp.array(dtype=float), grad: wp.array(dtype=wp.vec3)):
    k = wp.tid()
    v, g = bspline_sample_grad3(f, x[k], inv_dx, off)
    val[k] = v
    grad[k] = g


def _nodes(n, h, offset=(0.5, 0.5, 0.5), shape=None):
    shape = shape or (n, n, n)
    axes = [(np.arange(s) + o) * h for s, o in zip(shape, offset)]
    return np.meshgrid(*axes, indexing="ij")


def test_trilinear_and_hermite_reproduce_polynomials(device):
    """Trilinear is exact for linear fields; tricubic Hermite (with FD mixed derivatives) for quadratics."""
    n, h = 8, 0.125
    X, Y, Z = _nodes(n, h)
    a, b = np.array([0.3, -0.7, 0.2]), np.array([[0.5, 0.2, -0.1], [0.2, -0.4, 0.3], [-0.1, 0.3, 0.6]])

    def quad(x, y, z):
        p = np.stack([x, y, z], axis=-1)
        return p @ a + np.einsum("...i,ij,...j->...", p, b, p), p @ (b + b.T) + a

    phi, grad = quad(X, Y, Z)
    lin_field = (np.stack([X, Y, Z], axis=-1) @ a).astype(np.float32)
    rng = np.random.default_rng(0)
    q = rng.uniform(1.0, n - 2.0, size=(200, 3)).astype(np.float32)  # interior (central differences exact)
    xq = (q + 0.5) * h
    args = dict(dtype=float, device=device)
    G = wp.array(grad.astype(np.float32), dtype=wp.vec3, device=device)
    Xc = wp.zeros((n, n, n), dtype=wp.vec4, device=device)
    wp.launch(hermite_cross_terms, dim=(n, n, n), inputs=[G, Xc, h], device=device)
    out = [wp.zeros(len(q), **args), wp.zeros(len(q), **args), wp.zeros(len(q), dtype=wp.vec3, device=device)]
    for field, check in ((lin_field, "lin"), (phi.astype(np.float32), "herm")):
        wp.launch(_eval_interp, dim=len(q), inputs=[wp.array(field, **args), G, Xc,
                                                    wp.array(q, dtype=wp.vec3, device=device), h, *out], device=device)
        if check == "lin":
            np.testing.assert_allclose(out[0].numpy(), xq @ a, atol=2e-6)
        else:
            ev, eg = quad(xq[:, 0], xq[:, 1], xq[:, 2])
            np.testing.assert_allclose(out[1].numpy(), ev, atol=2e-5)
            np.testing.assert_allclose(out[2].numpy(), eg, atol=2e-4)


def test_bspline_faces_reproduce_linear(device):
    n, h = 10, 0.1
    shape = (n + 1, n, n)
    X, Y, Z = _nodes(n, h, offset=(0.0, 0.5, 0.5), shape=shape)
    c = np.array([0.4, -1.1, 0.7])
    f = (0.3 + c[0] * X + c[1] * Y + c[2] * Z).astype(np.float32)
    rng = np.random.default_rng(1)
    x = rng.uniform(0.25, 0.75, size=(100, 3)).astype(np.float32)
    val = wp.zeros(len(x), dtype=float, device=device)
    grad = wp.zeros(len(x), dtype=wp.vec3, device=device)
    wp.launch(_eval_bspline3, dim=len(x), inputs=[wp.array(f, dtype=float, device=device),
                                                 wp.array(x, dtype=wp.vec3, device=device), 1.0 / h,
                                                 wp.vec3(0.0, 0.5, 0.5), val, grad], device=device)
    np.testing.assert_allclose(val.numpy(), 0.3 + x @ c, atol=1e-5)
    np.testing.assert_allclose(grad.numpy(), np.tile(c, (len(x), 1)), atol=1e-4)


def test_mac_velocity_rigid_rotation(device):
    """A linear (rigid-rotation) MAC field is sampled exactly, gradient included."""
    n, h = 12, 1.0 / 12
    rot = Rotation3D(omega=1.3, axis=(1.0, 2.0, 2.0), center=(0.5, 0.5, 0.5))
    faces = []
    for axis, shape in enumerate(((n + 1, n, n), (n, n + 1, n), (n, n, n + 1))):
        off = [0.5, 0.5, 0.5]
        off[axis] = 0.0
        X, Y, Z = _nodes(n, h, offset=off, shape=shape)
        faces.append(wp.array(rot.velocity_numpy(X, Y, Z, 0.0)[axis].astype(np.float32), dtype=float, device=device))
    mac = MACVelocity3D(*faces, h)
    rng = np.random.default_rng(2)
    pts = rng.uniform(0.3, 0.7, size=(50, 3))
    u, G = mac.sample(pts)
    np.testing.assert_allclose(u, np.stack(rot.velocity_numpy(pts[:, 0], pts[:, 1], pts[:, 2], 0.0), axis=1),
                               atol=2e-5)
    w = rot.omega_vector
    skew = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    np.testing.assert_allclose(G, np.broadcast_to(skew, G.shape), atol=2e-4)


@wp.kernel
def _eval_flow(x: wp.array(dtype=wp.vec3), t: float, p: FlowParams3D, d: wp.array3d(dtype=float),
               u: wp.array(dtype=wp.vec3), G: wp.array(dtype=wp.mat33), which: int):
    k = wp.tid()
    a = wp.vec3()
    b = wp.mat33()
    if which == 0:
        a, b = _enright(x[k], t, p, d, d, d, 1.0)
    elif which == 1:
        a, b = _rotation3(x[k], t, p, d, d, d, 1.0)
    else:
        a, b = _distorted3(x[k], t, p, d, d, d, 1.0)
    u[k] = a
    G[k] = b


def test_analytic_flows_gradients_and_divergence(device):
    rng = np.random.default_rng(3)
    pts = rng.uniform(0.1, 0.9, size=(64, 3))
    eps = 1e-4
    for which, flow in enumerate((Enright3D(), Rotation3D(omega=0.7, axis=(0, 1, 1)), Distorted3D())):
        def ev(p, t=1.3):
            xa = wp.array(p.astype(np.float32), dtype=wp.vec3, device=device)
            u = wp.zeros(len(p), dtype=wp.vec3, device=device)
            G = wp.zeros(len(p), dtype=wp.mat33, device=device)
            wp.launch(_eval_flow, dim=len(p), inputs=[xa, t, FlowParams3D(*flow.params()), flow.mac_arrays(device)[0],
                                                      u, G, which], device=device)
            return u.numpy().astype(np.float64), G.numpy().astype(np.float64)

        u, G = ev(pts)
        np.testing.assert_allclose(u, np.stack(flow.velocity_numpy(*pts.T, 1.3), axis=1), atol=1e-5)
        assert np.abs(np.trace(G, axis1=1, axis2=2)).max() < 1e-4  # divergence free
        for b in range(3):  # central differences on the host (float64)
            e = np.zeros(3)
            e[b] = eps
            up = np.stack(flow.velocity_numpy(*(pts + e).T, 1.3), axis=1)
            um = np.stack(flow.velocity_numpy(*(pts - e).T, 1.3), axis=1)
            np.testing.assert_allclose(G[:, :, b], (up - um) / (2 * eps), atol=2e-3)


def test_rotation_exact_positions():
    rot = Rotation3D(omega=0.9, axis=(1.0, -1.0, 0.5), center=(0.4, 0.5, 0.6))
    p0 = np.array([[0.7, 0.2, 0.5], [0.4, 0.5, 0.9]])
    t, n = 1.7, 4000
    p = p0.copy()
    for _ in range(n):  # RK4
        f = lambda q: np.stack(rot.velocity_numpy(*q.T, 0.0), axis=1)  # noqa: E731
        h = t / n
        k1 = f(p)
        k2 = f(p + 0.5 * h * k1)
        k3 = f(p + 0.5 * h * k2)
        k4 = f(p + h * k3)
        p = p + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    np.testing.assert_allclose(rot.rotate_points(p0, t), p, atol=1e-9)


def test_eikonal3d_sphere(device):
    n, dx = 40, 1.0 / 40
    g = Grid3D.unit_cube(n)
    X, Y, Z = g.cell_centers()
    exact = sdf3d.sphere((0.45, 0.5, 0.55), 0.22)(X, Y, Z)
    distorted = (exact * (1.0 + 0.8 * X) * (1.2 - 0.5 * np.sin(3 * Y))).astype(np.float32)
    for init, near_tol in (("gradient", 0.35), ("crossing", 0.5)):
        src = wp.array(distorted, dtype=float, device=device)
        out = wp.zeros_like(src)
        Eikonal3D(n, n, n, dx, device=device, init=init).redistance(src, out)
        phi = out.numpy()
        assert np.all(np.sign(phi) == np.sign(distorted))
        err = np.abs(phi - exact)
        near = np.abs(exact) < 3 * dx
        assert err[near].max() < near_tol * dx, (init, err[near].max() / dx)
        assert err.max() < 1.5 * dx
    # narrow band: far cells keep their input, near cells match the converged solution
    banded = wp.zeros_like(src)
    Eikonal3D(n, n, n, dx, device=device, iterations=6).redistance(wp.array(distorted, dtype=float, device=device),
                                                                   banded)
    full = wp.zeros_like(src)
    Eikonal3D(n, n, n, dx, device=device).redistance(wp.array(distorted, dtype=float, device=device), full)
    b, f = banded.numpy(), full.numpy()
    np.testing.assert_allclose(b[np.abs(exact) < 2 * dx], f[np.abs(exact) < 2 * dx], atol=1e-6)
    far = np.abs(exact) > 14 * dx
    np.testing.assert_allclose(b[far], distorted[far])


def test_celllist3d_matches_numpy(device):
    nx, ny, nz, dx = 6, 5, 4, 0.1
    rng = np.random.default_rng(4)
    n = 4000
    x = rng.uniform(-0.05, [nx * dx + 0.05, ny * dx + 0.05, nz * dx + 0.05], size=(n, 3)).astype(np.float32)
    x[:50] = 0.25  # crowded cell (no cap)
    valid = (rng.uniform(size=n) > 0.25).astype(np.int32)
    cl = CellList3D(nx, ny, nz, dx, n, device=device)
    cl.build(wp.array(x, dtype=wp.vec3, device=device), wp.array(valid, dtype=wp.int32, device=device))
    start, ids = cl.start.numpy(), cl.ids.numpy()
    c = np.floor(x / np.float32(dx)).astype(int)
    inside = (valid == 1) & np.all(c >= 0, axis=1) & (c[:, 0] < nx) & (c[:, 1] < ny) & (c[:, 2] < nz)
    total = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cid = (i * ny + j) * nz + k
                listed = ids[start[cid]:start[cid + 1]]
                expected = np.nonzero(inside & (c[:, 0] == i) & (c[:, 1] == j) & (c[:, 2] == k))[0]
                np.testing.assert_array_equal(listed, expected)
                total += len(listed)
    assert total == inside.sum() and cl.counts().max() >= 50


def test_celllist3d_skips_nonfinite_positions(device):
    """NaN, inf and out-of-int-range coordinates are not binned (int(NaN) is 0 on CUDA)."""
    x = np.full((64, 3), 0.25, np.float32)
    x[0::2, 0] = np.nan
    x[1::4, 1] = np.inf
    x[3::8, 2] = -np.inf
    x[7, 0] = 1.0e30
    keep = np.isfinite(x).all(axis=1) & (np.abs(x) < 1.0).all(axis=1)
    cl = CellList3D(4, 4, 4, 0.25, len(x), device=device)
    cl.build(wp.array(x, dtype=wp.vec3, device=device), wp.ones(len(x), dtype=wp.int32, device=device))
    cid = (1 * 4 + 1) * 4 + 1
    assert cl.counts().sum() == keep.sum() == cl.counts()[1, 1, 1]
    np.testing.assert_array_equal(cl.ids.numpy()[cl.start.numpy()[cid]:cl.start.numpy()[cid + 1]], np.nonzero(keep)[0])


def test_measure_volume_components_and_ply(tmp_path):
    g = Grid3D.unit_cube(96)
    X, Y, Z = g.cell_centers()
    two = sdf3d.sphere((0.3, 0.5, 0.5), 0.15) | sdf3d.sphere((0.75, 0.5, 0.5), 0.08)
    verts, faces = measure3d.marching_cubes(two(X, Y, Z), g.dx)
    v_exact = 4 / 3 * math.pi * (0.15 ** 3 + 0.08 ** 3)
    assert abs(measure3d.mesh_volume(verts, faces) / v_exact - 1) < 6e-3  # marching cubes: O((h / r)^2)
    mv, mf, info = measure3d.largest_component(verts, faces)
    assert info["components"] == 2
    assert abs(info["fragment_volume"] / (4 / 3 * math.pi * 0.08 ** 3) - 1) < 1e-2
    # outward orientation: positive signed volumes, face normals pointing away from the centres
    labels = measure3d.connected_components(verts, faces)
    assert np.all(measure3d.signed_component_volumes(verts, faces, labels) > 0.0)
    tri = mv[mf]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert np.mean(np.einsum("ij,ij->i", normals, tri.mean(axis=1) - (0.3, 0.5, 0.5)) > 0.0) > 0.999
    errs = measure3d.sphere_errors(mv, mf, (0.3, 0.5, 0.5), 0.15, n_samples=20000)
    assert errs["surface"][0] < 0.05 * g.dx and errs["surface"][1] < 0.2 * g.dx
    # a box touching three walls is closed on the walls
    lo, hi = np.array([-0.5, -0.5, -0.5]), np.array([0.4, 0.6, 0.3])
    c, e = 0.5 * (lo + hi), 0.5 * (hi - lo)
    box = np.max([np.abs(a - ca) - ea for a, ca, ea in zip((X, Y, Z), c, e)], axis=0)  # same zero set as the SDF
    assert abs(measure3d.mesh_volume(*measure3d.marching_cubes(box, g.dx)) / (0.4 * 0.6 * 0.3) - 1) < 2e-3
    # binary PLY round trip
    path = str(tmp_path / "m.ply")
    measure3d.write_ply(path, mv, mf)
    with open(path, "rb") as f:
        data = f.read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii")
    assert f"element vertex {len(mv)}" in header and f"element face {len(mf)}" in header
    rv = np.frombuffer(data, dtype="<f4", count=3 * len(mv), offset=end).reshape(-1, 3)
    rec = np.frombuffer(data, dtype=[("n", "u1"), ("i", "<i4", (3,))], count=len(mf), offset=end + rv.nbytes)
    np.testing.assert_allclose(rv, mv, atol=1e-6)
    assert np.all(rec["n"] == 3)
    np.testing.assert_array_equal(rec["i"], mf)


def test_sdf3d_shapes():
    """Sphere and half-space distances and the CSG operators."""
    s = sdf3d.sphere((0.5, 0.5, 0.5), 0.2)
    assert abs(s(0.8, 0.5, 0.5) - 0.1) < 1e-12 and abs(s(0.5, 0.5, 0.5) + 0.2) < 1e-12
    pool = sdf3d.half_space((0.0, 0.0, 0.3), (0.0, 0.0, 2.0))
    assert abs(pool(0.1, 0.9, 0.25) + 0.05) < 1e-12 and abs(pool(0.7, 0.2, 0.5) - 0.2) < 1e-12
    a, b = sdf3d.sphere((0.4, 0.5, 0.5), 0.2), sdf3d.sphere((0.6, 0.5, 0.5), 0.2)
    p = (np.array([0.25, 0.5, 0.75]), np.full(3, 0.5), np.full(3, 0.5))
    np.testing.assert_allclose((a | b)(*p), np.minimum(a(*p), b(*p)))
    np.testing.assert_allclose((a & b)(*p), np.maximum(a(*p), b(*p)))
    np.testing.assert_allclose((a - b)(*p), np.maximum(a(*p), -b(*p)))
    np.testing.assert_allclose((-a)(*p), -a(*p))
    assert (a - b)(0.3, 0.5, 0.5) < 0.0 < (a - b)(0.5, 0.5, 0.5)
    assert (pool | s)(0.5, 0.5, 0.1) < 0.0 and (pool | s)(0.5, 0.5, 0.65) < 0.0 < (pool | s)(0.1, 0.1, 0.65)


@wp.kernel
def _eval_normalize3(v: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    k = wp.tid()
    out[k] = safe_normalize3(v[k])


def test_safe_normalize3_nonfinite_and_overflow(device):
    """safe_normalize3 returns the zero vector for zero or non-finite input and rescales overflowing vectors, as the
    2D safe_normalize does."""
    big = 3.0e38
    v = np.array([[3.0, 0.0, 4.0], [0.0, 0.0, 0.0], [np.inf, 1.0, 0.0], [np.nan, 0.0, 1.0], [big, big, 0.0]],
                 dtype=np.float32)
    out = wp.zeros(len(v), dtype=wp.vec3, device=device)
    wp.launch(_eval_normalize3, dim=len(v), inputs=[wp.array(v, dtype=wp.vec3, device=device), out], device=device)
    o = out.numpy()
    np.testing.assert_allclose(o[0], [0.6, 0.0, 0.8], atol=1e-6)
    assert np.array_equal(o[1], np.zeros(3)) and np.array_equal(o[2], np.zeros(3)) and np.array_equal(o[3], np.zeros(3))
    np.testing.assert_allclose(o[4], [np.sqrt(0.5), np.sqrt(0.5), 0.0], atol=1e-6)
