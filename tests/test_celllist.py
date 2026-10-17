"""Particle binning: every valid in-domain particle is listed once, in its cell, in id order."""

import numpy as np
import warp as wp

from pfmls.core.celllist import CellList2D


def test_binning_matches_numpy(device):
    nx, ny, dx = 12, 9, 0.1
    rng = np.random.default_rng(3)
    n = 6000
    x = rng.uniform(-0.05, [nx * dx + 0.05, ny * dx + 0.05], size=(n, 2)).astype(np.float32)
    x[:2000] = 0.55  # many particles in one cell (no cap)
    valid = (rng.uniform(size=n) > 0.2).astype(np.int32)
    cl = CellList2D(nx, ny, dx, n, device=device)
    cl.build(wp.array(x, dtype=wp.vec2, device=device), wp.array(valid, dtype=wp.int32, device=device))
    start, end, ids = cl.start.numpy(), cl.end.numpy(), cl.ids.numpy()

    ci = np.floor(x[:, 0] / np.float32(dx)).astype(int)
    cj = np.floor(x[:, 1] / np.float32(dx)).astype(int)
    inside = (valid == 1) & (ci >= 0) & (ci < nx) & (cj >= 0) & (cj < ny)
    total = 0
    for i in range(nx):
        for j in range(ny):
            c = i * ny + j
            listed = ids[start[c]:end[c]]
            expected = np.nonzero(inside & (ci == i) & (cj == j))[0]
            np.testing.assert_array_equal(listed, expected)  # stable sort -> increasing ids
            total += len(listed)
    assert total == inside.sum()
    assert cl.counts().max() >= 1500  # about 80% of the 2000 are valid

    # rebuilding after moving particles gives a consistent list again
    x2 = (x + 0.03).astype(np.float32)
    cl.build(wp.array(x2, dtype=wp.vec2, device=device), wp.array(valid, dtype=wp.int32, device=device))
    ci2 = np.floor(x2[:, 0] / np.float32(dx)).astype(int)
    cj2 = np.floor(x2[:, 1] / np.float32(dx)).astype(int)
    inside2 = (valid == 1) & (ci2 >= 0) & (ci2 < nx) & (cj2 >= 0) & (cj2 < ny)
    assert cl.counts().sum() == inside2.sum()
    assert not cl._counts.numpy().any()  # the scatter leaves the histogram at zero for the next build


def test_nonfinite_and_huge_positions_not_binned(device):
    """NaN, inf and out-of-int-range coordinates are not binned on any device."""
    nx, ny, dx = 8, 8, 0.125
    x = np.full((64, 2), 0.3, np.float32)
    x[0::2, 0] = np.nan
    x[1::4, 1] = np.inf
    x[3::8, 0] = -np.inf
    x[7, 0] = 1.0e30
    keep = np.isfinite(x).all(axis=1) & (np.abs(x) < 1.0).all(axis=1)
    cl = CellList2D(nx, ny, dx, len(x), device=device)
    cl.build(wp.array(x, dtype=wp.vec2, device=device), wp.ones(len(x), dtype=wp.int32, device=device))
    counts = cl.counts()
    assert counts.sum() == keep.sum() == counts[2, 2]
    c = 2 * ny + 2
    np.testing.assert_array_equal(cl.ids.numpy()[cl.start.numpy()[c]:cl.end.numpy()[c]], np.nonzero(keep)[0])
