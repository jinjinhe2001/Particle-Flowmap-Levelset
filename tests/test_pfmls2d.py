"""End-to-end tests of PFMLevelSet2D: API checks, a CPU smoke test and GPU regression tests."""

import dataclasses
import math

import numpy as np
import pytest
import warp as wp

import pfmls
from pfmls.core import measure

ZALESAK_OMEGA = math.pi / (157 * 0.02)


def _mac_rotation(n, omega, device):
    dx = 1.0 / n
    rot = pfmls.flows.Rotation2D(omega=omega)
    xf, yc = np.arange(n + 1) * dx, (np.arange(n) + 0.5) * dx
    Xu, Yu = np.meshgrid(xf, yc, indexing="ij")
    Xv, Yv = np.meshgrid(yc, xf, indexing="ij")
    ux = wp.array(rot.velocity_numpy(Xu, Yu, 0)[0].astype(np.float32), dtype=float, device=device)
    uy = wp.array(rot.velocity_numpy(Xv, Yv, 0)[1].astype(np.float32), dtype=float, device=device)
    return pfmls.flows.MACVelocity2D(ux, uy, dx)


def test_cpu_smoke_full_method():
    """All kernels run on the CPU device (small grid, long + short reinitialisation)."""
    ls = pfmls.PFMLevelSet2D(res=32, ppc=9, band=2.0, reinit_long=4, reinit_short=2, device="cpu")
    ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.6), 0.2))
    a0 = ls.area()
    assert abs(a0 - math.pi * 0.04) < 2e-3
    flow = pfmls.flows.LeVequeVortex2D(period=2.0)
    for _ in range(8):
        ls.step(0.02, flow)
    phi = ls.phi_grid()
    assert np.all(np.isfinite(phi))
    assert ls.num_valid() > 0
    assert abs(ls.area() / a0 - 1.0) < 0.02


def test_argument_validation():
    with pytest.raises(ValueError, match="ppc=4"):
        pfmls.PFMLevelSet2D(res=16, ppc=4, device="cpu")
    with pytest.raises(ValueError, match="ppc >= 9"):
        pfmls.PFMLevelSet2D(res=16, ppc=1, device="cpu")
    with pytest.raises(ValueError, match="perfect square"):
        pfmls.PFMLevelSet2D(res=16, ppc=8, device="cpu")
    with pytest.raises(ValueError, match="p2g_radius"):
        pfmls.PFMLevelSet2D(res=16, device="cpu", p2g_radius=0.6)
    with pytest.raises(ValueError, match="integer"):
        pfmls.PFMLevelSet2D(res=16, reinit_long=1.5, device="cpu")
    with pytest.raises(TypeError, match=r"\['not_an_option'\]"):  # only the unknown key is reported
        pfmls.PFMLevelSet2D(res=16, device="cpu", not_an_option=1, blend_e1=0.95)
    with pytest.raises(TypeError, match="frozen_time"):  # a string "false" would read as True
        pfmls.PFMLS2DOptions(frozen_time="false")
    with pytest.raises(TypeError, match="newton_max_iter"):
        pfmls.PFMLS2DOptions(newton_max_iter=64.5)
    from pfmls.levelset.pfmls2d import PROTOCOLS, parse_options
    assert parse_options(["frozen_time=false", "eikonal_iterations=none", "p2g_weights=bspline"]) == {
        "frozen_time": False, "eikonal_iterations": None, "p2g_weights": "bspline"}
    for bad in (["frozen_time=Flase"], ["newton_max_iter=1.5"], ["not_an_option=1"]):
        with pytest.raises(ValueError):
            parse_options(bad)
    assert PROTOCOLS["full"] == (12, 6, 0.5) and PROTOCOLS["advection"] == (0, 0, 0.5)
    assert pfmls.PFMLevelSet2D(res=16, device="cpu").options.seed_min == 8  # default: ppc / 2
    assert pfmls.PFMLevelSet2D(res=16, ppc=9, device="cpu").options.seed_min == 4
    assert pfmls.PFMLevelSet2D(res=16, device="cpu", seed_min=0).options.seed_min == 0
    for bad in (dict(composite_newborns="all"), dict(reconstruct_newborns="on"), dict(seed_coherence=1.5),
                dict(seed_min=-1)):
        with pytest.raises(ValueError):
            pfmls.PFMLS2DOptions(**bad)
    ls = pfmls.PFMLevelSet2D(res=16, device="cpu", options=pfmls.PFMLS2DOptions(hessian_scale=0.5), blend_e1=0.95)
    assert ls.options.hessian_scale == 0.5 and ls.options.blend_e1 == 0.95
    with pytest.raises(dataclasses.FrozenInstanceError):
        ls.options.blend_e1 = 2.0
    with pytest.raises(AttributeError):
        ls.band = 0.5
    with pytest.raises(ValueError, match="band"):
        pfmls.PFMLevelSet2D(res=16, band=float("nan"), device="cpu")
    with pytest.warns(RuntimeWarning, match="band"):
        pfmls.PFMLevelSet2D(res=16, band=6.0, device="cpu")


def test_mac_grid_mismatch_rejected():
    ls = pfmls.PFMLevelSet2D(res=16, ppc=9, reinit_long=0, reinit_short=0, device="cpu")
    ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.5), 0.2))
    with pytest.raises(ValueError, match="do not match"):
        ls.step(0.01, _mac_rotation(8, 1.0, "cpu"))


def test_bad_dt_and_sdf_rejected_without_side_effects():
    """Non-finite or negative dt and malformed SDFs raise ValueError and leave the state untouched."""
    ls = pfmls.PFMLevelSet2D(res=16, ppc=9, reinit_long=4, reinit_short=2, device="cpu")
    ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.5), 0.2))
    flow = pfmls.flows.Rotation2D(omega=1.0)
    ls.step(0.02, flow)
    ref = ls.state_dict()
    for dt in (float("nan"), float("inf"), -0.01):
        with pytest.raises(ValueError, match="dt"):
            ls.step(dt, flow)
    X, _ = ls.grid.cell_centers()
    bad = [(lambda x, y: np.zeros(5), "shape"), (lambda x, y: 0.3, "shape"),
           (lambda x, y: np.ones((x.shape[-1], 3)), "shape"),
           (lambda x, y: np.full_like(x, np.nan), "non-finite"),
           (lambda x, y: np.where(x > 0.9, np.inf, x - 0.5), "non-finite"),
           (np.full(X.shape, np.nan), "non-finite"), (np.zeros((16, 15)), "shape"),
           (np.full(X.shape, 0.3), "no interface"), (lambda x, y: x + 1.0, "no interface")]
    for sdf, match in bad:
        with pytest.raises(ValueError, match=match):
            ls.init_from_sdf(sdf)
    state = ls.state_dict()
    for key, value in ref.items():
        np.testing.assert_array_equal(state[key], value)
    ls.step(0.02, flow)
    assert ls.step_index == 2 and np.isfinite(ls.phi_grid()).all()


@pytest.mark.gpu
def test_velocity_and_counter_on_another_device_rejected(cuda_device):
    """Arrays on another device raise ValueError before any launch (no crash, no broken context)."""
    for ls_dev, other in (("cpu", cuda_device), (cuda_device, "cpu")):
        ls = pfmls.PFMLevelSet2D(res=16, ppc=9, reinit_long=0, reinit_short=0, device=ls_dev)
        ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.5), 0.2))
        with pytest.raises(ValueError, match="live on"):
            ls.step(0.01, _mac_rotation(16, 1.0, other))
        with pytest.raises(ValueError, match="counter"):
            ls.scrub_nonfinite(wp.zeros(2, dtype=wp.int32, device=other))
        ls.step(0.01, _mac_rotation(16, 1.0, ls_dev))
        ls.scrub_nonfinite(wp.zeros(2, dtype=wp.int32, device=ls_dev))
        assert ls.step_index == 1 and np.isfinite(ls.phi_grid()).all()
    with pytest.raises(ValueError, match="different devices"):
        pfmls.flows.MACVelocity2D(wp.zeros((17, 16), dtype=float, device="cpu"),
                                  wp.zeros((16, 17), dtype=float, device=cuda_device), 1.0 / 16)


def test_area_closes_wall_contours():
    """A pool touching three walls is measured (open contours are closed on the walls)."""
    ls = pfmls.PFMLevelSet2D(res=32, ppc=9, device="cpu")
    ls.init_from_sdf(pfmls.sdf.half_plane((0.0, 0.3), (0.0, 1.0)))
    assert abs(ls.area() - 0.3) < 1e-3


def test_grid_input_and_mac_velocity(device):
    """Initialisation from a grid array and transport by a MAC velocity (rigid rotation)."""
    n = 32
    ls = pfmls.PFMLevelSet2D(res=n, ppc=16, reinit_long=0, reinit_short=0, device=device)
    X, Y = ls.grid.cell_centers()
    ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.65), 0.15)(X, Y))
    dx = ls.grid.dx
    mac = _mac_rotation(n, 1.0, device)
    a0 = ls.area()
    for _ in range(10):
        ls.step(0.05, mac)
    pts = measure.zero_contour(ls.reconstruct(5), dx / 5)
    ref = measure.rotate_points(measure.circle_points((0.5, 0.65), 0.15, 20000), 0.5)
    chamfer, hausdorff = measure.chamfer_hausdorff(pts, ref)
    assert chamfer < 0.1 * dx and hausdorff < 0.5 * dx
    assert abs(ls.area() / a0 - 1.0) < 1e-3


def test_time_schedule_matches_step_schedule():
    """Intervals in time units reproduce the step schedule at constant dt (bitwise)."""
    out = []
    for kw in (dict(reinit_long=4, reinit_short=2), dict(reinit_long=0.08, reinit_short=0.04, reinit_unit="time")):
        ls = pfmls.PFMLevelSet2D(res=24, ppc=9, device="cpu", **kw)
        ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.6), 0.2))
        due = []
        for _ in range(9):
            due.append(ls.reinit_due(0.02))
            ls.step(0.02, pfmls.flows.LeVequeVortex2D(period=2.0))
        out.append((due, ls.reconstruct(3)))
    assert out[0][0] == out[1][0]
    assert out[0][0][3] == (True, True) and out[0][0][1] == (False, True) and out[0][0][2] == (False, False)
    np.testing.assert_array_equal(out[0][1], out[1][1])


def test_snapshot_restore_roundtrip():
    """restore() + the same steps reproduce the run bitwise; state_dict() round-trips."""
    flow = pfmls.flows.LeVequeVortex2D(period=2.0)
    ls = pfmls.PFMLevelSet2D(res=24, ppc=9, reinit_long=4, reinit_short=2, device="cpu")
    ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.6), 0.2))
    for _ in range(3):
        ls.step(0.02, flow)
    snap = ls.snapshot()
    host = ls.state_dict()
    for _ in range(4):
        ls.step(0.02, flow)
    ref = ls.reconstruct(3)
    ls.restore(snap)
    assert ls.step_index == 3
    for _ in range(4):
        ls.step(0.02, flow)
    np.testing.assert_array_equal(ls.reconstruct(3), ref)
    ls2 = pfmls.PFMLevelSet2D(res=24, ppc=9, reinit_long=4, reinit_short=2, device="cpu")
    ls2.load_state_dict(host)
    for _ in range(4):
        ls2.step(0.02, flow)
    np.testing.assert_array_equal(ls2.reconstruct(3), ref)


def test_state_shape_mismatch_rejected():
    """restore() / load_state_dict() refuse a state from another resolution, ppc, dx or seeding setting."""
    def make(res, ppc):
        ls = pfmls.PFMLevelSet2D(res=res, ppc=ppc, reinit_long=4, reinit_short=2, device="cpu")
        ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.6), 0.2))
        return ls

    small, big, dense = make(16, 9), make(24, 9), make(16, 16)
    for target in (big, dense):
        with pytest.raises(ValueError, match="expected"):
            target.restore(small.snapshot())
        with pytest.raises(ValueError, match="expected"):
            target.load_state_dict(small.state_dict())
    with pytest.raises(ValueError, match="expected"):
        small.load_state_dict(big.state_dict())
    other_dx = pfmls.PFMLevelSet2D(res=16, ppc=9, dx=0.05, device="cpu")
    with pytest.raises(ValueError, match="dx"):
        other_dx.load_state_dict(small.state_dict())
    small.load_state_dict(small.state_dict())  # same configuration: accepted
    unseeded = pfmls.PFMLevelSet2D(res=16, ppc=9, reinit_long=4, reinit_short=2, device="cpu", seed_min=0)
    unseeded.init_from_sdf(pfmls.sdf.circle((0.5, 0.6), 0.2))
    for a, b in ((unseeded, small), (small, unseeded)):  # the free-slot list belongs to a seeded state only
        with pytest.raises(ValueError, match="seeding"):
            a.restore(b.snapshot())
        with pytest.raises(ValueError, match="seeding"):
            a.load_state_dict(b.state_dict())


def test_float_overflow_does_not_reach_the_grid():
    """An overflowing short-map Jacobian (C = Js^T C0 Js = inf) leaves the grid finite."""
    ls = pfmls.PFMLevelSet2D(res=24, ppc=9, reinit_long=0, reinit_short=0, device="cpu")
    ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.6), 0.2))
    ls.Js.assign(np.tile(np.diag([1.0e30, 1.0e30]).astype(np.float32), (ls.num_slots, 1, 1)))
    ls.step(0.01, pfmls.flows.Rotation2D(omega=1.0))
    assert not np.isfinite(ls.C.numpy()[ls.valid.numpy() == 1]).all()
    assert np.isfinite(ls.phi_grid()).all()
    assert np.isfinite(ls.reconstruct(2)).all()


@pytest.mark.parametrize("rl, rs", [(0, 0), (4, 2)])
def test_seeding_is_an_output_filter(rl, rs):
    """With the default composite_newborns="off", seeded particles do not enter the dynamics (composite, Shepard
    gradients, the reconstructions of the reinitialisations): through short and long reinitialisations the
    background grid and the lattice particles equal those of a run without seeding, bitwise. Newborns only fill
    holes of the fine reconstruction ("fallback")."""
    flow = pfmls.flows.LeVequeVortex2D(period=2.0)
    runs = {}
    for seed in (0, 5):
        ls = pfmls.PFMLevelSet2D(res=32, ppc=9, reinit_long=rl, reinit_short=rs, device="cpu", seed_min=seed)
        ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.6), 0.2))
        for _ in range(14):
            ls.step(0.02, flow)
        runs[seed] = ls
    seeded, plain = runs[5], runs[0]
    p = seeded.particles()
    assert p["newborn"].any() and seeded.dropped_newborns == 0
    np.testing.assert_array_equal(seeded.phi_grid(), plain.phi_grid())
    lattice, ref = {k: v[~p["newborn"]] for k, v in p.items()}, plain.particles()
    for key in ("x", "phi", "g"):
        np.testing.assert_array_equal(lattice[key], ref[key])
    off = seeded.reconstruct(3, newborns="off")
    np.testing.assert_array_equal(off, plain.reconstruct(3))
    fallback, mixed = seeded.reconstruct(3), seeded.reconstruct(3, newborns="mixed")
    assert np.any(mixed != off)
    if rl == 0:  # (right after a reseed the newborns lie where lattice particles cover every point)
        assert np.any(fallback != off)
    with pytest.raises(ValueError, match="newborns"):
        seeded.reconstruct(3, newborns="all")


def test_seeding_keeps_a_stretched_band_populated():
    """LeVeque's vortex, advection only, 64^2: at the maximum stretch (t = T/2) the particles of the thin tail
    are spread apart and the grid loses its sign there, so without seeding the tail falls apart (the largest
    inside region holds about 78% of the area) and half the band cells hold fewer than ppc / 2 particles. With
    seed_min = ppc / 2 most band cells keep seed_min particles, the newborns carry the sign of the exact solution
    and the tail stays one region; at t = T both runs return to the same circle."""
    from scipy import ndimage

    flow = pfmls.flows.LeVequeVortex2D(period=6.0)
    stats = {}
    for seed in (0, 8):
        ls = pfmls.PFMLevelSet2D(res=64, ppc=16, reinit_long=0, reinit_short=0, device="cpu", seed_min=seed)
        ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.75), 0.15))
        a0 = ls.area()
        for _ in range(150):
            ls.step(0.02, flow)
        sizes = np.bincount(ndimage.label(ls.reconstruct(5) < 0)[0].ravel())[1:]
        band = np.abs(ls.phi_grid()) < ls.band * ls.grid.dx  # (the reconstruction has binned the newborns)
        populated = np.mean(ls.cells.counts()[band] >= 8)
        if seed:  # newborn signs against the exact solution (positions traced back to the initial circle)
            p = ls.particles()
            x0 = measure.advect_points(p["x"][p["newborn"]], flow.velocity_numpy, ls.t, 0.0, dt=2.0e-3)
            exact = np.hypot(x0[:, 0] - 0.5, x0[:, 1] - 0.75) - 0.15
            wrong = np.mean(np.sign(p["phi"][p["newborn"]]) != np.sign(exact))
            assert p["newborn"].sum() > 1000 and wrong < 5e-3, wrong  # (measured: 5 of 9228)
        for _ in range(150):
            ls.step(0.02, flow)
        stats[seed] = (sizes.max() / sizes.sum(), populated, ls.area() / a0,
                       ndimage.label(ls.reconstruct(5) < 0)[1])
    assert stats[0][0] < 0.9 and stats[8][0] > 0.99, stats
    assert stats[0][1] < 0.6 and stats[8][1] > 0.85, stats
    assert abs(stats[8][2] - stats[0][2]) < 1e-4 and abs(stats[8][2] - 1.0) < 5e-3 and stats[8][3] == 1, stats


def test_seeding_snapshot_restore_roundtrip():
    """With seeding the free-slot list is part of the state: restore() / load_state_dict() + the same steps
    reproduce the run bitwise."""
    flow = pfmls.flows.LeVequeVortex2D(period=2.0)
    ls = pfmls.PFMLevelSet2D(res=24, ppc=9, reinit_long=0, reinit_short=0, device="cpu", seed_min=4)
    ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.6), 0.2))
    for _ in range(5):
        ls.step(0.02, flow)
    snap, host = ls.snapshot(), ls.state_dict()
    for _ in range(5):
        ls.step(0.02, flow)
    ref, ref_p = ls.reconstruct(3), ls.particles()
    assert ref_p["newborn"].any()
    ls2 = pfmls.PFMLevelSet2D(res=24, ppc=9, reinit_long=0, reinit_short=0, device="cpu", seed_min=4)
    ls.restore(snap)
    ls2.load_state_dict(host)
    for target in (ls, ls2):
        for _ in range(5):
            target.step(0.02, flow)
        np.testing.assert_array_equal(target.reconstruct(3), ref)
        np.testing.assert_array_equal(target.particles()["x"], ref_p["x"])


def test_newborn_curvature_survives_short_reinit():
    """With seeding and short reinitialisations (composite_newborns="off"), a short reinitialisation refits C0 of the
    lattice particles only; a newborn continues its transported C (C0 = C before the map reset) instead of returning
    to its value at birth. A step with dt = 0 that runs the short reinitialisation leaves C of the newborns as is."""
    flow = pfmls.flows.LeVequeVortex2D(period=2.0)
    ls = pfmls.PFMLevelSet2D(res=32, ppc=9, reinit_long=0, reinit_short=4, device="cpu", seed_min=5)
    ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.6), 0.2))
    for _ in range(7):
        ls.step(0.02, flow)
    assert ls.reinit_due(0.02) == (False, True)
    newborn = ls.valid.numpy() == 2
    C, C0 = ls.C.numpy()[newborn], ls.C0.numpy()[newborn]
    assert newborn.any() and not np.allclose(C, C0)  # transported since birth
    ls.step(0.0, flow)
    np.testing.assert_allclose(ls.C.numpy()[newborn], C, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ls.C0.numpy()[newborn], C, rtol=1e-6, atol=1e-6)


def test_skip_long_reinit():
    ls = pfmls.PFMLevelSet2D(res=24, ppc=9, reinit_long=2, reinit_short=1, device="cpu")
    ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.6), 0.2))
    flow = pfmls.flows.Rotation2D(omega=1.0)
    ls.step(0.02, flow)
    assert ls.reinit_due(0.02) == (True, True)
    ls.step(0.02, flow, skip_long_reinit=True)
    assert ls.reinit_due(0.02) == (False, True)
    assert ls.reinit_due(0.02) == (False, True)


@pytest.mark.gpu
def test_zalesak_regression(cuda_device):
    """Zalesak, advection only, 100^2, 16 ppc: area conserved to ~1e-5 (the full method gives ~1e-3)."""
    ls = pfmls.PFMLevelSet2D(res=100, ppc=16, band=3.0, reinit_long=0, reinit_short=0, device=cuda_device,
                             p2g_weights="bspline")
    ls.init_from_sdf(pfmls.sdf.slotted_disk())
    a0 = ls.area()
    flow = pfmls.flows.Rotation2D(omega=ZALESAK_OMEGA)
    errs = []
    for i in range(1, 41):
        ls.step(0.02, flow)
        if i % 10 == 0:
            errs.append(abs(ls.area() / a0 - 1.0))
    assert max(errs) < 1e-4, errs
    pts = measure.zero_contour(ls.reconstruct(5), ls.grid.dx / 5)
    ref = measure.rotate_points(measure.slotted_disk_points(), ZALESAK_OMEGA * ls.t)
    chamfer, hausdorff = measure.chamfer_hausdorff(pts, ref)
    assert chamfer < 0.05 * ls.grid.dx and hausdorff < 0.3 * ls.grid.dx


@pytest.mark.gpu
def test_graph_matches_eager(cuda_device):
    """CUDA-graph replay gives bitwise the same result as eager launches (full method), also when the
    flow parameters change every step (they are uploaded, not baked into the graph)."""
    class Ramp(pfmls.flows.LeVequeVortex2D):
        def __init__(self):
            super().__init__()
            self.k = 0

        def params(self):
            self.k += 1
            return (self.period, 1.0 + 0.01 * self.k, 0.0, 0.0)

    phis, graphs = [], []
    for use_graph in (False, True):
        ls = pfmls.PFMLevelSet2D(res=64, ppc=16, reinit_long=6, reinit_short=3, device=cuda_device,
                                 use_graph=use_graph)
        ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.75), 0.15))
        flow = Ramp()
        for _ in range(20):
            ls.step(0.02, flow)
        phis.append(ls.reconstruct(5))
        graphs.append(len(ls._graphs))
    np.testing.assert_array_equal(phis[0], phis[1])
    assert graphs[1] == 3  # plain, short-only and long steps


@pytest.mark.gpu
def test_seeding_graph_matches_eager(cuda_device):
    """Adaptive seeding is captured into the step graph (no host synchronisation) and replays bitwise like
    eager launches, in advection-only mode and with long reinitialisations (which free the newborns)."""
    flow = pfmls.flows.LeVequeVortex2D(period=2.0)
    for rl, rs in ((0, 0), (6, 3)):
        out = []
        for use_graph in (False, True):
            ls = pfmls.PFMLevelSet2D(res=64, ppc=16, reinit_long=rl, reinit_short=rs, device=cuda_device,
                                     use_graph=use_graph, seed_min=8)
            ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.75), 0.15))
            for _ in range(20):
                ls.step(0.02, flow)
            p = ls.particles()
            out.append((ls.reconstruct(5), p["x"], p["newborn"], len(ls._graphs)))
        assert out[0][2].any()
        for k in range(3):
            np.testing.assert_array_equal(out[0][k], out[1][k])
        assert out[1][3] == (1 if rl == 0 else 3)


@pytest.mark.gpu
def test_half_hessian_reduces_reinit_drift(cuda_device):
    """At zero velocity every long reinitialisation shrinks a small circle slightly with the full
    extrapolation (hessian_scale = 1); the Taylor-consistent hessian_scale = 0.5 removes most of it."""
    drift = {}
    for s in (1.0, 0.5):
        ls = pfmls.PFMLevelSet2D(res=200, ppc=16, reinit_long=12, reinit_short=6, device=cuda_device,
                                 hessian_scale=s)
        ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.5), 0.05))
        a0 = ls.area()
        for _ in range(600):
            ls.step(0.02, pfmls.flows.Rotation2D(omega=0.0))
        drift[s] = abs(ls.area() / a0 - 1.0)
    assert drift[0.5] < 0.25 * drift[1.0], drift


@pytest.mark.gpu
def test_graph_cache_is_bounded(cuda_device):
    """Fresh MAC arrays every step never replay a graph: the caches stay bounded and a warning says why."""
    n = 16
    ls = pfmls.PFMLevelSet2D(res=n, ppc=9, reinit_long=0, reinit_short=0, device=cuda_device)
    ls.init_from_sdf(pfmls.sdf.circle((0.5, 0.5), 0.2))
    keep = []
    with pytest.warns(RuntimeWarning, match="in place"):
        for _ in range(4 * ls.MAX_GRAPHS + 2):
            keep.append(_mac_rotation(n, 1.0, cuda_device))
            ls.step(0.01, keep[-1])
    assert len(ls._graphs.warm) <= 4 * ls.MAX_GRAPHS and len(ls._graphs) <= ls.MAX_GRAPHS
    mac = keep[-1]
    for _ in range(3):  # arrays updated in place: captured and replayed
        ls.step(0.01, mac)
    assert len(ls._graphs) == 1
