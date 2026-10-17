"""End-to-end tests of PFMLevelSet3D: API checks, CPU smoke tests and GPU regression tests."""

import dataclasses
import math

import numpy as np
import pytest
import warp as wp

from pfmls.core import measure3d
from pfmls.flows.analytic3d import Enright3D, Rotation3D
from pfmls.flows.mac3d import MACVelocity3D
from pfmls.levelset import particles3d as K
from pfmls.levelset import sdf3d
from pfmls.levelset.pfmls3d import PARTICLE_ARRAYS, PFMLevelSet3D, PFMLS3DOptions, parse_options

SPHERE = sdf3d.sphere((0.35, 0.35, 0.35), 0.15)


def _mac_rotation(n, omega, device):
    h = 1.0 / n
    rot = Rotation3D(omega=omega, axis=(0.0, 0.0, 1.0))
    faces = []
    for axis, shape in enumerate(((n + 1, n, n), (n, n + 1, n), (n, n, n + 1))):
        off = [0.5, 0.5, 0.5]
        off[axis] = 0.0
        axes = [(np.arange(s) + o) * h for s, o in zip(shape, off)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        faces.append(wp.array(rot.velocity_numpy(X, Y, Z, 0.0)[axis].astype(np.float32), dtype=float, device=device))
    return MACVelocity3D(*faces, h), rot


def test_cpu_smoke_full_method():
    """All kernels run on the CPU device (small grid, long + short reinitialisation)."""
    ls = PFMLevelSet3D(res=20, ppc=27, reinit_long=4, reinit_short=2, device="cpu")
    ls.init_from_sdf(sdf3d.sphere((0.5, 0.5, 0.5), 0.25))
    v0 = ls.volume(2)
    assert abs(v0 / (4 / 3 * math.pi * 0.25 ** 3) - 1.0) < 2e-2
    flow = Enright3D(period=1.0, u0=0.25)
    for _ in range(6):
        ls.step(0.02, flow)
    assert np.all(np.isfinite(ls.phi_grid()))
    assert ls.num_valid() > 0
    assert abs(ls.volume(2) / v0 - 1.0) < 2e-2
    mem = ls.memory_bytes()
    assert mem["particles"] > 0 and mem["total"] >= mem["particles"] + mem["grid"]


def test_argument_validation():
    with pytest.raises(ValueError, match="ppc=8"):
        PFMLevelSet3D(res=8, ppc=8, device="cpu")
    with pytest.raises(ValueError, match="ppc >= 27"):
        PFMLevelSet3D(res=8, ppc=1, device="cpu")
    with pytest.warns(RuntimeWarning, match="band"):
        PFMLevelSet3D(res=8, band=4.5, device="cpu", newton_band=6.0)
    with pytest.raises(ValueError, match="normal_gradient"):
        PFMLevelSet3D(res=8, device="cpu", normal_gradient="hessian")
    with pytest.raises(ValueError, match="reconstruct_radius"):
        PFMLevelSet3D(res=8, device="cpu", reconstruct_radius=0.2)
    with pytest.raises(ValueError, match="composite_radius"):
        PFMLevelSet3D(res=8, device="cpu", composite_radius=0.6)
    assert PFMLevelSet3D(res=8, ppc=27, device="cpu").p2g_radius == pytest.approx(1.0 / 3.0)
    assert PFMLevelSet3D(res=8, ppc=125, device="cpu").p2g_radius == pytest.approx(0.2)
    assert PFMLevelSet3D(res=8, ppc=64, device="cpu").p2g_radius == 0.25
    with pytest.raises(ValueError, match="perfect cube"):
        PFMLevelSet3D(res=8, ppc=30, device="cpu")
    with pytest.raises(ValueError, match="newton_band"):
        PFMLevelSet3D(res=8, band=3.5, device="cpu")
    with pytest.raises(ValueError, match="eikonal_init"):
        PFMLevelSet3D(res=8, device="cpu", eikonal_init="fmm")
    ls = PFMLevelSet3D(res=8, device="cpu", options=PFMLS3DOptions(hessian_scale=0.5), seed_min=4)
    assert ls.options.hessian_scale == 0.5 and ls.options.seed_min == 4
    with pytest.raises(dataclasses.FrozenInstanceError):
        ls.options.seed_min = 2
    with pytest.raises(RuntimeError):
        ls.step(0.01, Enright3D())
    with pytest.raises(ValueError, match="no interface"):
        ls.init_from_sdf(np.full((8, 8, 8), 0.3, dtype=np.float32))
    bad = np.where(np.arange(512).reshape(8, 8, 8) == 100, np.nan, SPHERE(*ls.grid.cell_centers()))
    with pytest.raises(ValueError, match="non-finite"):
        ls.init_from_sdf(bad)
    with pytest.raises(ValueError, match="non-finite"):
        ls.init_from_sdf(lambda x, y, z: np.where(x > 0.9, np.nan, SPHERE(x, y, z)))
    with pytest.raises(ValueError, match="elementwise"):
        ls.init_from_sdf(lambda x, y, z: -1.0)
    ls.init_from_sdf(SPHERE)
    for dt in (float("nan"), -0.01, float("inf")):
        with pytest.raises(ValueError, match="dt"):
            ls.step(dt, Enright3D())
    assert ls.t == 0.0 and ls.step_index == 0


def test_option_types_and_parsing():
    """Option values are type-checked (a string "false" for a bool is an error, not True); command-line
    strings are converted by the field's annotation; unknown keys are reported as such."""
    assert parse_options(["frozen_time=false", "composite_newborns=mixed", "cull_distance=None", "newton_max_iter=8",
                          "hessian_scale=1"]) == {"frozen_time": False, "composite_newborns": "mixed",
                                                  "cull_distance": None, "newton_max_iter": 8, "hessian_scale": 1.0}
    for bad in (["frozen_time=maybe"], ["newton_max_iter=1.5"], ["not_an_option=1"], ["hessian_scale"]):
        with pytest.raises(ValueError):
            parse_options(bad)
    with pytest.raises(TypeError, match="frozen_time"):
        PFMLS3DOptions(frozen_time="false")
    with pytest.raises(TypeError, match="newton_max_iter"):
        PFMLS3DOptions(newton_max_iter=2.5)
    with pytest.raises(ValueError, match="composite_newborns"):
        PFMLS3DOptions(composite_newborns="yes")
    with pytest.raises(TypeError, match="unknown option"):
        PFMLevelSet3D(res=8, device="cpu", not_an_option=1)
    with pytest.raises(ValueError, match="cull_distance"):  # a genuine value error is not reported as unknown
        PFMLevelSet3D(res=8, device="cpu", cull_distance=-1.0)
    assert PFMLS3DOptions(hessian_scale=1, frozen_time=np.bool_(True)).hessian_scale == 1


def test_failed_growth_leaves_state_intact(monkeypatch):
    """An allocation failure (e.g. out of memory) during a storage growth leaves the particles untouched."""
    ls = PFMLevelSet3D(res=12, ppc=27, reinit_long=2, reinit_short=1, device="cpu")
    ls.init_from_sdf(sdf3d.sphere((0.5, 0.5, 0.5), 0.25))
    ref = {name: getattr(ls, name).numpy().copy() for name in PARTICLE_ARRAYS}
    cap, real, calls = ls.capacity, wp.zeros, [0]

    def failing(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 6:
            raise RuntimeError("simulated out of memory")
        return real(*args, **kwargs)

    monkeypatch.setattr(wp, "zeros", failing)
    with pytest.raises(RuntimeError, match="simulated"):
        ls._ensure_capacity(cap + 1, 1.5)
    monkeypatch.setattr(wp, "zeros", real)
    assert ls.capacity == cap and ls.grow_events == 0 and ls.cells.capacity == cap
    for name in PARTICLE_ARRAYS:
        np.testing.assert_array_equal(getattr(ls, name).numpy(), ref[name])
    assert ls._parts.x.ptr == ls.x.ptr and ls._parts.valid.ptr == ls.valid.ptr
    ls._ensure_capacity(cap + 1, 1.5)
    assert ls.capacity > cap and ls.grow_events == 1
    np.testing.assert_array_equal(ls.phi.numpy()[:cap], ref["phi"])
    ls.step(0.02, Enright3D(period=1.0))
    assert np.all(np.isfinite(ls.phi_grid()))


@pytest.mark.parametrize("ppc", [27, 64])
def test_droplet_at_rest_is_unchanged(ppc):
    """Full method, zero velocity: a droplet of radius 1.2 dx keeps its shape through repeated long
    reinitialisations (each lattice particle is reseeded from itself alone; a reconstruction radius equal to
    the lattice spacing blended in the neighbours' extrapolations, and droplets drifted at every reinit)."""
    n = 20  # dx = 0.05 is not a power of two: lattice coordinates are rounded
    h = 1.0 / n
    ls = PFMLevelSet3D(res=n, ppc=ppc, reinit_long=6, reinit_short=3, device="cpu")
    ls.init_from_sdf(sdf3d.sphere((0.5 + 0.13 * h, 0.5 - 0.21 * h, 0.5 + 0.07 * h), 1.2 * h))
    rest = Rotation3D(omega=0.0)
    for _ in range(12):
        ls.step(0.02, rest)
    v1, r1 = ls.volume(res=160), ls.reconstruct(res=80)
    for _ in range(18):
        ls.step(0.02, rest)
    assert abs(ls.volume(res=160) / v1 - 1.0) < 2e-5  # radius = lattice spacing: ~1e-3 per reinitialisation
    near = np.abs(r1) < 0.5 * h
    assert np.abs(ls.reconstruct(res=80) - r1)[near].max() < 5e-3 * h


def test_long_reinit_shepard_subset_matches_full():
    """The long reinitialisation computes Shepard gradients only for the particles it reads (cell-centre P2G
    and near-interface particles); in a rigid rotation the reseeded state equals the one obtained with the
    Shepard gradients of every particle."""
    rot = Rotation3D(omega=2 * math.pi / 2.0, axis=(0.3, 1.0, 0.2))
    ls = PFMLevelSet3D(res=16, ppc=27, reinit_long=6, reinit_short=3, device="cpu")
    ls.init_from_sdf(sdf3d.sphere((0.5, 0.62, 0.45), 0.22))
    for _ in range(5):
        ls.step(0.02, rot)
    snap = ls.snapshot()
    ls.step(0.02, rot)
    subset = ls.reconstruct(2), ls.phi.numpy().copy(), ls.valid.numpy().copy()

    def all_particles(phi_below):
        wp.launch(K.shepard_gradient3, dim=ls.capacity, inputs=[ls.x, ls.phi, ls.valid, ls.cells.view, ls.grid.dx,
                                                                1, ls.g], device="cpu")

    ls.restore(snap)
    ls._shepard_subset = all_particles
    ls.step(0.02, rot)
    np.testing.assert_array_equal(ls.valid.numpy(), subset[2])
    np.testing.assert_allclose(ls.phi.numpy(), subset[1], rtol=0.0, atol=1e-6 * ls.grid.dx)
    np.testing.assert_allclose(ls.reconstruct(2), subset[0], rtol=0.0, atol=1e-6 * ls.grid.dx)


@pytest.mark.parametrize("rl, rs", [(0, 2), (4, 2)])
def test_seeding_is_an_output_filter(rl, rs):
    """With the default composite_newborns="off", seeded particles do not enter the dynamics (composite,
    normal-gradient fit, Shepard gradients, the reconstructions of the long reinitialisation): through short and
    long reinitialisations and compactions the background grid and the lattice particles equal those of a run
    without seeding, bitwise. Newborns only fill holes of the output reconstruction ("fallback")."""
    flow = Enright3D(period=1.0, u0=0.5)
    runs = {}
    for seed in (0, 27):
        ls = PFMLevelSet3D(res=16, ppc=27, reinit_long=rl, reinit_short=rs, device="cpu", seed_min=seed,
                           compact_every=5)
        ls.init_from_sdf(sdf3d.sphere((0.5, 0.5, 0.5), 0.25))
        for _ in range(11):
            ls.step(0.02, flow)
        runs[seed] = ls
    seeded, plain = runs[27], runs[0]
    p = seeded.particles()
    assert p["newborn"].any()
    np.testing.assert_array_equal(seeded.phi_grid(), plain.phi_grid())
    lattice, ref = {k: v[~p["newborn"]] for k, v in p.items()}, plain.particles()
    for key in ("x", "phi", "g"):
        np.testing.assert_array_equal(lattice[key], ref[key])
    np.testing.assert_array_equal(seeded.reconstruct(2, newborns="off"), plain.reconstruct(2))
    fallback, mixed = seeded.reconstruct(2), seeded.reconstruct(2, newborns="mixed")
    off = seeded.reconstruct(2, newborns="off")
    assert np.any(fallback != off) and np.any(mixed != fallback)
    with pytest.raises(ValueError, match="newborns"):
        seeded.reconstruct(2, newborns="all")


def test_newton_converges_within_the_cap():
    """The quasi-Newton projection reaches the surface within the default 16 iterations: the redistanced grid of
    a long reinitialisation matches the one with 256 iterations to 0.01 dx. With a fixed clamp (every
    step limited to 0.02 dx, newton_max_step = 0) 16 iterations stop short of it by up to 0.09 dx here."""
    flow = Enright3D(period=1.0, u0=0.5)
    ls = PFMLevelSet3D(res=16, ppc=27, reinit_long=1000, reinit_short=3, device="cpu")
    ls.init_from_sdf(sdf3d.sphere((0.5, 0.45, 0.5), 0.27))
    for _ in range(10):
        ls.step(0.02, flow)
    snap = ls.snapshot()

    def redistanced(max_iter, max_step):
        ls.restore(snap)
        ls._redistancer.max_iter, ls._redistancer.max_step = max_iter, max_step
        ls._long_prepare(True)
        return ls._phi_r.numpy().copy(), ls._mask.numpy() != 0

    ref, band = redistanced(256, 1.0)
    assert band.sum() > 1000
    h = ls.grid.dx
    assert np.abs(redistanced(16, 1.0)[0] - ref)[band].max() < 0.01 * h
    assert np.abs(redistanced(16, 0.0)[0] - ref)[band].max() > 0.03 * h


def test_seeding_candidate_buffers_grow():
    """More flagged cells than the candidate buffers hold: the overflow is counted on the device and the
    buffers grow at the next storage check."""
    ls = PFMLevelSet3D(res=16, ppc=27, reinit_long=0, reinit_short=2, device="cpu", seed_min=28,
                       capacity_check_every=1)
    ls.init_from_sdf(sdf3d.sphere((0.5, 0.5, 0.5), 0.3))
    ls._allocate_seeding(1)
    small = ls._seed_capacity
    used0 = ls.num_used()
    ls.step(0.02, Enright3D(period=1.0))
    assert ls._seed_capacity > small
    assert used0 < ls.num_used() <= used0 + small
    ls.step(0.02, Enright3D(period=1.0))
    assert ls.num_used() > used0 + small


def test_state_validation():
    """restore / load_state_dict reject states of another resolution, ppc or dx before changing anything."""
    ls = PFMLevelSet3D(res=12, ppc=27, device="cpu")
    ls.init_from_sdf(sdf3d.sphere((0.5, 0.5, 0.5), 0.25))
    ref = ls.phi_grid()
    for kwargs in (dict(res=10, ppc=27), dict(res=14, ppc=27), dict(res=12, ppc=64), dict(res=12, ppc=27, dx=0.05)):
        other = PFMLevelSet3D(device="cpu", **kwargs)
        other.init_from_sdf(sdf3d.sphere((0.25, 0.25, 0.25), 0.12) if "dx" in kwargs else
                            sdf3d.sphere((0.5, 0.5, 0.5), 0.25))
        with pytest.raises(ValueError):
            ls.load_state_dict(other.state_dict())
        with pytest.raises(ValueError):
            ls.restore(other.snapshot())
        np.testing.assert_array_equal(ls.phi_grid(), ref)
    state = ls.state_dict()
    state.pop("res"), state.pop("ppc"), state.pop("dx")  # older checkpoints: the array shapes are still checked
    ls.load_state_dict(state)
    del state["Js"]
    with pytest.raises(ValueError, match="Js"):
        ls.load_state_dict(state)


def test_scrub_nonfinite():
    ls = PFMLevelSet3D(res=12, ppc=27, reinit_long=2, reinit_short=1, device="cpu")
    ls.init_from_sdf(sdf3d.sphere((0.5, 0.5, 0.5), 0.25))
    counter = wp.zeros(2, dtype=wp.int32, device="cpu")
    ls.scrub_nonfinite(counter)
    assert counter.numpy().tolist() == [0, 0]
    ls.phi_g.numpy()[3, 4, 5] = np.nan
    phi = ls.phi.numpy()
    first = int(np.nonzero(ls.valid.numpy())[0][0])
    phi[first] = np.inf
    ls.scrub_nonfinite(counter)
    assert counter.numpy().tolist() == [1, 1]
    assert ls.valid.numpy()[first] == 0 and np.all(np.isfinite(ls.phi_grid()))
    for _ in range(4):
        ls.step(0.02, Enright3D(period=1.0))
    assert np.all(np.isfinite(ls.phi_grid()))
    with pytest.raises(ValueError, match="counter"):
        ls.scrub_nonfinite(wp.zeros(1, dtype=wp.int32, device="cpu"))


def test_compaction_is_transparent():
    """Slot compaction reorders the particles into cell order and frees the holes. It changes results only
    at the rounding level (particles that later share a cell are summed in another order); with seeding it
    bounds the slots in use."""
    flow = Enright3D(period=1.0, u0=0.5)
    runs = []
    for every in (0, 3):
        ls = PFMLevelSet3D(res=16, ppc=27, reinit_long=0, reinit_short=2, device="cpu", seed_min=20,
                           compact_every=every, cull_distance=0.0)
        ls.init_from_sdf(sdf3d.sphere((0.5, 0.5, 0.5), 0.25))
        for _ in range(10):
            ls.step(0.02, flow)
        runs.append((ls.reconstruct(2), ls.num_used(), ls.num_valid(), ls.particles()))
    np.testing.assert_allclose(runs[0][0], runs[1][0], rtol=0.0, atol=1e-4 * ls.grid.dx)
    assert runs[0][2] == runs[1][2]
    assert runs[1][1] == runs[1][2] < runs[0][1]  # holes freed (newborns are appended without holes)
    assert runs[1][3]["newborn"].any()
    np.testing.assert_array_equal(np.sort(runs[0][3]["x"], axis=0), np.sort(runs[1][3]["x"], axis=0))
    np.testing.assert_allclose(np.sort(runs[0][3]["phi"]), np.sort(runs[1][3]["phi"]), rtol=0.0, atol=1e-6)


def test_compaction_culls_far_newborns():
    """At compaction the newborns further than cull_distance from the background-grid interface are dropped;
    lattice particles are kept wherever they are (they carry the evolved level set)."""
    ls = PFMLevelSet3D(res=16, ppc=27, reinit_long=0, reinit_short=0, device="cpu", compact_every=4,
                       cull_distance=0.5, seed_min=27)
    ls.init_from_sdf(sdf3d.sphere((0.5, 0.5, 0.5), 0.25))
    n0 = ls.num_valid()
    for _ in range(4):
        ls.step(0.02, Enright3D(period=1.0))
    p = ls.particles()
    assert ls.num_used() == len(p["x"])
    assert np.count_nonzero(~p["newborn"]) == n0
    x = p["x"][p["newborn"]]
    assert len(x) > 0
    phi_s = ls._phi_s.numpy()
    ijk = np.clip(np.floor(x / ls.grid.dx - 0.5).astype(int), 0, 14)
    near = np.min(np.abs(np.stack([phi_s[ijk[:, 0] + a, ijk[:, 1] + b, ijk[:, 2] + c]
                                   for a in (0, 1) for b in (0, 1) for c in (0, 1)])), axis=0)
    assert np.all(near <= 0.5 * ls.grid.dx + 1e-6)  # a trilinear value <= 0.5 dx needs a corner <= 0.5 dx


def test_short_reinit_shepard_subset_matches_full():
    """The short reinitialisation computes Shepard gradients only for the particles the cell-centre
    reconstruction reads; the reconstruction equals the one from the gradients of all particles."""
    ls = PFMLevelSet3D(res=14, ppc=27, reinit_long=0, reinit_short=0, device="cpu", normal_gradient="faces")
    ls.init_from_sdf(sdf3d.sphere((0.5, 0.5, 0.5), 0.25))
    for _ in range(3):
        ls.step(0.02, Enright3D(period=1.0))
    snap = ls.snapshot()
    ls._short_reinit()
    subset = ls._phi_r.numpy().copy(), ls._g_r.numpy().copy(), ls._mask.numpy().copy()
    ls.restore(snap)
    wp.launch(K.shepard_gradient3, dim=ls.capacity, inputs=[ls.x, ls.phi, ls.valid, ls.cells.view, ls.grid.dx, 1,
                                                            ls.g], device="cpu")
    ls._particle_to_grid()
    covered = ls._mask.numpy() == 1
    assert covered.sum() > 100
    np.testing.assert_array_equal(subset[2], ls._mask.numpy())
    np.testing.assert_array_equal(subset[0], ls._phi_r.numpy())
    np.testing.assert_array_equal(subset[1][covered], ls._g_r.numpy()[covered])  # uncovered: eikonal normals


def test_particles_stay_in_the_domain():
    """A MAC velocity that does not vanish on a wall pushes particles against it, not out of the domain."""
    n = 16
    ls = PFMLevelSet3D(res=n, ppc=27, reinit_long=0, reinit_short=0, device="cpu", compact_every=0)
    ls.init_from_sdf(sdf3d.sphere((0.8, 0.5, 0.5), 0.15))
    nv = ls.num_valid()
    faces = [wp.full((n + 1, n, n), 1.0, dtype=float, device="cpu"), wp.zeros((n, n + 1, n), dtype=float, device="cpu"),
             wp.zeros((n, n, n + 1), dtype=float, device="cpu")]
    flow = MACVelocity3D(*faces, 1.0 / n)
    for _ in range(10):
        ls.step(0.05, flow)
    x = ls.particles()["x"]
    assert len(x) == nv and x.min() >= 0.0 and x.max() < 1.0
    assert int(ls.cells.start.numpy()[-1]) == nv  # every particle is binned


def test_seeding_overflow_grows_storage():
    """Newborns that do not fit into the storage are counted, reported and the storage grows."""
    ls = PFMLevelSet3D(res=16, ppc=27, reinit_long=0, reinit_short=2, device="cpu", capacity=1, seed_min=27,
                       capacity_check_every=12, compact_every=0)
    ls.init_from_sdf(sdf3d.sphere((0.5, 0.5, 0.5), 0.25))
    cap0 = ls.capacity
    with pytest.warns(RuntimeWarning, match="dropped"):
        for _ in range(12):
            ls.step(0.02, Enright3D(period=1.0, u0=0.5))
    assert ls.dropped_newborns > 0 and ls.capacity > cap0 and ls.grow_events == 1


def test_storage_growth_is_transparent():
    """Growing the slot storage mid-run (as happens when the band needs more slots) does not change the
    result: a run started at the minimum capacity and grown twice matches a run with spare capacity."""
    flow = Enright3D(period=0.5, u0=0.5)
    runs = []
    for capacity in (None, 1):
        ls = PFMLevelSet3D(res=16, ppc=27, reinit_long=3, reinit_short=1, device="cpu", capacity=capacity,
                           seed_min=20)
        ls.init_from_sdf(sdf3d.sphere((0.5, 0.5, 0.5), 0.25))
        used0 = ls.num_used()
        for k in range(5):
            if capacity == 1 and k in (1, 3):
                ls._ensure_capacity(ls.capacity + 1, 1.2)
            ls.step(0.02, flow)
        assert ls.num_used() > used0  # adaptive seeding appended particles
        runs.append((ls.reconstruct(2), ls.grow_events))
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    assert runs[0][1] == 0 and runs[1][1] == 2


def test_grid_input_and_mac_velocity(device):
    """Initialisation from a grid array and transport by a MAC velocity (rigid rotation, advection only)."""
    n = 24
    ls = PFMLevelSet3D(res=n, ppc=27, reinit_long=0, reinit_short=0, device=device)
    center, radius = (0.5, 0.68, 0.5), 0.15
    ls.init_from_sdf(sdf3d.sphere(center, radius)(*ls.grid.cell_centers()))
    mac, rot = _mac_rotation(n, 1.0, device)
    v0 = ls.volume(3)
    for _ in range(10):
        ls.step(0.05, mac)
    verts, faces = ls.mesh(3)
    c1 = rot.rotate_points(np.array([center]), ls.t)[0]
    err = measure3d.sphere_errors(verts, faces, c1, radius, n_samples=20000, device=device)["surface"]
    dx = ls.grid.dx
    assert err[0] < 0.06 * dx and err[1] < 0.3 * dx, err
    assert abs(measure3d.mesh_volume(verts, faces) / v0 - 1.0) < 2e-3
    with pytest.raises(ValueError, match="do not match"):
        ls.step(0.05, _mac_rotation(12, 1.0, device)[0])


def test_snapshot_restore_and_state_dict_roundtrip():
    flow = Enright3D(period=1.0)
    ls = PFMLevelSet3D(res=14, ppc=27, reinit_long=3, reinit_short=2, device="cpu", seed_min=10)
    ls.init_from_sdf(sdf3d.sphere((0.5, 0.5, 0.5), 0.25))
    for _ in range(2):
        ls.step(0.02, flow)
    snap = ls.snapshot()
    host = ls.state_dict()
    for _ in range(3):
        ls.step(0.02, flow)
    ref = ls.reconstruct(2)
    ls.restore(snap)
    assert ls.step_index == 2
    for _ in range(3):
        ls.step(0.02, flow)
    np.testing.assert_array_equal(ls.reconstruct(2), ref)
    ls2 = PFMLevelSet3D(res=14, ppc=27, reinit_long=3, reinit_short=2, device="cpu", seed_min=10)
    ls2.load_state_dict(host)
    for _ in range(3):
        ls2.step(0.02, flow)
    np.testing.assert_array_equal(ls2.reconstruct(2), ref)


@pytest.mark.gpu
def test_graph_matches_eager(cuda_device):
    """CUDA-graph replay equals eager launches bitwise (full method + adaptive seeding), also when the flow
    parameters change every step."""
    class Ramp(Enright3D):
        def __init__(self):
            super().__init__(period=2.0)
            self.k = 0

        def params(self):
            self.k += 1
            return (self.period, 0.25 + 0.002 * self.k, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    fields, graphs = [], []
    for use_graph in (False, True):
        ls = PFMLevelSet3D(res=32, ppc=64, reinit_long=6, reinit_short=3, device=cuda_device, use_graph=use_graph,
                           seed_min=16)
        ls.init_from_sdf(SPHERE)
        flow = Ramp()
        for _ in range(20):
            ls.step(0.02, flow)
        fields.append(ls.reconstruct(2))
        graphs.append(len(ls._graphs))
    np.testing.assert_array_equal(fields[0], fields[1])
    assert graphs[1] >= 3  # plain, short-only and the two halves of a long step


@pytest.mark.gpu
def test_small_droplet_keeps_its_volume(cuda_device):
    """Full method, a sphere of radius 1.5 dx in rigid rotation for half a revolution (20 long
    reinitialisations): the particle normal gradient keeps the volume (the B-spline "faces" estimate, which
    underestimates the curvature of small radii, inflates it by ~3.4%)."""
    n = 32
    h = 1.0 / n
    rot = Rotation3D(omega=2 * math.pi / 10, axis=(0.0, 0.0, 1.0), center=(0.5, 0.5, 0.5))
    center = (0.5 + 0.13 * h, 0.75 - 0.21 * h, 0.5 + 0.07 * h)
    vols = {}
    for mode in ("particles", "faces"):
        ls = PFMLevelSet3D(res=n, ppc=64, device=cuda_device, normal_gradient=mode)
        ls.init_from_sdf(sdf3d.sphere(center, 1.5 * h))
        v0 = ls.volume(res=256)
        for _ in range(250):
            ls.step(0.02, rot)
        vols[mode] = ls.volume(res=256) / v0
    assert abs(vols["particles"] - 1.0) < 1e-2 and vols["faces"] > 1.02, vols


@pytest.mark.gpu
def test_rigid_rotation_conserves_volume(cuda_device):
    """Advection only, rigid rotation at 40^3: volume conserved to ~1e-5, shape to a few percent of dx."""
    ls = PFMLevelSet3D(res=40, ppc=64, reinit_long=0, reinit_short=0, device=cuda_device)
    center = (0.5, 0.72, 0.5)
    ls.init_from_sdf(sdf3d.sphere(center, 0.15))
    rot = Rotation3D(omega=2 * math.pi / 4.0, axis=(0.0, 0.0, 1.0))
    v0 = ls.volume(4)
    for _ in range(100):  # half a revolution
        ls.step(0.02, rot)
    verts, faces = ls.mesh(4)
    assert abs(measure3d.mesh_volume(verts, faces) / v0 - 1.0) < 5e-5
    c1 = rot.rotate_points(np.array([center]), ls.t)[0]
    err = measure3d.sphere_errors(verts, faces, c1, 0.15, device=cuda_device)["surface"]
    assert err[0] < 0.02 * ls.grid.dx and err[1] < 0.1 * ls.grid.dx, err


@pytest.mark.gpu
def test_leveque_sphere_research_protocol(cuda_device):
    """Research protocol at 32^3 for 100 steps: volume within 1% of t = 0 (the newborns that fill the output
    reconstruction bias it low by O(h^2), see docs/method.md, section 5), no growth failures."""
    ls = PFMLevelSet3D(res=32, ppc=64, reinit_long=0, reinit_short=5, device=cuda_device, seed_min=16)
    ls.init_from_sdf(SPHERE)
    v0 = ls.volume(4)
    flow = Enright3D()
    for _ in range(100):
        ls.step(0.02, flow)
    assert abs(ls.volume(4) / v0 - 1.0) < 1e-2
    assert ls.capacity >= ls.num_used()


@pytest.mark.gpu
def test_near_wall_sheet_survives_compaction(cuda_device):
    """A sphere next to a wall is squeezed into a sheet that the 128^3 grid cannot resolve (research protocol,
    one Enright period). The lattice particles carrying it are never culled, so the shape returns: culling them
    by their distance from the grid interface lost about 1% of the volume here."""
    n = 128
    ls = PFMLevelSet3D(res=n, ppc=64, reinit_long=0, reinit_short=5, device=cuda_device)
    ls.init_from_sdf(sdf3d.sphere((0.16, 0.5, 0.5), 0.15))
    v0 = ls.volume(2)
    dt = 0.02 * 50 / n
    flow = Enright3D()
    for _ in range(int(round(10.0 / dt))):
        ls.step(dt, flow)
    assert abs(ls.volume(2) / v0 - 1.0) < 5e-3


def _torus(center, major, minor):
    """Torus about the z axis through ``center`` with ring radius ``major`` and tube radius ``minor``."""
    cx, cy, cz = center
    return lambda x, y, z: np.sqrt((np.hypot(x - cx, y - cy) - major) ** 2 + (z - cz) ** 2) - minor


@pytest.mark.gpu
def test_seeding_guards_prevent_ghost_liquid(cuda_device):
    """Adaptive seeding on a torus (thin tube) through a full Enright period. Without the guards
    (newborns enter the composite, no newborn guards, no culling of far particles) wrong-sign newborns grow
    ghost liquid (about 310%); with the defaults the volume returns to within 1%."""
    unguarded = dict(seed_coherence=0.0, cull_distance=0.0, composite_newborns="mixed", composite_radius=0.25)
    vols = {}
    for name, opts in (("unguarded", unguarded), ("default", {})):
        ls = PFMLevelSet3D(res=48, ppc=64, reinit_long=0, reinit_short=5, device=cuda_device, seed_min=16, **opts)
        ls.init_from_sdf(_torus((0.35, 0.35, 0.35), 0.12, 0.05))
        v0 = ls.volume(2)
        flow = Enright3D()
        for _ in range(500):
            ls.step(0.02, flow)
        vols[name] = ls.volume(2) / v0
    assert abs(vols["default"] - 1.0) < 1e-2 and vols["unguarded"] > 1.1, vols


@pytest.mark.gpu
def test_graph_cache_is_bounded(cuda_device):
    """Velocity arrays that change address: at most MAX_GRAPHS graphs are kept, and a warning is issued when
    the step keys never repeat."""
    ls = PFMLevelSet3D(res=16, ppc=27, reinit_long=0, reinit_short=0, device=cuda_device)
    ls._graphs.max_graphs = 3
    ls.init_from_sdf(sdf3d.sphere((0.5, 0.6, 0.5), 0.2))
    macs = [_mac_rotation(16, 1.0, cuda_device)[0] for _ in range(5)]
    for k in range(20):
        ls.step(0.01, macs[k % 5])
    assert 0 < len(ls._graphs) <= 3
    fresh = [_mac_rotation(16, 1.0, cuda_device)[0] for _ in range(14)]
    with pytest.warns(RuntimeWarning, match="keep changing address"):
        for mac in fresh:
            ls.step(0.01, mac)
    assert len(ls._graphs) <= 3 and len(ls._graphs.warm) <= 4 * 3
