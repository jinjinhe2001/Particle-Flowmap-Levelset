"""Free-surface solver tests (CPU-capable; GPU tests run when CUDA is available)."""

import json
import os
from argparse import Namespace
from dataclasses import replace

import numpy as np
import pytest

import warp as wp

from pfmls.core import quiet_warp
from pfmls.fluid import FluidConfig, FreeSurfaceSolver2D, PFMBackendConfig
from pfmls.fluid.pfm_backend import parse_options as parse_levelset_options
from pfmls.fluid.pressure import PressureConfig, PressureSolver, build_system_kernel
from pfmls.fluid.redistance import redistance_closest_point
from pfmls.fluid.runner import build_parser, failure_reason, ignored_flags, make_config, run, update_summary
from pfmls.fluid.scenes import dam_break, droplet_splash, front_position, get_scene, sdf_below
from pfmls.levelset import sdf as _sdf

quiet_warp()


def sdf_circle(cx, cy, r):
    return _sdf.circle((cx, cy), r)


CPU = "cpu"
needs_cuda = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")


def _solver(sdf, n=32, device=CPU, **kw):
    """A small solver on the CPU; the PFM-LS interface with 9 particles per cell keeps the CPU tests fast."""
    kw.setdefault("pfm_levelset", PFMBackendConfig(ppc=9))
    return FreeSurfaceSolver2D(FluidConfig(nx=n, ny=n, **kw), sdf, device=device)


def _grid(n):
    x = (np.arange(n) + 0.5) / n
    return np.meshgrid(x, x, indexing="ij")


def test_hydrostatic_column():
    """A still pool keeps zero velocity and gets the hydrostatic pressure P = dt g (h - y)."""
    n, h, dt, g = 32, 0.5, 1.0e-2, 1.0
    s = _solver(sdf_below(h), n=n, dt=dt, gravity=(0.0, -g))
    for _ in range(3):
        s.step()
    u, v = s.u.numpy(), s.v.numpy()
    vu, vv = s.valid_u.numpy() == 1, s.valid_v.numpy() == 1
    assert np.abs(u[vu]).max() < 1e-5
    assert np.abs(v[vv]).max() < 1e-5
    p = s.pressure.p.numpy()
    y = (np.arange(n) + 0.5) / n
    liquid = y < h
    expected = dt * g * (h - y[liquid])
    assert np.allclose(p[n // 2, liquid], expected, rtol=1e-3, atol=1e-6)


def test_projection_is_divergence_free():
    """After the ghost-fluid projection the liquid cells are discretely divergence-free."""
    n = 48
    dx = 1.0 / n
    rng = np.random.default_rng(0)
    X, Y = _grid(n)
    phi = wp.array(sdf_circle(0.5, 0.5, 0.3)(X, Y).astype(np.float32), dtype=float, device=CPU)
    u = wp.array(rng.standard_normal((n + 1, n)).astype(np.float32), dtype=float, device=CPU)
    v = wp.array(rng.standard_normal((n, n + 1)).astype(np.float32), dtype=float, device=CPU)
    vu = wp.zeros(u.shape, dtype=wp.int32, device=CPU)
    vv = wp.zeros(v.shape, dtype=wp.int32, device=CPU)
    ps = PressureSolver(n, n, dx, device=CPU)
    ps.project(u, v, phi, 1.0e-3, 0.05, 1.0, vu, vv)
    assert ps.max_divergence(u, v, phi) < 10.0 * ps.cfg.tol
    assert 0 < ps.last_iters < ps.cfg.max_iters


@needs_cuda
def test_pcg_graph_matches_eager():
    """The captured whole-loop CG (device-side while node) reproduces the eagerly launched iteration."""
    n = 96
    dx = 1.0 / n
    X, Y = _grid(n)
    phi = sdf_circle(0.5, 0.45, 0.3)(X, Y).astype(np.float32)
    rng = np.random.default_rng(1)
    u0 = rng.standard_normal((n + 1, n)).astype(np.float32)
    v0 = rng.standard_normal((n, n + 1)).astype(np.float32)
    out = []
    for graph in (True, False):
        ps = PressureSolver(n, n, dx, PressureConfig(use_graph=graph, warm_start=False), device="cuda:0")
        d = ps.device
        args = [wp.array(phi, device=d), ps.kappa, wp.array(u0, device=d), wp.array(v0, device=d), ps.open_walls,
                dx, 0.0, 0.05, 0.5 / dx]
        for _ in range(2):  # the graph path captures on its first call and replays on the second
            ps.p.zero_()
            wp.launch(build_system_kernel, dim=(n, n), inputs=args, outputs=[ps.flag, ps.diag, ps.rhs, ps.p], device=d)
            ps._pcg()
        out.append((ps.p.numpy(), ps.last_iters, ps.last_residual))
    (pg, ig, rg), (pe, ie, re) = out
    assert abs(ig - ie) <= ps.cfg.check_every
    assert rg <= ps.cfg.tol and re <= ps.cfg.tol
    assert np.allclose(pg, pe, rtol=1e-3, atol=1e-3 * np.abs(pe).max())


def test_liquid_area_measure():
    """The per-step marching-squares area matches the disc area; the smoothed-Heaviside volume is recorded too."""
    s = _solver(sdf_circle(0.5, 0.5, 0.3), n=48)
    vs = s.volume_series()
    assert abs(vs["volume"][0] / (np.pi * 0.09) - 1.0) < 2e-3
    assert abs(vs["volume_h"][0] / (np.pi * 0.09) - 1.0) < 1e-2
    # a pool touching three walls is closed along the walls
    s = _solver(sdf_below(0.37), n=40)
    assert abs(s.volume_series()["volume"][0] - 0.37) < 1e-4


def test_capillary_time_step_limit():
    s = _solver(sdf_circle(0.5, 0.5, 0.2), n=32, sigma=0.5, dt=1.0e-2)
    dt_cap = np.sqrt(s.dx ** 3 / (2.0 * np.pi * 0.5))
    assert np.isclose(s.choose_dt(), 0.5 * dt_cap)
    s = _solver(sdf_circle(0.5, 0.5, 0.2), n=32, sigma=0.0, dt=1.0e-2)
    assert s.choose_dt() == 1.0e-2


def test_closest_point_redistancing_preserves_interface():
    """The redistancing behind the curvature source turns a distorted level set (same zero set) into the signed
    distance near the interface without moving the interface."""
    n = 64
    dx = 1.0 / n
    X, Y = _grid(n)
    exact = sdf_circle(0.5, 0.5, 0.25)(X, Y)
    distorted = wp.array((2.0 * exact + 0.1 * exact ** 2).astype(np.float32), dtype=float, device=CPU)
    work = [wp.zeros((n, n), dtype=float, device=CPU) for _ in range(2)]
    fixed = wp.zeros((n, n), dtype=wp.int32, device=CPU)
    out = redistance_closest_point(distorted, dx, 2.0 * dx, 8, 10, work[0], work[1], fixed, CPU)
    phi = out.numpy()
    band = np.abs(exact) < 2.5 * dx
    assert np.abs(phi[band] - exact[band]).max() < 0.05 * dx
    assert np.array_equal(phi < 0, exact < 0)


def test_checkpoint_roundtrip(tmp_path):
    """APIC + PFM-LS (ghost layer, film closure): a checkpoint restores the full state, including the APIC
    particles and the level-set particles; a solver of another configuration refuses it."""
    sdf = sdf_below(0.3) | sdf_circle(0.5, 0.62, 0.12)

    def make(**kw):
        return _solver(sdf, n=32, dt=4e-3, sigma=1e-3, cfl=0.0, **kw)

    a = make()
    for _ in range(14):
        a.step()
    path = str(tmp_path / "ck.npz")
    a.save_checkpoint(path)
    for _ in range(5):
        a.step()
    b = make()
    b.load_checkpoint(path)
    assert b.step_count == 14 and b.ls.close_films
    for _ in range(5):
        b.step()
    assert np.array_equal(a.u.numpy(), b.u.numpy())
    assert np.array_equal(a.phi.numpy(), b.phi.numpy())
    assert np.array_equal(a.transport.vel.numpy(), b.transport.vel.numpy())
    vs = b.volume_series()
    assert len(vs["volume"]) == 20 and len(vs["cg_iters"]) == 20
    c = make(velocity="sl")
    with pytest.raises(ValueError, match="velocity"):
        c.load_checkpoint(path)
    assert c.step_count == 0


def test_fluid_levelset_config():
    """``FluidConfig.pfm_levelset`` is a typed config whose defaults are the free-surface settings; the
    reinitialisation intervals in nominal steps become time units of the nominal time step; level-set options
    are set by name, from Python or from command-line strings, and unknown or mistyped names are errors."""
    cfg = FluidConfig(nx=24, ny=24, dt=4e-3, pfm_levelset=PFMBackendConfig().updated(ppc=9))
    s = FreeSurfaceSolver2D(cfg, sdf_below(0.4), device=CPU)
    assert s.ls.pad == 4 and s.ls.close_films and s.ls.curvature_passes == 1
    assert s.ls.ls.ppc == 9 and s.ls.ls.reinit_unit == "time" and s.ls.ls.options.hessian_scale == 0.5
    assert np.isclose(s.ls.ls.reinit_long, 12 * 4e-3) and np.isclose(s.ls.ls.reinit_short, 6 * 4e-3)
    opts = parse_levelset_options(["close_films=false", "hessian_scale=1", "pad=0", "film_deep=0"])
    assert opts == {"close_films": False, "hessian_scale": 1.0, "pad": 0, "film_deep": 0.0}
    pc = PFMBackendConfig().updated(**opts)
    assert not pc.close_films and pc.options.hessian_scale == 1.0 and pc.pad == 0
    for bad in (["close_films=Flase"], ["film_widht=1"], ["ppc=9.5"]):
        with pytest.raises(ValueError):
            parse_levelset_options(bad)
    with pytest.raises(TypeError, match="film_widht"):
        PFMBackendConfig().updated(film_widht=1.0)
    with pytest.raises(TypeError, match="close_films"):
        PFMBackendConfig(close_films="false")
    with pytest.raises(ValueError, match="velocity_sampling"):
        PFMBackendConfig(velocity_sampling="cubic")
    plain = PFMBackendConfig.plain(ppc=9)
    assert plain.pad == 0 and not plain.close_films and plain.options.hessian_scale == 1.0 and plain.ppc == 9


def test_fluid_config_validation():
    """FluidConfig is immutable, uses APIC by default and rejects invalid values at construction."""
    for kw in (dict(nx=0), dict(dt=0.0), dict(velocity="flip"), dict(cfl=-1.0), dict(extension_passes=-1)):
        with pytest.raises(ValueError):
            FluidConfig(**kw)
    with pytest.raises(TypeError):
        FluidConfig(pfm_levelset={"ppc": 9})
    cfg = FluidConfig()
    assert cfg.velocity == "apic"
    with pytest.raises(AttributeError):
        cfg.nx = 10
    assert replace(cfg, nx=10).nx == 10


@pytest.mark.parametrize("velocity", ["apic", "sl"])
def test_transports_free_fall(velocity):
    """A drop in free fall keeps a uniform velocity -g t with both transports."""
    s = _solver(sdf_circle(0.5, 0.6, 0.15), n=32, velocity=velocity, dt=5e-3)
    for _ in range(10):
        s.step()
    uc, vc = s.cell_velocity()
    liq = s.phi.numpy() < 0
    assert np.allclose(vc[liq], -s.t, atol=1e-4)
    assert np.abs(uc[liq]).max() < 1e-4


def _angular_velocity(s):
    uc, vc = s.cell_velocity()
    X, Y = _grid(s.nx)
    liq = s.phi.numpy() < 0
    rx, ry = X - 0.5, Y - 0.5
    return np.sum((rx * vc - ry * uc)[liq]) / np.sum((rx * rx + ry * ry)[liq])


def test_rotating_disk_apic():
    """A free liquid disk in rigid rotation (g = 0, held round by surface tension) keeps most of its angular
    velocity with APIC: the affine matrix C_p carries the velocity gradient through the transfers (a transposed
    C_p or a missing affine term loses the rotation within a fraction of a revolution)."""
    omega = 1.0
    vel = lambda x, y: (-omega * (y - 0.5), omega * (x - 0.5))  # noqa: E731
    cfg = FluidConfig(nx=40, ny=40, sigma=0.02, gravity=(0.0, 0.0), dt=1.0e-2)
    s = FreeSurfaceSolver2D(cfg, sdf_circle(0.5, 0.5, 0.25), vel, device=CPU)
    w0 = _angular_velocity(s)
    s.advance_to(0.5 * np.pi)  # a quarter revolution
    ratio = _angular_velocity(s) / w0
    vol = s.volume_series()["volume"]
    assert abs(vol[-1] / vol[0] - 1.0) < 1e-2
    assert 0.85 < ratio <= 1.0, ratio  # measured about 0.90 at 40^2


@needs_cuda
def test_default_pool_at_rest():
    """Default settings (APIC, PFM-LS): a pool stays at rest with the hydrostatic pressure."""
    n, h = 64, 0.3525
    sc = droplet_splash(res=n)
    cfg = FluidConfig(**{**sc.overrides, "nx": n, "ny": n})
    s = FreeSurfaceSolver2D(cfg, sdf_below(h), device="cuda:0")
    for _ in range(400):
        s.step()
    vu, vv = s.valid_u.numpy() == 1, s.valid_v.numpy() == 1
    assert max(np.abs(s.u.numpy()[vu]).max(), np.abs(s.v.numpy()[vv]).max()) < 1e-4
    liq = s.phi.numpy()[n // 2] < 0
    y = (np.arange(n) + 0.5) / n
    p = s.pressure.p.numpy()[n // 2, liq] / cfg.dt
    assert np.allclose(p, h - y[liq], atol=2e-3)
    vol = s.volume_series()["volume"]
    assert abs(vol[-1] / vol[0] - 1.0) < 1e-3


@needs_cuda
def test_default_static_drop_laplace_pressure():
    """Default settings, zero gravity: a round drop holds the Laplace pressure sigma / R and stays nearly still."""
    n, r, sigma = 64, 0.25, 0.02
    cfg = FluidConfig(nx=n, ny=n, sigma=sigma, gravity=(0.0, 0.0), dt=2.0e-3)
    s = FreeSurfaceSolver2D(cfg, sdf_circle(0.5, 0.5, r), device="cuda:0")
    s.advance_to(1.0)
    phi = s.phi.numpy()
    inner = phi < -3.0 / n
    p_in = s.pressure.p.numpy()[inner] / s.choose_dt()
    assert abs(p_in.mean() / (sigma / r) - 1.0) < 0.03
    uc, vc = s.cell_velocity()
    assert np.hypot(uc, vc)[phi < 0].max() < 0.1 * np.sqrt(sigma / r)
    vol = s.volume_series()["volume"]
    assert abs(vol[-1] / vol[0] - 1.0) < 2e-3


@pytest.mark.parametrize("separation", ["lid", "none"])
def test_liquid_detaches_from_lid(separation):
    """A blob touching the lid falls away under gravity (unilateral walls with the PFM-LS ghost layer); with
    closed walls the projection holds it by suction."""
    n = 32
    s = _solver(sdf_circle(0.5, 1.0, 0.25), n=n, dt=5.0e-3, gravity=(0.0, -1.0),
                pressure=PressureConfig(wall_separation=separation))
    s.advance_to(0.4)
    phi = s.phi.numpy()
    attached = bool((phi[:, -1] < 0).any())
    vol = s.volume_series()["volume"]
    assert abs(vol[-1] / vol[0] - 1.0) < 2e-2
    assert attached == (separation == "none")


def test_runner_status(tmp_path):
    """The runner reports 'completed' for a healthy run, 'degraded' when the volume criteria fail, and
    classifies NaN repairs and CG failures as failures."""
    args = build_parser("t").parse_args(["--no-render", "--steps", "6", "--res", "24", "--device", CPU, "--quiet",
                                         "--levelset-opt", "ppc=9", "--out", str(tmp_path / "a")])
    s = run(droplet_splash(res=24), args)
    assert s.status == "completed" and s.summary["device"] == "cpu" and "warp" in s.summary and s.out
    assert s.summary["velocity"] == "apic"
    assert len(s.frames) >= 1 and s.solver.step_count == 6
    assert s.summary["wall_separation"] == "lid" and s.summary["notes"] == []
    for name in ("volume.csv", "frames.csv", "summary.json"):
        assert os.path.exists(tmp_path / "a" / name)
    assert not os.path.exists(tmp_path / "a" / "checkpoint_final.npz")
    args = Namespace(**{**vars(args), "jump_tol": -1.0, "out": str(tmp_path / "b")})
    assert run(droplet_splash(res=24), args)["status"].startswith("degraded")
    args = Namespace(**{**vars(args), "jump_tol": 0.005, "energy_tol": 1e-6, "out": str(tmp_path / "c")})
    s = run(droplet_splash(res=24), args)
    assert "kinetic energy" in s["status"] and 0.0 < s["ke_over_budget"] < 1.0
    ok = {"velocity_nan_scrubs": 0, "nan_scrubs": 0, "nan_particles": 0, "cg_cap_hits": 0}
    assert failure_reason(ok) is None
    for k in ok:
        assert failure_reason({**ok, k: 3}) is not None


def test_dam_break_scene(tmp_path):
    """The dam-break preset (column a x 2a at the left wall, APIC velocity, dt = 0.1 dx) runs on the CPU, and its
    front probe, written for every frame, starts at x = a; update_summary adds entries to summary.json."""
    n = 24
    sc = get_scene("dam_break", res=n)
    cfg = sc.make_config()
    assert (cfg.nx, cfg.ny, cfg.velocity, cfg.sigma) == (n, n, "apic", 1.0e-3) and cfg.dt == pytest.approx(0.1 / n)
    X, Y = _grid(n)
    phi = sc.sdf(X, Y)
    assert np.allclose(phi[(X < 0.2) & (Y < 0.45)], -np.minimum(0.25 - X, 0.5 - Y)[(X < 0.2) & (Y < 0.45)])
    assert front_position(phi, 1.0 / n) == pytest.approx(0.25)
    assert front_position(np.full((n, n), -1.0), 1.0 / n) == 1.0
    assert np.isnan(front_position(np.ones((n, n)), 1.0 / n))
    args = build_parser("t").parse_args(["--no-render", "--steps", "6", "--res", str(n), "--device", CPU, "--quiet",
                                         "--levelset-opt", "ppc=9", "--out", str(tmp_path)])
    s = run(dam_break(res=n), args)
    assert s.status == "completed" and s.solver.step_count == 6
    assert list(s.frames[0]) == ["step", "t", "front", "area"] and abs(s.frames[0]["front"] - 0.25) < 0.5 / n
    update_summary(s, {"wall_hit_t": None})
    with open(tmp_path / "summary.json") as f:
        assert "wall_hit_t" in json.load(f) and s["wall_hit_t"] is None


def test_scene_defaults_and_cli_overrides():
    """Unset solver flags keep the scene preset (APIC); given flags override it; unknown level-set options stop
    with a message; flags without effect in a configuration are reported."""
    sc = get_scene("droplet_splash", res=24)
    assert sc.make_config().velocity == "apic"
    p = build_parser("t")
    assert make_config(sc, p.parse_args([])).velocity == "apic"
    cfg = make_config(sc, p.parse_args(["--velocity", "sl", "--cfl", "0.25", "--wall-separation", "none"]))
    assert cfg.velocity == "sl" and cfg.cfl == 0.25 and cfg.pressure.wall_separation == "none"
    cfg = make_config(sc, p.parse_args(["--levelset-opt", "close_films=false", "--levelset-opt", "ppc=25"]))
    assert not cfg.pfm_levelset.close_films and cfg.pfm_levelset.ppc == 25
    with pytest.raises(SystemExit):
        make_config(sc, p.parse_args(["--levelset-opt", "close_film=false"]))
    with pytest.raises(SystemExit):
        p.parse_args(["--velocity", "flip"])
    args = p.parse_args(["--velocity", "sl", "--particles", "--reseed-every", "4", "--levelset-opt", "pad=0"])
    notes = ignored_flags(make_config(sc, args), args)
    assert len(notes) == 3 and "--particles" in notes[1] and "pad > 0" in notes[2]
    assert ignored_flags(make_config(sc, p.parse_args([])), p.parse_args([])) == []
    s = sc.make_solver(device=CPU, velocity="sl", pfm_levelset=PFMBackendConfig(ppc=9))
    assert s.cfg.velocity == "sl" and s.cfg.nx == 24 and s.velocity_particle_positions() is None
    with pytest.raises(KeyError):
        get_scene("no_such_scene")


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("velocity", ["apic", "sl"])
def test_gpu_droplet_splash_smoke(velocity):
    sc = droplet_splash(res=96)
    cfg = FluidConfig(**sc.overrides, velocity=velocity)
    s = FreeSurfaceSolver2D(cfg, sc.sdf, device="cuda:0")
    for _ in range(400):
        s.step(2e-3)
    vol = s.volume_series()["volume"]
    assert np.all(np.isfinite(vol))
    assert abs(vol[-1] / vol[0] - 1.0) < 1e-2
    h = s.health()
    assert h["velocity_nan_scrubs"] == 0 and h["cg_cap_hits"] == 0


def test_film_closure_squeeze():
    """A squeezed sub-cell air film between two liquid slabs is closed; without the notch test (film_deep = 0) a
    one-cell notch in a flat liquid surface (liquid on three sides) is closed too."""
    from pfmls.fluid.pfm_backend import _mark_films_kernel

    n = 16
    i = np.arange(n)[:, None] * np.ones((1, n))
    phi = np.where(i < 8, i - 7.85, 8.15 - i).astype(np.float32)   # slabs i <= 7 and i >= 9, air cell 8 at +0.15
    notch = np.minimum(np.arange(n)[None, :] - 7.5, 4.0) * np.ones((n, 1))  # surface between rows 7 and 8
    notch[8, 7] = 0.12                                                     # one-cell dent in the top row
    u = np.zeros((n + 1, n), np.float32)
    u[8, :], u[9, :] = 1.0, -1.0                                           # converging across column 8
    v = np.zeros((n, n + 1), np.float32)
    marks = []
    for field, uu in ((phi, u), (notch.astype(np.float32), u)):
        out = wp.zeros((n, n), dtype=wp.int32, device=CPU)
        wp.launch(_mark_films_kernel, dim=(n, n),
                  inputs=[wp.array(field, dtype=float, device=CPU), wp.array(uu, dtype=float, device=CPU),
                          wp.array(v, dtype=float, device=CPU), 0, 0.5, 0.0],
                  outputs=[out], device=CPU)
        marks.append(out.numpy())
    assert marks[0][8, 2:-2].all()
    assert marks[1][8, 7] == 1


def test_film_closure_notch_depth():
    """``film_deep``: a notch whose neighbour across the squeezing axis lies deeper in the air than ``film_deep``
    cells is left open; a film between two slabs (thin air across it too) and a shallow notch are still closed."""
    from pfmls.fluid.pfm_backend import _mark_films_kernel

    n = 16
    i = np.arange(n)[:, None] * np.ones((1, n))
    j = np.ones((n, 1)) * np.arange(n)[None, :]
    film = np.where(i < 8, i - 7.85, 8.15 - i)      # air column 8 at +0.15 between two slabs
    shallow = np.minimum(j - 7.5, 4.0)              # row 8 at +0.5 above the surface
    deep = np.minimum(j - 7.2, 4.0)                 # row 8 at +0.8 above the surface
    shallow[8, 7] = deep[8, 7] = 0.05               # one-cell dent in the top liquid row
    u = np.zeros((n + 1, n), np.float32)
    u[8, :], u[9, :] = 1.0, -1.0                    # converging across column 8
    v = np.zeros((n, n + 1), np.float32)
    marks = {}
    for name, field in (("film", film), ("shallow", shallow), ("deep", deep)):
        out = wp.zeros((n, n), dtype=wp.int32, device=CPU)
        wp.launch(_mark_films_kernel, dim=(n, n),
                  inputs=[wp.array(field.astype(np.float32), dtype=float, device=CPU),
                          wp.array(u, dtype=float, device=CPU), wp.array(v, dtype=float, device=CPU), 0, 0.5, 0.75],
                  outputs=[out], device=CPU)
        marks[name] = out.numpy()
    assert marks["film"][8, 2:-2].all()
    assert marks["shallow"][8, 7] == 1 and marks["deep"][8, 7] == 0


@needs_cuda
def test_graph_steps_match_eager():
    """Steps replayed from captured CUDA graphs reproduce eagerly launched steps (float atomics only reorder sums),
    including the CFL-limited time-step levels and the reseed / reinitialisation step types."""
    sc = droplet_splash(res=64)
    out = []
    for graph in (True, False):
        cfg = FluidConfig(**{**sc.overrides, "nx": 64, "ny": 64, "dt": 2.0e-3, "cfl": 0.02}, use_graph=graph)
        s = FreeSurfaceSolver2D(cfg, sc.sdf, device="cuda:0")
        for _ in range(120):
            s.step()
        out.append((s.volume_series(), s.u.numpy(), s.v.numpy(), s.stats()))
    (vg, ug, wg, sg), (ve, ue, we, se) = out
    assert sg["cuda_graphs"] > 0 and se["cuda_graphs"] == 0
    assert np.allclose(vg["t"], ve["t"]) and np.allclose(vg["volume"], ve["volume"], rtol=1e-5)
    scale = np.abs(we).max()
    assert scale > 0.1 and np.allclose(wg, we, atol=1e-4 * scale) and np.allclose(ug, ue, atol=1e-4 * scale)
