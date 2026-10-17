"""Command-line runner of the water examples.

``build_parser`` defines the common flags, ``run`` drives a :class:`~pfmls.fluid.scenes.Scene`: it steps the
solver to ``t_end``, renders frames into MP4/GIF, records the per-step volume series (``volume.csv``) and, for
every frame, the scene's probes and the rendered liquid area (``frames.csv``), optionally writes checkpoints, and
writes a ``summary.json`` with the volume drift, timings, robustness counters and the device and versions used.
It returns a :class:`RunResult`.

Every run ends with a status:

* ``completed`` -- no NaN repair, no CG failure, liquid volume within ``--drift-tol`` of its initial value (both
  the per-step area of the level set on the simulation grid and the per-frame area of the fine reconstruction
  drawn in the video), no single step changing it by more than ``--jump-tol``, and the liquid kinetic energy never
  above ``--energy-tol`` times the energy available to an inviscid flow (initial kinetic energy + potential energy
  above the flat layer + surface energy; :attr:`FreeSurfaceSolver2D.energy_budget`), which catches finite-valued
  blow-ups;
* ``degraded: ...`` -- finished, but the volume or energy criteria were missed;
* ``failed: ...`` -- a non-finite state had to be repaired (velocity NaN scrub, level-set NaN repair) or the
  pressure solve hit its iteration cap.  The run stops at the first frame where this is detected unless
  ``--keep-going`` is given.

The example scripts exit with code 0 only for ``completed``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field, replace

import numpy as np

from ..core.options import HelpFormatter
from ..core.runtime import check_device, quiet_warp, runtime_info
from .freesurface2d import FluidConfig, FreeSurfaceSolver2D, failure_reason
from .pfm_backend import parse_options as parse_levelset_options
from .scenes import Scene

__all__ = ["RunResult", "build_parser", "default_out", "exit_code", "failure_reason", "ignored_flags", "make_config",
           "run", "update_summary"]

#: Steps excluded from the steady-state cost (kernel loading, CUDA-graph capture of every step type).
WARMUP_STEPS = 50
#: Seconds between progress lines.
PROGRESS_EVERY = 10.0


def build_parser(name: str, description: str | None = None, **defaults) -> argparse.ArgumentParser:
    """Common example flags, grouped by purpose.

    ``defaults`` override the per-scene default values (``res``, ``t_end``, ``sigma``).  Solver flags without a
    default fall back to the scene preset, then to :class:`~pfmls.fluid.FluidConfig`.
    """
    p = argparse.ArgumentParser(prog=name, description=description, formatter_class=HelpFormatter)
    g = p.add_argument_group("scene")
    g.add_argument("--res", type=int, default=defaults.get("res", 200), help="grid cells per unit length")
    g.add_argument("--t-end", type=float, default=defaults.get("t_end"), help="end time")
    g.add_argument("--steps", type=int, default=None, help="run a fixed number of steps instead of --t-end")
    g.add_argument("--sigma", type=float, default=defaults.get("sigma"), help="surface tension coefficient")
    g.add_argument("--out", default=None, help="output directory (default: outputs/water2d/<scene>)")
    g.add_argument("--device", default=None,
                   help="Warp device: cuda:0, cpu, ... (default: Warp's preferred device)")

    g = p.add_argument_group("solver (advanced)")
    g.add_argument("--dt", type=float, default=None, help="nominal time step (scene default if unset)")
    g.add_argument("--cfl", type=float, default=None,
                   help="advective limit dt <= cfl dx / max|u|, 0 = off (FluidConfig: 0.5)")
    g.add_argument("--velocity", choices=["apic", "sl"], default=None,
                   help="velocity transport: apic (affine particle-in-cell, the default) or sl (semi-Lagrangian)")
    g.add_argument("--levelset-opt", action="append", default=[], metavar="KEY=VALUE",
                   help="PFM-LS option, repeatable: a PFMBackendConfig or PFMLS2DOptions field, e.g. hessian_scale=1, "
                        "close_films=false, ppc=25")
    g.add_argument("--reseed-every", type=int, default=None,
                   help="APIC particle reseeding interval (FluidConfig: 8)")
    g.add_argument("--wall-separation", choices=["none", "lid", "all"], default=None,
                   help="walls the liquid may detach from (scene default if unset: lid)")
    g.add_argument("--theta-min", type=float, default=None, help="ghost-fluid interface fraction clamp")
    g.add_argument("--curvature-smoothing", type=int, default=None,
                   help="3x3 filter passes on the curvature kappa in the pressure solve (the level set also smooths "
                        "the field kappa is taken from: --levelset-opt curvature_passes=N)")

    g = p.add_argument_group("media")
    g.add_argument("--render", dest="render", action="store_true", default=True,
                   help="write MP4 frames (default)")
    g.add_argument("--no-render", dest="render", action="store_false", help="skip rendering (faster)")
    g.add_argument("--gif", action="store_true", help="also write a GIF (half size, every 2nd frame)")
    g.add_argument("--png", action="store_true", help="also write every frame as PNG")
    g.add_argument("--frame-dt", type=float, default=None, help="time between frames (scene default if unset)")
    g.add_argument("--fps", type=int, default=30, help="video frame rate")
    g.add_argument("--height-px", type=int, default=720, help="frame height in pixels")
    g.add_argument("--style", choices=["light", "dark"], default="light", help="colour scheme")
    g.add_argument("--tint", type=float, default=0.35, help="velocity-magnitude foam tint strength (0 = off)")
    g.add_argument("--particles", action="store_true", help="overlay the APIC particles")
    g.add_argument("--fine", type=int, default=4, help="upsampling factor of the rendered level set")

    g = p.add_argument_group("status criteria")
    g.add_argument("--drift-tol", type=float, default=0.01,
                   help="max |volume drift| (simulation-grid level set and rendered fine area)")
    g.add_argument("--jump-tol", type=float, default=0.005, help="max one-step volume change")
    g.add_argument("--energy-tol", type=float, default=1.5,
                   help="max liquid kinetic energy / energy available to an inviscid flow")
    g.add_argument("--keep-going", action="store_true", help="do not stop when a failure is detected")
    g.add_argument("--quiet", action="store_true", help="print nothing unless the run does not complete")

    g = p.add_argument_group("checkpoints")
    g.add_argument("--ckpt-every", type=int, default=0,
                   help="full-state checkpoint interval in steps (0 = off)")
    g.add_argument("--save-final", action="store_true", help="write a full-state checkpoint at the end")
    g.add_argument("--resume", default=None, help="checkpoint .npz to resume from (same scene and options)")
    return p


def make_config(scene: Scene, args) -> FluidConfig:
    """The scene's :class:`FluidConfig` with the command-line overrides applied (flags left unset keep the scene's
    values)."""
    cfg = FluidConfig(**scene.overrides)
    try:
        opts = parse_levelset_options(getattr(args, "levelset_opt", None) or [])
        if opts:
            cfg = replace(cfg, pfm_levelset=cfg.pfm_levelset.updated(**opts))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"--levelset-opt: {exc}") from None
    for key in ("velocity", "reseed_every", "cfl", "dt"):
        value = getattr(args, key, None)
        if value is not None:
            cfg = replace(cfg, **{key: value})
    if getattr(args, "wall_separation", None) is not None:
        cfg = replace(cfg, pressure=replace(cfg.pressure, wall_separation=args.wall_separation))
    if getattr(args, "theta_min", None) is not None:
        cfg = replace(cfg, pressure=replace(cfg.pressure, theta_min=args.theta_min))
    if getattr(args, "curvature_smoothing", None) is not None:
        cfg = replace(cfg, pressure=replace(cfg.pressure, curvature_smoothing=args.curvature_smoothing))
    return cfg


def make_renderer(solver, scene: Scene, args):
    """Renderer configured from the CLI flags and the scene's ``speed_scale``."""
    from ..render.water2d import WaterRenderer, WaterStyle

    kw = dict(speed_tint=args.tint, speed_scale=scene.speed_scale)
    style = WaterStyle.dark(**kw) if args.style == "dark" else WaterStyle(**kw)
    return WaterRenderer(solver.lx, solver.ly, args.height_px, style)


def render_frame(renderer, solver, args, label: str | None = None, phi_f=None) -> np.ndarray:
    """Render the solver state (fine level set, optional speed tint and particles) to an RGB image (the shading runs
    on the solver's device; only the image is copied to the host)."""
    if phi_f is None:
        phi_f = solver.fine_phi_device(args.fine)
    speed = solver.cell_speed_device() if renderer.style.speed_tint > 0 else None
    parts = solver.velocity_particle_positions() if args.particles else None
    return renderer.render(phi_f, t=solver.t, speed=speed, particles=parts, label=label)


@dataclass
class RunResult:
    """What :func:`run` returns: the summary (also in ``summary.json``) and the objects behind it."""

    #: the summary dictionary written to ``summary.json``
    summary: dict
    #: the solver in its final state
    solver: FreeSurfaceSolver2D
    #: one row per frame: step, t, the scene's probes and the rendered liquid area (``frames.csv``)
    frames: list = field(default_factory=list)
    #: output directory
    out: str = ""

    @property
    def status(self) -> str:
        return self.summary["status"]

    def __getitem__(self, key: str):
        """Summary entries, e.g. ``result["volume_drift_max"]``."""
        return self.summary[key]


def _log_fn(args):
    return (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))


def _check_device(dev, args, log) -> None:
    """Warn when the run falls back to the CPU (see :func:`pfmls.core.runtime.check_device`)."""
    check_device(dev, args.device, "For a quick CPU run try --res 64 --t-end 0.5 --no-render.", log,
                 f"[runner] note: --device cpu at --res {args.res} is slow; --res 64 runs in about a minute"
                 if args.res >= 128 else None)


def default_out(scene: Scene) -> str:
    """Default output directory of a 2D run: ``outputs/water2d/<scene>``."""
    return os.path.join("outputs", "water2d", scene.name)


def _step_loop(solver, scene: Scene, args, out: str, t_end: float, frame_dt: float, on_frame, log):
    """Step to ``t_end`` (or ``args.steps`` steps), calling ``on_frame()`` (which returns the rendered liquid area)
    every ``frame_dt``.

    Returns ``(status, steps, wall_s, steady_ms_per_step)``; the steady cost excludes the first ``WARMUP_STEPS``
    steps (kernel loading and graph capture)."""
    import warp as wp

    max_steps = getattr(args, "steps", None)
    n0 = solver.step_count
    status = "completed"
    next_frame = solver.t + frame_dt
    wall0 = last_log = time.perf_counter()
    log_step, log_t = n0, solver.t
    warm_t = None
    try:
        while True:
            if max_steps is not None:
                if solver.step_count - n0 >= max_steps:
                    break
            elif solver.t >= t_end - 1e-9:
                break
            dt = solver.cfg.dt
            if max_steps is None:
                dt = min(dt, t_end - solver.t)
            solver.step(dt)
            if solver.step_count - n0 == WARMUP_STEPS:
                wp.synchronize_device(solver.device)
                warm_t = time.perf_counter()
            if solver.t >= next_frame - 1e-9:
                area = on_frame()
                next_frame += frame_dt
                reason = solver.failure_reason()
                if reason is None and not np.isfinite(area):
                    reason = "non-finite level set"
                if reason is not None and status == "completed":
                    status = f"failed: {reason} (t={solver.t:.4f})"
                    log(f"[{scene.name}] FAILURE at t={solver.t:.4f}: {reason}")
                    if not args.keep_going:
                        break
            if args.ckpt_every and solver.step_count % args.ckpt_every == 0:
                solver.save_checkpoint(os.path.join(out, f"ckpt_{solver.step_count:06d}.npz"))
            now = time.perf_counter()
            if now - last_log > PROGRESS_EVERY:
                vs = solver.volume_series()
                drift = vs["volume"][-1] / solver.volume0 - 1.0
                ms = 1000 * (now - last_log) / max(1, solver.step_count - log_step)
                if max_steps is None:
                    rate = (solver.t - log_t) / (now - last_log)
                    eta = (t_end - solver.t) / rate if rate > 0 else float("inf")
                    where = f"t={solver.t:.4f}/{t_end:g} ({100 * solver.t / t_end:.0f}%, ETA {eta:.0f} s)"
                else:
                    where = f"t={solver.t:.4f}"
                log(f"  step {solver.step_count:6d} {where} cg={int(vs['cg_iters'][-1]):4d} "
                    f"vol drift={drift:+.3e} {ms:.2f} ms/step")
                last_log, log_step, log_t = now, solver.step_count, solver.t
    except KeyboardInterrupt:
        status = "interrupted"
    wp.synchronize()
    end = time.perf_counter()
    steps = solver.step_count - n0
    steady = 1000.0 * (end - warm_t) / (steps - WARMUP_STEPS) if warm_t is not None and steps > WARMUP_STEPS else None
    return status, steps, end - wall0, steady


def grade_run(solver, frame_rows: list, status: str, args) -> tuple[str, dict]:
    """Final status (see the module docstring) and the drift / energy statistics of a finished run."""
    vs = solver.volume_series()
    vol, vol_h = vs["volume"], vs["volume_h"]
    rel = vol / solver.volume0 - 1.0
    rel_h = vol_h / vol_h[0] - 1.0
    jumps = np.abs(np.diff(vol)) / solver.volume0 if len(vol) > 1 else np.zeros(1)
    cg = vs["cg_iters"][1:] if len(vs["cg_iters"]) > 1 else np.zeros(1)
    cfl = vs["max_speed"][1:] * vs["dt"][1:] / solver.dx if len(vol) > 1 else np.zeros(1)
    areas = np.array([r["area"] for r in frame_rows])
    rel_a = areas / areas[0] - 1.0 if len(areas) else np.full(1, np.nan)
    reason = solver.failure_reason()
    if not np.all(np.isfinite(vol)):
        reason = "non-finite volume"
    if reason is not None and not status.startswith("failed"):
        status = f"failed: {reason}"
    budget = solver.energy_budget
    ke_ratio = float(np.max(vs["ke"]) / budget) if budget else float("nan")
    if status == "completed":
        problems = []
        if np.max(np.abs(rel)) > args.drift_tol:
            problems.append(f"max volume drift {100 * np.max(np.abs(rel)):.2f}% > {100 * args.drift_tol:g}%")
        if np.max(np.abs(rel_a)) > args.drift_tol:
            problems.append(f"max drift of the rendered liquid area {100 * np.max(np.abs(rel_a)):.2f}% > "
                            f"{100 * args.drift_tol:g}%")
        if np.max(jumps) > args.jump_tol:
            problems.append(f"one-step volume change {100 * np.max(jumps):.2f}% > {100 * args.jump_tol:g}%")
        if budget and ke_ratio > args.energy_tol:
            problems.append(f"kinetic energy {ke_ratio:.2f} x the available energy > {args.energy_tol:g}")
        if problems:
            status = "degraded: " + "; ".join(problems)
    stats = {
        "volume_drift_final": float(rel[-1]), "volume_drift_max": float(np.max(np.abs(rel))),
        "volume_max_step_jump": float(np.max(jumps)),
        "volume_h_drift_final": float(rel_h[-1]), "volume_h_drift_max": float(np.max(np.abs(rel_h))),
        "area_drift_final": float(rel_a[-1]), "area_drift_max": float(np.max(np.abs(rel_a))),
        "cg_iters_mean": float(np.mean(cg)), "cg_iters_max": int(np.max(cg)),
        "max_speed": float(np.max(vs["max_speed"])), "cfl_max": float(np.max(cfl)),
        "kinetic_energy_max": float(np.max(vs["ke"])), "energy_budget": budget, "ke_over_budget": ke_ratio,
        "dt_min": float(np.min(vs["dt"][1:])) if len(vol) > 1 else solver.cfg.dt,
    }
    return status, stats


def write_outputs(out: str, solver, frame_rows: list) -> None:
    """``volume.csv`` (per step) and ``frames.csv`` (per frame)."""
    vs = solver.volume_series()
    with open(os.path.join(out, "volume.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "t", "dt", "volume", "volume_h", "liquid_cells", "kinetic_energy", "cg_iters",
                    "cg_residual", "max_speed"])
        for row in zip(vs["step"], vs["t"], vs["dt"], vs["volume"], vs["volume_h"], vs["cells"], vs["ke"],
                       vs["cg_iters"], vs["cg_residual"], vs["max_speed"]):
            w.writerow([int(row[0]), f"{row[1]:.6f}", f"{row[2]:.3e}", f"{row[3]:.8e}", f"{row[4]:.8e}", int(row[5]),
                        f"{row[6]:.5e}", int(row[7]), f"{row[8]:.3e}", f"{row[9]:.4e}"])
    if frame_rows:
        with open(os.path.join(out, "frames.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
            w.writeheader()
            w.writerows(frame_rows)


def ignored_flags(cfg: FluidConfig, args) -> list[str]:
    """Notes on command-line settings that have no effect with the configuration ``cfg``."""
    notes = []
    if cfg.velocity == "sl":
        if getattr(args, "reseed_every", None) is not None:
            notes.append("--reseed-every has no effect with --velocity sl (no APIC particles)")
        if getattr(args, "particles", False):
            notes.append("--particles has no effect with --velocity sl (no APIC particles)")
    if cfg.pressure.wall_separation != "none" and cfg.pfm_levelset.pad == 0:
        notes.append(f"wall separation '{cfg.pressure.wall_separation}' needs the level-set ghost layer (pad > 0); "
                     "all walls stay closed")
    return notes


def run(scene: Scene, args) -> RunResult:
    """Run ``scene`` with parsed ``args``; returns a :class:`RunResult` (the summary is also written to
    ``summary.json``)."""
    import warp as wp

    quiet_warp()
    wp.init()
    log = _log_fn(args)
    out = args.out or default_out(scene)
    os.makedirs(out, exist_ok=True)
    cfg = make_config(scene, args)
    t_end = scene.t_end if getattr(args, "t_end", None) is None else args.t_end
    frame_dt = args.frame_dt or scene.frame_dt
    dev = wp.get_device(args.device)
    info = runtime_info(dev)

    if scene.description:
        log(f"[{scene.name}] {scene.description}")
    log(f"[{scene.name}] {cfg.nx}x{cfg.ny} dx={cfg.dx:.4g} sigma={cfg.sigma:g} velocity={cfg.velocity} "
        f"t_end={t_end:g} device={info['device']} ({info['device_name']}) -> {out}")
    _check_device(dev, args, log)
    notes = ignored_flags(cfg, args)
    for note in notes:
        log(f"[{scene.name}] note: {note}")
    log(f"[{scene.name}] building the solver (the first run compiles the Warp kernels, which can take a minute; "
        f"later runs load them from the kernel cache)")
    t_setup = time.perf_counter()
    solver = FreeSurfaceSolver2D(cfg, scene.sdf, device=dev)
    if getattr(args, "resume", None):
        solver.load_checkpoint(args.resume)
        log(f"[resume] {args.resume} at step {solver.step_count}, t={solver.t:.4f}")
    setup_s = time.perf_counter() - t_setup
    log(f"[{scene.name}] ready in {setup_s:.1f} s: dt <= {solver.dt_nominal:g}")
    label = f"PFM-LS interface | {cfg.velocity.upper()} velocity"

    writer = renderer = None
    if args.render:
        from ..render.media import MediaWriter

        renderer = make_renderer(solver, scene, args)
        writer = MediaWriter(os.path.join(out, scene.name), fps=args.fps, gif=args.gif,
                             png_dir=os.path.join(out, "frames") if args.png else None)

    frame_rows, last_img = [], [None]

    def on_frame():
        row = {"step": solver.step_count, "t": solver.t}
        if scene.probes:
            phi = solver.phi.numpy()
            row.update({k: fn(phi, solver.dx) for k, fn in scene.probes.items()})
        fine = solver.fine_phi_device(args.fine)
        row["area"] = solver.liquid_area(fine, solver.dx / args.fine)
        if writer is not None:
            img = render_frame(renderer, solver, args, label, fine)
            writer.add(img)
            last_img[0] = img
        frame_rows.append(row)
        return row["area"]

    on_frame()
    status, steps_run, wall, steady = _step_loop(solver, scene, args, out, t_end, frame_dt, on_frame, log)
    if args.save_final:
        solver.save_checkpoint(os.path.join(out, "checkpoint_final.npz"))

    status, stats = grade_run(solver, frame_rows, status, args)
    write_outputs(out, solver, frame_rows)
    media = writer.close() if writer is not None else []
    if last_img[0] is not None:
        from ..render.media import save_png

        save_png(os.path.join(out, scene.name + "_final.png"), last_img[0])
    summary = {
        "scene": scene.name, "status": status, "velocity": cfg.velocity,
        "wall_separation": cfg.pressure.wall_separation if cfg.pfm_levelset.pad > 0 else "none", "notes": notes,
        "res": [cfg.nx, cfg.ny], "dt": solver.dt_nominal, "sigma": cfg.sigma, "t_end": float(solver.t),
        "steps": int(solver.step_count), "setup_s": setup_s, "wall_s": wall,
        "ms_per_step": 1000.0 * wall / max(1, steps_run), "ms_per_step_steady": steady,
        **stats, "media": media, **{k: v for k, v in solver.stats().items() if k not in ("step", "t")},
        "cg_cap_hits": solver.health()["cg_cap_hits"], **info,
    }
    _write_summary(out, summary)
    steady_txt = f", steady {steady:.2f} ms/step" if steady is not None else ""
    msg = (f"[{scene.name}] {status}: {summary['steps']} steps, t={summary['t_end']:.3f}, "
           f"{summary['ms_per_step']:.2f} ms/step incl. frames{steady_txt}, volume drift final "
           f"{100 * summary['volume_drift_final']:+.2f}% max {100 * summary['volume_drift_max']:.2f}% (rendered area: "
           f"final {100 * summary['area_drift_final']:+.2f}% max {100 * summary['area_drift_max']:.2f}%; smoothed "
           f"Heaviside: max {100 * summary['volume_h_drift_max']:.2f}%), "
           f"max step change {100 * summary['volume_max_step_jump']:.3f}%")
    log(msg)
    if status != "completed" and args.quiet:
        print(msg, file=sys.stderr, flush=True)
    return RunResult(summary=summary, solver=solver, frames=frame_rows, out=out)


def update_summary(result: RunResult, extra: dict) -> None:
    """Add ``extra`` (e.g. quantities an example derives from the probes) to the summary of a finished run, in
    ``result.summary`` and in ``summary.json``."""
    result.summary.update(extra)
    _write_summary(result.out, result.summary)


def _write_summary(out: str, summary: dict) -> None:
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def exit_code(result) -> int:
    """Process exit code of an example: 0 for ``completed``, 1 otherwise."""
    return 0 if result["status"] == "completed" else 1

