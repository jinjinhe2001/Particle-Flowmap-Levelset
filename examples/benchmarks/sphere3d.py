"""3D LeVeque sphere: a sphere (r = 0.15 at (0.35, 0.35, 0.35)) deformed by Enright's field and brought back.

Velocity ``u0 = 0.25``, period ``T = 10`` (maximum deformation at t = T/2 = 5, step 250 for dt = 0.02).

Full method (long/short reinit 12/6 with redistancing):  python examples/benchmarks/sphere3d.py
Research protocol (long map from t = 0, short reinit every 5 steps, adaptive seeding):
                                                         python examples/benchmarks/sphere3d.py --protocol research
Rigid rotation (advection only, one revolution about z): python examples/benchmarks/sphere3d.py --flow rotation

Writes into ``--out``: ``volume.csv`` (volume every ``--measure-every`` steps, ratios against t = 0, step 5 and
the analytic volume), ``meshes/mesh_XXXX.ply`` (marching cubes of the ``--fine`` x reconstruction) and
``summary.json`` (volume ratios at T/2 and T, chamfer / Hausdorff distance of the main component to the exact
sphere at T (Enright; the shape at T/2 is not a sphere) or at T/2 and T (rotation), ms/step with graphs, memory).
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np

from _common3d import StepTimer, add_levelset_args, gpu_memory, logger, make_levelset, save_mesh, write_json

from pfmls.core import measure3d
from pfmls.flows.analytic3d import Enright3D, Rotation3D
from pfmls.levelset import sdf3d


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_levelset_args(p)
    p.add_argument("--flow", choices=("enright", "rotation"), default="enright",
                   help="enright (default; default protocol: full) or rigid rotation (default protocol: advection)")
    p.add_argument("--steps", type=int, default=None, help="number of steps (default: one period)")
    p.add_argument("--dt", type=float, default=0.02, help="time step (default 0.02)")
    p.add_argument("--period", type=float, default=10.0, help="Enright period T / rotation period (default 10)")
    p.add_argument("--u0", type=float, default=0.25, help="Enright amplitude (default 0.25)")
    p.add_argument("--fine", type=int, default=4, help="refinement of the measurement grid (default 4)")
    p.add_argument("--fine-extra", type=int, default=0,
                   help="also measure the volume at t = 0 and at the check steps (T/2, T) with this refinement "
                        "(the T/2 sheet is under-resolved by the 4x grid; 0 = off)")
    p.add_argument("--measure-every", type=int, default=5, help="volume measurement interval in steps (default 5)")
    p.add_argument("--mesh-steps", type=str, default="auto",
                   help="comma-separated steps whose mesh is saved ('auto' = 0, T/4, T/2, 3T/4, T; '' = none)")
    p.add_argument("--out", type=str, default=None,
                   help="output directory (default outputs/sphere3d_<flow>_<protocol>_<res>)")
    args = p.parse_args()
    if args.protocol is None:
        args.protocol = "advection" if args.flow == "rotation" else "full"

    log = logger(args)
    n_period = int(round(args.period / args.dt))
    steps = args.steps if args.steps is not None else n_period
    if args.flow == "enright":
        center, radius = (0.35, 0.35, 0.35), 0.15
        flow = Enright3D(period=args.period, u0=args.u0)
        exact_center = lambda t: center  # noqa: E731
        checks = {s for s in (n_period // 2, n_period) if s <= steps}
        shape_checks = {s for s in checks if s % n_period == 0}  # the shape is a sphere again only at t = k T
    else:
        center, radius = (0.5, 0.75, 0.5), 0.15
        flow = Rotation3D(omega=2.0 * math.pi / args.period, axis=(0.0, 0.0, 1.0), center=(0.5, 0.5, 0.5))
        exact_center = lambda t: tuple(flow.rotate_points(np.array([center]), t)[0])  # noqa: E731
        checks = {s for s in (n_period // 2, n_period) if s <= steps}
        shape_checks = checks
    mesh_steps = ({0, n_period // 4, n_period // 2, 3 * n_period // 4, n_period} if args.mesh_steps == "auto"
                  else {int(s) for s in args.mesh_steps.split(",") if s.strip()})
    out = args.out or os.path.join("outputs", f"sphere3d_{args.flow}_{args.protocol}_{args.res}"
                                   + (f"_fine{args.fine}" if args.fine != 4 else ""))
    os.makedirs(out, exist_ok=True)

    ls, cfg = make_levelset(args)
    ls.init_from_sdf(sdf3d.sphere(center, radius))
    dx, fine = ls.grid.dx, args.fine
    v_exact = 4.0 / 3.0 * math.pi * radius ** 3
    verts, faces = ls.mesh(fine)
    rows = [(0, 0.0, measure3d.mesh_volume(verts, faces), ls.num_valid(), ls.capacity)]
    extra = {0: ls.volume(args.fine_extra)} if args.fine_extra else {}
    seeding = ls.options.seed_min > 0
    lattice = {}  # volumes without the newborns of adaptive seeding (the evolved state; seeding only fills holes)
    ls.release_fine()
    if 0 in mesh_steps:
        save_mesh(os.path.join(out, "meshes"), "mesh_0000", verts, faces)
    log(f"[sphere3d] {cfg} capacity={ls.capacity} valid={ls.num_valid()} V0/V_exact={rows[0][2] / v_exact:.6f}")

    timer = StepTimer(ls.device)
    shape = {}
    timer.start(1)
    for i in range(1, steps + 1):
        ls.step(args.dt, flow)
        measure_now = i % args.measure_every == 0 or i in (5, steps) or i in checks or i in mesh_steps
        if not measure_now:
            continue
        timer.stop(i)
        verts, faces = ls.mesh(fine)
        rows.append((i, ls.t, measure3d.mesh_volume(verts, faces), ls.num_valid(), ls.capacity))
        if i in mesh_steps:
            save_mesh(os.path.join(out, "meshes"), f"mesh_{i:04d}", verts, faces)
        if i in checks and args.fine_extra:
            extra[i] = ls.volume(args.fine_extra)
        if i in checks and seeding:
            lattice[i] = ls.volume(fine, newborns="off")
        ls.release_fine()
        if i in shape_checks:
            mv, mf, info = measure3d.largest_component(verts, faces)
            err = measure3d.sphere_errors(mv, mf, exact_center(ls.t), radius, device=ls.device)
            shape[i] = {"t": ls.t, "chamfer_dx": err["surface"][0] / dx, "hausdorff_dx": err["surface"][1] / dx,
                        "chamfer_points_dx": err["points"][0] / dx, "hausdorff_points_dx": err["points"][1] / dx,
                        "components": info["components"],
                        "fragment_volume_fraction": info["fragment_volume"] / max(info["total_volume"], 1e-300)}
        if i % (20 * args.measure_every) == 0 or i in checks:
            log(f"  step {i:4d} t={ls.t:6.3f} V/V0={rows[-1][2] / rows[0][2]:.6f} valid={rows[-1][3]} "
                f"capacity={rows[-1][4]}")
        timer.start(i + 1)

    v0 = rows[0][2]
    v5 = next((v for s, _, v, _, _ in rows if s == 5), float("nan"))
    with open(os.path.join(out, "volume.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "t", "volume", "ratio_t0", "ratio_step5", "ratio_exact", "valid_particles", "capacity"])
        for s, t, v, nv, cap in rows:
            w.writerow([s, f"{t:.6f}", f"{v:.10e}", f"{v / v0:.10f}", f"{v / v5:.10f}", f"{v / v_exact:.10f}", nv, cap])

    ratio = {s: v / v0 for s, _, v, _, _ in rows}
    warm = 2 * max(cfg["reinit_long"], cfg["reinit_short"], args.measure_every)
    summary = {
        "benchmark": f"sphere3d_{args.flow}", **cfg, "dt": args.dt, "steps": steps, "fine": fine,
        "volume_exact": v_exact, "volume_t0": v0, "volume_step5": v5, "volume_t0_over_exact": v0 / v_exact,
        "ratio_t0": {str(s): ratio[s] for s in sorted(checks | {steps}) if s in ratio},
        "ratio_step5": {str(s): ratio[s] * v0 / v5 for s in sorted(checks | {steps}) if s in ratio},
        "min_ratio_t0": min(ratio.values()), "max_ratio_t0": max(ratio.values()),
        "fine_extra": args.fine_extra,
        "ratio_t0_fine_extra": {str(s): v / extra[0] for s, v in sorted(extra.items()) if s > 0},
        "ratio_t0_without_newborns": {str(s): v / v0 for s, v in sorted(lattice.items())},
        "shape_error": {str(k): v for k, v in shape.items()},
        "ms_per_step_all": timer.ms_per_step(), "ms_per_step_steady": timer.ms_per_step(after=warm),
        "capacity_final": ls.capacity, "grow_events": ls.grow_events, "valid_particles_final": ls.num_valid(),
        "memory_bytes": ls.memory_bytes(), "gpu_memory": gpu_memory(ls.device),
    }
    write_json(os.path.join(out, "summary.json"), summary)
    for s in sorted(checks | {steps}):
        if s in ratio:
            print(f"[sphere3d] step {s} t={s * args.dt:.3f}: V/V(t0) = {ratio[s]:.6f}  V/V(step5) = "
                  f"{ratio[s] * v0 / v5:.6f}  V/V_exact = {ratio[s] * v0 / v_exact:.6f}"
                  + (f"  ({args.fine_extra}x grid: V/V(t0) = {extra[s] / extra[0]:.6f})" if s in extra else "")
                  + (f"  (without newborns: V/V(t0) = {lattice[s] / v0:.6f})" if s in lattice else ""))
    for s, e in shape.items():
        print(f"[sphere3d] shape error at step {s} (main component vs exact sphere): chamfer {e['chamfer_dx']:.4f} dx, "
              f"Hausdorff {e['hausdorff_dx']:.4f} dx (point clouds: {e['chamfer_points_dx']:.4f} / "
              f"{e['hausdorff_points_dx']:.4f} dx); {e['components']} components, fragments "
              f"{100 * e['fragment_volume_fraction']:.3f}% of the volume")
    mem = summary["memory_bytes"]
    steady = summary["ms_per_step_steady"]
    steady_txt = "not measured (run shorter than the warm-up)" if steady is None else f"{steady:.3f} ms/step"
    print(f"[sphere3d] {summary['ms_per_step_all']:.3f} ms/step overall, steady {steady_txt} "
          f"(graph={ls.use_graph}); level-set memory "
          f"{mem['total'] / 2**20:.1f} MiB (particles {mem['particles'] / 2**20:.1f} MiB, capacity {ls.capacity}, "
          f"{ls.grow_events} growths)")
    print(f"[sphere3d] outputs in {out}")


if __name__ == "__main__":
    main()
