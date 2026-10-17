"""Shared driver for the 2D level-set transport benchmarks (Zalesak, LeVeque, Distorted).

Writes into ``--out`` (default ``outputs/<name>_<protocol>_<res>``, e.g. ``outputs/leveque_full_200``;
``_frozen`` is appended with ``--frozen-time``, and ``outputs/<name>_custom_<res>`` is used for other intervals):
  volume.csv        step, t, area, ratio_t0, ratio_step5   (every --measure-every steps and at the report steps)
  phi_fine_XXXX.npy fine (5x) reconstruction at the final step (and every --save-every steps)
  interface.png     final zero contour vs the exact solution
  summary.json      area ratios, shape errors (dx), timings, device and versions
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import warp as wp

try:
    import pfmls
except ModuleNotFoundError:  # source checkout without `pip install -e .`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import pfmls
from pfmls.core import measure, quiet_warp, runtime_info  # noqa: E402
from pfmls.levelset.pfmls2d import PROTOCOLS, parse_options  # noqa: E402


def make_parser(description: str, steps: int, band: float = 2.0):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=int, default=200, help="grid resolution (default 200)")
    p.add_argument("--ppc", type=int, default=16, help="particles per cell, a perfect square (default 16)")
    p.add_argument("--steps", type=int, default=steps, help=f"number of time steps (default {steps})")
    p.add_argument("--dt", type=float, default=0.02, help="time step (default 0.02)")
    p.add_argument("--band", type=float, default=band, help=f"particle band half-width in dx (default {band:g})")
    p.add_argument("--protocol", choices=sorted(PROTOCOLS), default="full",
                   help="full = long/short reinitialisation every 12/6 steps (default); advection = no "
                        "reinitialisation after t = 0 (the paper's advection-only setting); both seed "
                        "adaptively with n_min = ppc / 2")
    p.add_argument("--reinit-long", type=float, default=None,
                   help="override the protocol's long reinitialisation interval (0 = never)")
    p.add_argument("--reinit-short", type=float, default=None,
                   help="override the protocol's short reinitialisation interval (0 = never)")
    p.add_argument("--reinit-unit", choices=("steps", "time"), default="steps",
                   help="unit of the reinitialisation intervals (default steps)")
    p.add_argument("--seed-min", type=int, default=None,
                   help="override the protocol's adaptive seeding threshold (particles per band cell, 0 = off)")
    p.add_argument("--frozen-time", action="store_true",
                   help="evaluate the velocity at the start of the step in all RK stages (first order in time)")
    p.add_argument("--fine", type=int, default=5, help="refinement factor of the measurement grid (default 5)")
    p.add_argument("--measure-every", type=int, default=5, help="area measurement interval in steps (default 5)")
    p.add_argument("--save-every", type=int, default=0, help="save the fine field every N steps (0 = final only)")
    p.add_argument("--out", type=str, default=None,
                   help="output directory (default outputs/<benchmark>_<protocol>_<res>)")
    p.add_argument("--device", type=str, default=None, help="Warp device, e.g. cuda:0 or cpu")
    p.add_argument("--no-graph", action="store_true", help="disable CUDA graph capture")
    p.add_argument("--quiet", action="store_true", help="only print the final summary")
    p.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                   help="advanced PFMLS2DOptions field, e.g. --opt p2g_weights=bspline (repeatable)")
    return p


def protocol_settings(args) -> tuple[float, float, int, str]:
    """``(reinit_long, reinit_short, seed_min, label)``: the protocol's intervals and seeding threshold with the
    overrides applied, and the label of the default output directory (``full``, ``advection`` or ``custom``)."""
    rl, rs, frac = PROTOCOLS[args.protocol]
    seed = int(frac * args.ppc)
    label = args.protocol
    if (args.reinit_long is not None or args.reinit_short is not None or args.reinit_unit != "steps"
            or args.seed_min is not None):
        rl = rl if args.reinit_long is None else args.reinit_long
        rs = rs if args.reinit_short is None else args.reinit_short
        seed = seed if args.seed_min is None else args.seed_min
        match = [k for k, v in PROTOCOLS.items() if (v[0], v[1], int(v[2] * args.ppc)) == (rl, rs, seed)]
        label = match[0] if match and args.reinit_unit == "steps" else "custom"
    if args.reinit_unit == "steps":
        for flag, v in (("--reinit-long", rl), ("--reinit-short", rs)):
            if float(v) != int(v):
                raise SystemExit(f"{flag} {v}: a step interval must be an integer (use --reinit-unit time)")
        rl, rs = int(rl), int(rs)
    return rl, rs, seed, label


def _plot(path, phi_fine, h, ref_pts, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure as skm

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(ref_pts[:, 0], ref_pts[:, 1], ".", ms=0.5, color="#eb6834", label="exact", rasterized=True)
    first = True
    for c in skm.find_contours(phi_fine.T, 0.0):
        ax.plot((c[:, 1] + 0.5) * h, (c[:, 0] + 0.5) * h, "-", lw=1.3, color="#2a78d6",
                label="PFM-LS" if first else None)
        first = False
    center = 0.5 * (ref_pts.min(axis=0) + ref_pts.max(axis=0))
    half = 0.5 * (ref_pts.max(axis=0) - ref_pts.min(axis=0)).max() + 0.04
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, color="#0b0b0b")
    ax.tick_params(colors="#52514e", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#c9c8c2")
    ax.legend(loc="lower left", fontsize=8, frameon=False, markerscale=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def run(args, name: str, flow, shape, reference, shape_checks=None, report_steps=(), options=None):
    """Run one benchmark.

    ``reference(t)`` returns dense exact interface points at time ``t`` (used at the final step);
    ``shape_checks`` maps a step number to ``(label, points)`` for additional shape-error reports
    during the run (e.g. against the initial shape at the end of a reversal period). The area is also
    measured at every step of ``report_steps``.
    """
    rl, rs, seed, label = protocol_settings(args)
    out = args.out or os.path.join("outputs", f"{name}_{label}_{args.res}" + ("_frozen" if args.frozen_time else ""))
    os.makedirs(out, exist_ok=True)
    log = (lambda *a: None) if args.quiet else print
    if args.quiet:
        quiet_warp()

    opts = dict(options or {})
    if seed:
        opts["seed_min"] = seed
    if args.frozen_time:
        opts["frozen_time"] = True
    try:
        opts.update(parse_options(args.opt))
    except ValueError as exc:
        raise SystemExit(f"--opt: {exc}") from None
    ls = pfmls.PFMLevelSet2D(res=args.res, ppc=args.ppc, band=args.band, reinit_long=rl, reinit_short=rs,
                             reinit_unit=args.reinit_unit, device=args.device, use_graph=not args.no_graph, **opts)
    dx, fine = ls.grid.dx, args.fine
    ls.init_from_sdf(shape)
    rows = [(0, 0.0, ls.area(fine))]
    shape_checks = dict(shape_checks or {})
    shape_err = {}
    log(f"[{name}] res={args.res} ppc={args.ppc} band={args.band} reinit={rl}/{rs} {args.reinit_unit} "
        f"options={opts} device={ls.device} graph={ls.use_graph} valid particles={ls.num_valid()} "
        f"area0={rows[0][2]:.10f}")

    seg_times, seg_steps = [], []
    wp.synchronize_device(ls.device)
    t0 = time.perf_counter()
    nseg = 0
    for i in range(1, args.steps + 1):
        ls.step(args.dt, flow)
        nseg += 1
        measure_now = i % args.measure_every == 0 or i == args.steps or i == 5 or i in report_steps
        save_now = args.save_every > 0 and i % args.save_every == 0
        check_now = i in shape_checks
        if measure_now or save_now or check_now:
            wp.synchronize_device(ls.device)
            seg_times.append(time.perf_counter() - t0)
            seg_steps.append((i - nseg + 1, nseg))
            phi_f = ls.reconstruct(fine)
            if check_now:
                label, ref_pts = shape_checks[i]
                c, h = measure.chamfer_hausdorff(measure.zero_contour(phi_f, dx / fine), ref_pts)
                shape_err[label] = [c / dx, h / dx]
            if measure_now:
                rows.append((i, ls.t, measure.contour_area(phi_f, dx / fine)))
                if i in report_steps or i % (args.measure_every * 20) == 0:
                    log(f"  step {i:5d} t={ls.t:7.3f} area ratio {rows[-1][2] / rows[0][2]:.8f}")
            if save_now and i != args.steps:
                np.save(os.path.join(out, f"phi_fine_{i:04d}.npy"), phi_f)
            nseg = 0
            wp.synchronize_device(ls.device)
            t0 = time.perf_counter()

    phi_f = ls.reconstruct(fine)
    np.save(os.path.join(out, f"phi_fine_{args.steps:04d}.npy"), phi_f)
    np.save(os.path.join(out, "phi_grid_final.npy"), ls.phi_grid())

    area0 = rows[0][2]
    area5 = next((a for (s, _, a) in rows if s == 5), float("nan"))
    with open(os.path.join(out, "volume.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "t", "area", "ratio_t0", "ratio_step5"])
        for s, t, a in rows:
            w.writerow([s, f"{t:.6f}", f"{a:.10f}", f"{a / area0:.10f}", f"{a / area5:.10f}"])

    t_final = ls.t
    pts = measure.zero_contour(phi_f, dx / fine)
    ref = reference(t_final)
    shape_err = {f"exact solution at the final time t = {t_final:.4g}":
                 [v / dx for v in measure.chamfer_hausdorff(pts, ref)], **shape_err}

    # timings: whole run and steady state (segments after the first 2 reinit cycles)
    total_t = sum(seg_times)
    total_n = sum(n for _, n in seg_steps)
    warm = 2 * max(rl, rs, args.measure_every) if args.reinit_unit == "steps" else 2 * args.measure_every
    st = [(tt, n) for tt, (first, n) in zip(seg_times, seg_steps) if first > warm]
    steady = 1e3 * sum(t for t, _ in st) / sum(n for _, n in st) if st else None

    ratio = {s: a / area0 for (s, _, a) in rows}
    summary = {
        "benchmark": name, "res": args.res, "ppc": args.ppc, "band": args.band, "dt": args.dt,
        "steps": args.steps, "reinit_long": rl, "reinit_short": rs, "reinit_unit": args.reinit_unit,
        "graph": ls.use_graph, "options": opts, **runtime_info(ls.device),
        "area_t0": area0, "area_step5": area5, "area_final": rows[-1][2],
        "ratio_t0_final": rows[-1][2] / area0, "ratio_step5_final": rows[-1][2] / area5,
        "ratio_t0_at": {str(s): ratio[s] for s in report_steps if s in ratio},
        "ratio_step5_at": {str(s): ratio[s] * area0 / area5 for s in report_steps if s in ratio},
        "min_ratio_t0": min(ratio.values()), "max_ratio_t0": max(ratio.values()),
        "shape_error_dx": {k: {"chamfer": v[0], "hausdorff": v[1]} for k, v in shape_err.items()},
        "ms_per_step_all": 1e3 * total_t / max(1, total_n), "ms_per_step_steady": steady,
        "valid_particles_final": ls.num_valid(), "newborns_final": int(ls.particles()["newborn"].sum()),
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    final_err = next(iter(shape_err.values()))
    title = (f"{name}: step {args.steps}, t = {t_final:.2f}, area ratio {rows[-1][2] / area0:.6f}\n"
             f"chamfer {final_err[0]:.3f} dx, Hausdorff {final_err[1]:.3f} dx")
    _plot(os.path.join(out, "interface.png"), phi_f, dx / fine, ref, title)

    print(f"[{name}] final step {args.steps} t={t_final:.3f}: area/area(t0) = {summary['ratio_t0_final']:.10f}, "
          f"area/area(step5) = {summary['ratio_step5_final']:.10f}")
    for s in report_steps:
        if s in ratio:
            print(f"[{name}]   step {s}: ratio_t0 = {ratio[s]:.10f}, ratio_step5 = {ratio[s] * area0 / area5:.10f}")
    for k, (c, h) in shape_err.items():
        print(f"[{name}] shape error vs {k}: chamfer {c:.4f} dx, Hausdorff {h:.4f} dx")
    steady_txt = f"{steady:.3f} ms/step" if steady is not None else "not measured (run shorter than the warm-up)"
    print(f"[{name}] {summary['ms_per_step_all']:.3f} ms/step overall, steady state {steady_txt} "
          f"(synchronized, measurement excluded; graph={ls.use_graph})")
    print(f"[{name}] outputs in {out}")
    return summary
