"""Shared helpers of the 3D benchmark (sphere3d.py)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import warp as wp

try:
    import pfmls  # noqa: F401
except ModuleNotFoundError:  # source checkout without `pip install -e .`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pfmls.core import measure3d, quiet_warp, runtime_info  # noqa: E402
from pfmls.levelset.pfmls3d import PROTOCOLS, PFMLevelSet3D, parse_options  # noqa: E402


def add_levelset_args(p: argparse.ArgumentParser) -> None:
    """Level-set flags of the 3D benchmark (``--protocol`` defaults to None: the script picks it per flow)."""
    p.add_argument("--res", type=int, default=50, help="grid resolution (default 50)")
    p.add_argument("--ppc", type=int, default=64, help="particles per cell, a perfect cube: 27, 64, ... (default 64)")
    p.add_argument("--band", type=float, default=2.0, help="particle band half-width in dx (default 2)")
    p.add_argument("--protocol", choices=sorted(PROTOCOLS), default=None,
                   help="full = long/short reinit 12/6 with redistancing; research = long map from t = 0, short "
                        "reinit every 5 steps, adaptive seeding n_min = 16; "
                        "advection = no reinitialisation after t = 0 (default: full for --flow enright, advection "
                        "for --flow rotation)")
    p.add_argument("--reinit-long", type=float, default=None,
                   help="override the protocol's long interval (steps, or time with --reinit-unit time)")
    p.add_argument("--reinit-short", type=float, default=None, help="override the protocol's short interval")
    p.add_argument("--reinit-unit", choices=("steps", "time"), default="steps",
                   help="unit of the reinitialisation intervals")
    p.add_argument("--seed-min", type=int, default=None, help="override the protocol's adaptive seeding threshold")
    p.add_argument("--frozen-time", action="store_true",
                   help="evaluate the velocity at the start of the step in all RK stages (first order in time)")
    p.add_argument("--device", type=str, default=None, help="Warp device, e.g. cuda:0 or cpu")
    p.add_argument("--no-graph", action="store_true", help="disable CUDA graph capture")
    p.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                   help="advanced PFMLS3DOptions field, e.g. --opt eikonal_init=crossing (repeatable)")
    p.add_argument("--quiet", action="store_true", help="only print the results (no Warp or progress messages)")


def make_levelset(args) -> tuple[PFMLevelSet3D, dict]:
    """Build the level set from the parsed arguments; returns it and the effective configuration."""
    rl, rs, seed = PROTOCOLS[args.protocol]
    rl = rl if args.reinit_long is None else args.reinit_long
    rs = rs if args.reinit_short is None else args.reinit_short
    seed = seed if args.seed_min is None else args.seed_min
    opts = {"seed_min": seed}
    if args.frozen_time:
        opts["frozen_time"] = True
    try:
        opts.update(parse_options(args.opt))
    except ValueError as exc:
        raise SystemExit(f"--opt: {exc}") from None
    if args.reinit_unit == "steps":
        for name, v in (("--reinit-long", rl), ("--reinit-short", rs)):
            if float(v) != int(v):
                raise SystemExit(f"{name} {v}: a step interval must be an integer (use --reinit-unit time)")
        rl, rs = int(rl), int(rs)
    ls = PFMLevelSet3D(res=args.res, ppc=args.ppc, band=args.band, reinit_long=rl, reinit_short=rs,
                       reinit_unit=args.reinit_unit, device=args.device, use_graph=not args.no_graph, **opts)
    cfg = {"protocol": args.protocol, "res": args.res, "ppc": args.ppc, "band": args.band,
           "reinit_long": rl, "reinit_short": rs, "reinit_unit": args.reinit_unit, "options": opts,
           "graph": ls.use_graph, **runtime_info(ls.device)}
    return ls, cfg


def logger(args):
    """``print`` unless ``--quiet`` (which also silences Warp's messages; call before the first Warp launch)."""
    if args.quiet:
        quiet_warp()
        return lambda *a, **k: None
    return print


def gpu_memory(device) -> dict:
    """Memory-pool usage of ``device`` (current / peak bytes), if the Warp build reports it."""
    dev = wp.get_device(device)
    out = {}
    if dev.is_cuda:
        for key, fn in (("pool_current", "get_mempool_used_mem_current"), ("pool_peak", "get_mempool_used_mem_high")):
            f = getattr(wp, fn, None)
            if f is not None:
                try:
                    out[key] = int(f(dev))
                except Exception:  # older Warp or no memory pool
                    pass
    return out


class StepTimer:
    """Accumulates synchronized wall time of stepping segments (measurement excluded)."""

    def __init__(self, device):
        self.device = device
        self.segments: list[tuple[int, int, float]] = []  # (first step, steps, seconds)
        self._t0 = None
        self._first = 1

    def start(self, first_step: int) -> None:
        wp.synchronize_device(self.device)
        self._first = first_step
        self._t0 = time.perf_counter()

    def stop(self, last_step: int) -> None:
        wp.synchronize_device(self.device)
        self.segments.append((self._first, last_step - self._first + 1, time.perf_counter() - self._t0))

    def ms_per_step(self, after: int = 0) -> float | None:
        seg = [(n, t) for first, n, t in self.segments if first > after]
        steps = sum(n for n, _ in seg)
        return 1e3 * sum(t for _, t in seg) / steps if steps else None


def save_mesh(out_dir: str, name: str, verts: np.ndarray, faces: np.ndarray) -> str:
    """Write ``<out_dir>/<name>.ply`` (binary PLY) and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.ply")
    measure3d.write_ply(path, verts, faces)
    return path


def write_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
