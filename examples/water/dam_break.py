"""Dam break: a water column a x 2a (a = 0.25) collapses in the unit box (Martin & Moyce, n^2 = 2).

The interface is the PFM-LS level set, the velocity is transported by APIC (400^2, dt = 2.5e-4, sigma = 1e-3).  The
front runs along the floor and hits the right wall, a thin jet shoots up the wall to the lid and breaks up into
drops, the column on the wall falls back as a plunging breaker that lands at t ~ 3.2, and its jet runs back across
the tank and splashes up the left wall (t ~ 3.6-5.5).  The clip ends at t = 6 on the first low pool.

The front position (third cell row above the floor) is written to frames.csv for every frame; the summary reports
the time t at which the front reaches the right wall and the non-dimensional tau = t sqrt(2 g / a).  The surface
tension sigma = 1e-3 (Bond number 1000: water in a tank about 9 cm wide) rounds the tips of the thin jets.

    python examples/water/dam_break.py                  # t = 6, MP4 in outputs/water2d/dam_break
    python examples/water/dam_break.py --no-render      # measurements only
"""

import os
import sys

try:
    import pfmls  # noqa: F401
except ModuleNotFoundError:  # source checkout without `pip install -e .`
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402

from pfmls.fluid.runner import build_parser, exit_code, run, update_summary  # noqa: E402
from pfmls.fluid.scenes import dam_break  # noqa: E402

A = 0.25  # column width (height 2a)


def main(argv=None):
    args = build_parser("dam_break", __doc__, res=400, t_end=6.0, sigma=1.0e-3).parse_args(argv)
    res = run(dam_break(res=args.res, t_end=args.t_end, sigma=args.sigma, a=A), args)
    t = np.array([r["t"] for r in res.frames])
    front = np.array([r["front"] for r in res.frames])
    hit = np.nonzero(front >= 1.0 - 1.5 / args.res)[0]
    extra = {"a": A, "wall_hit_t": None, "wall_hit_tau": None}
    if len(hit):
        th = round(float(t[hit[0]]), 6)  # frame times are accumulated step by step: drop the round-off
        extra.update(wall_hit_t=th, wall_hit_tau=float(th * np.sqrt(2.0 / A)))
        if not args.quiet:
            print(f"[dam_break] the front reaches the right wall at t = {extra['wall_hit_t']:.3f} "
                  f"(tau = t sqrt(2g/a) = {extra['wall_hit_tau']:.2f})")
    update_summary(res, extra)
    return exit_code(res)


if __name__ == "__main__":
    sys.exit(main())
