"""LeVeque's single vortex: a circle stretched into a spiral and brought back (period T = 12.56).

Advection only:  python examples/benchmarks/leveque.py --protocol advection
Full method:     python examples/benchmarks/leveque.py
Frozen time:     add --frozen-time (velocity evaluated at the start of the step in all RK stages, first
                 order in time; changes the mid-run areas, e.g. 0.9953 -> 0.9964 at step 310 with
                 --protocol advection)
Other periods:   python examples/benchmarks/leveque.py --period 16 --steps 800   (one period: T / dt steps)
Animation:       python examples/benchmarks/leveque.py --protocol advection --save-every 4
                 (the fine field every 4 steps up to t = T, the frames of the README animation)

The default run covers one period (628 steps of dt = 0.02, ending at t = T). The shape error is
reported against the exact solution at the final time (accurate forward integration of the initial
circle) and, at the step that reaches t = T exactly, against the initial circle.
"""

from _common import make_parser, run

import pfmls
from pfmls.core import measure

PERIOD = 12.56


def main():
    p = make_parser(__doc__, steps=628)
    p.add_argument("--period", type=float, default=PERIOD, help=f"vortex period T (default {PERIOD})")
    args = p.parse_args()
    flow = pfmls.flows.LeVequeVortex2D(period=args.period)
    seeds = measure.circle_points((0.5, 0.75), 0.15, n=20_000)
    checks = {}
    n_period = round(args.period / args.dt)
    if abs(n_period * args.dt - args.period) < 1e-9 * args.period and n_period <= args.steps:
        checks[n_period] = (f"initial circle at t = T (step {n_period})", measure.circle_points((0.5, 0.75), 0.15))
    half = round(0.5 * args.period / args.dt)
    report = sorted({5, half - 4, half, half + 1, n_period, args.steps} if args.period != PERIOD
                    else {5, 310, 314, 315, 628})
    name = "leveque" if args.period == PERIOD else f"leveque_T{args.period:g}"
    run(args, name, flow, pfmls.sdf.circle((0.5, 0.75), 0.15),
        reference=lambda t: measure.advect_points(seeds, flow.velocity_numpy, 0.0, t, dt=0.005),
        shape_checks=checks, report_steps=tuple(s for s in report if 0 < s <= args.steps))


if __name__ == "__main__":
    main()
