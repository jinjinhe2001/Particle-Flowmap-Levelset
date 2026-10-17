"""Circle in the steady differential rotation u = u0 (y - 0.5, 0.5 - x) / r, u0 = 0.4 (spiral to t = 6).

Advection only:  python examples/benchmarks/distorted.py --protocol advection
Full method:     python examples/benchmarks/distorted.py
Stronger shear:  python examples/benchmarks/distorted.py --u0 0.8
"""

from _common import make_parser, run

import pfmls
from pfmls.core import measure


def main():
    p = make_parser(__doc__, steps=300)
    p.add_argument("--u0", type=float, default=0.4, help="rotation speed scale u0 (default 0.4)")
    args = p.parse_args()
    name = "distorted" if args.u0 == 0.4 else f"distorted_u{args.u0:g}"
    run(args, name, pfmls.flows.Distorted2D(u0=args.u0, center=(0.5, 0.5), r0=0.0),
        pfmls.sdf.circle((0.5, 0.75), 0.2),
        reference=lambda t: measure.distorted_points(t, u0=args.u0, center=(0.5, 0.5), radius=0.2),
        report_steps=tuple(s for s in (5, args.steps // 2, args.steps) if s > 0))


if __name__ == "__main__":
    main()
