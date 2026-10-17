"""Droplet splash: a drop (r = 0.08) falls from (0.5, 0.75) into a pool of depth 0.3525 (g = 1).

This is the free-surface setup of the PFM-LS paper (200^2, dt = 5e-4, sigma = 2e-4, We ~ 508): the interface is
the PFM-LS level set, the velocity is transported by APIC.  The drop hits the pool at t ~ 0.8 and throws a crown
against the side walls; the crater closes, a central jet rebounds and collapses, and the clip ends at t = 3.75.

    python examples/water/droplet_splash.py                     # t = 3.75, MP4 in outputs/water2d/droplet_splash
    python examples/water/droplet_splash.py --sigma 1e-3 --gif
"""

import os
import sys

try:
    import pfmls  # noqa: F401
except ModuleNotFoundError:  # source checkout without `pip install -e .`
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from pfmls.fluid.runner import build_parser, exit_code, run  # noqa: E402
from pfmls.fluid.scenes import droplet_splash  # noqa: E402


def main(argv=None):
    args = build_parser("droplet_splash", __doc__, res=200, t_end=3.75, sigma=2.0e-4).parse_args(argv)
    scene = droplet_splash(sigma=args.sigma, res=args.res, t_end=args.t_end)
    return exit_code(run(scene, args))


if __name__ == "__main__":
    sys.exit(main())
