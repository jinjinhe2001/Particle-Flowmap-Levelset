"""Zalesak's slotted disk in rigid rotation (two revolutions in 628 steps of dt = 0.02).

Advection only (the paper's Table setting):
    python examples/benchmarks/zalesak.py --protocol advection
Full method (long/short reinitialisation every 12/6 steps):
    python examples/benchmarks/zalesak.py
The particle band is 3 dx wide and the particle -> grid reconstruction uses quadratic B-spline weights
by default (override with --band and --opt p2g_weights=uniform).
"""

import math

from _common import make_parser, run

import pfmls
from pfmls.core import measure

OMEGA = math.pi / (157 * 0.02)


def main():
    args = make_parser(__doc__, steps=630, band=3.0).parse_args()
    shape = pfmls.sdf.slotted_disk(center=(0.5, 0.75), radius=0.15, slot_width=0.05, slot_top=0.85)
    disk = measure.slotted_disk_points()
    run(args, "zalesak", pfmls.flows.Rotation2D(omega=OMEGA, center=(0.5, 0.5)), shape,
        reference=lambda t: measure.rotate_points(disk, OMEGA * t),
        report_steps=(5, 310, 314, 315, 628, 630), options={"p2g_weights": "bspline"})


if __name__ == "__main__":
    main()
