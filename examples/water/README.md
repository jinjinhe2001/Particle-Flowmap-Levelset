# Water examples (2D free surface)

Two incompressible free-surface flows on a MAC grid. The interface is the PFM-LS level set (`pfmls.PFMLevelSet2D`,
configured by `pfmls.fluid.PFMBackendConfig`), the velocity is transported with APIC (Jiang et al. 2015), and the
pressure is a ghost-fluid Poisson solve with the surface-tension jump `p = sigma kappa`, free-slip walls and
Jacobi-preconditioned CG on the device.

| script | scene | default | RTX 4090 |
|---|---|---|---|
| `droplet_splash.py` | The paper's free-surface setup: a drop of radius 0.08 falls from (0.5, 0.75) into a pool of depth 0.3525 (We ~ 508), throws a crown against the side walls, and a central jet rebounds and collapses. | 200^2, t = 3.75, dt = 5e-4, sigma = 2e-4 | 10 s (13 s with the MP4), 0.6 GB |
| `dam_break.py` | A water column a x 2a (a = 0.25) collapses (Martin & Moyce 1952); the front hits the right wall at t ~ 1.1, a jet runs up to the lid, and a plunging breaker lands at t ~ 3.2 and runs back to the left wall. | 400^2, t = 6, dt = 2.5e-4, sigma = 1e-3 | about 1 min, 1.0 GB |

Times are for the stepping loop on an idle GPU (`wall_s` in `summary.json`) once the kernels are compiled (the first
run compiles them, about 30 s); the memory is the peak GPU memory in use.

```
python examples/water/droplet_splash.py                     # MP4 in outputs/water2d/droplet_splash
python examples/water/droplet_splash.py --gif --style dark --particles
python examples/water/droplet_splash.py --sigma 1e-3 --t-end 2
python examples/water/droplet_splash.py --no-render         # measurements only
python examples/water/dam_break.py                          # MP4 in outputs/water2d/dam_break
```

Each run writes into `--out` (default `outputs/water2d/<scene>`) an H.264 MP4 (PNG frames if ffmpeg is missing),
`volume.csv` (volume, kinetic energy, CG iterations and time step for every step), `frames.csv` (rendered liquid
area and, for the dam break, the front position for every frame) and `summary.json` (status, volume drift,
timings, device and versions; for the dam break also the wall-hit time `wall_hit_t` and `wall_hit_tau`).

## Key options

- `--res`, `--t-end`, `--sigma`: resolution, end time and surface tension (defaults in the table above).
- `--no-render`, `--gif`, `--png`, `--style light|dark`, `--particles`: media.
- `--velocity sl`: semi-Lagrangian velocity transport instead of APIC.
- `--levelset-opt KEY=VALUE`: a `PFMBackendConfig` or `PFMLS2DOptions` field, e.g. `hessian_scale=1` (repeatable).
- `--ckpt-every N`, `--save-final`, `--resume ckpt.npz`: checkpoints.
- `--device cpu`: run on Warp's CPU device.

`--help` lists all flags by group.

## Status

Each run ends with a status line, also the exit code (0 only for `completed`): `completed` (no NaN repair, pressure
solve converged, volume within 1% and no step above 0.5%, kinetic energy at most 1.5 times what an inviscid flow
could have; thresholds `--drift-tol`, `--jump-tol`, `--energy-tol`), `degraded: ...` (a volume or energy criterion
missed) or `failed: ...` (a NaN repair or the pressure solve at its iteration cap; the run stops unless
`--keep-going`). Repeated default runs of each scene on an RTX 4090 ended `completed`, with a maximum volume drift
of 0.25-0.29% (splash) and 0.29-0.56% (dam break); the drift varies slightly from run to run.

## Limitations

- Liquid only: air has zero pressure, so enclosed air pockets (where the crown sheets reconnect with the side walls,
  or under the dam break's plunging breaker) collapse as if they were empty.
- Sheets and films thinner than about one cell are carried by the PFM-LS particles but contain no cell centre, so
  the pressure solve does not see them; they are the main source of the volume drift.
- In 2D a liquid neck does not pinch by the Rayleigh-Plateau instability.
- `--velocity sl` is not stable with strong surface tension; use it for gravity-dominated flows only.
- Only the default settings were validated; for other settings, the status says whether a run met the volume and
  energy criteria.

