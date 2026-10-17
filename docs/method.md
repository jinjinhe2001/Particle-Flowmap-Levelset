# PFM-LS: method, parameters and limitations

This page describes `pfmls.PFMLevelSet2D` (sections 1-3), `pfmls.PFMLevelSet3D` (section 4) and their limitations
(section 5). Accuracy figures come from the scripts in `examples/benchmarks` (200^2, 16 particles per cell,
dt = 0.02 unless stated; `--opt KEY=VALUE` sets an options field). Timings and memory were measured on an RTX 4090
with a separate timing harness, and a few single figures (the stationary circle, the contour loops, the inflow
wall, the small and the near-wall spheres) come from short one-off runs. The benchmark results are in the
[README](../README.md).

## 1. Method summary

**State.** Particles occupy the slots of a regular `a x a` lattice per cell (`ppc = a^2`). At every long
reinitialisation the slots with `|phi| < band * dx` become active; they form the narrow band. Each active particle
carries:

- its level-set value `phi_p`, constant between long reinitialisations;
- a unit gradient `g_p = normalize(J_long^T g0_p)`, transported by the long flow map;
- the short-map Jacobian `J_s` (RK4 integration of `dJ/dt = -J grad u`);
- a normal-gradient matrix `C = J_s^T C0 J_s`, where `C0` is the gradient of the unit normal at the last short
  reinitialisation.

A background grid level set `phi_g` (cell centres) is advected semi-Lagrangian (RK2) and redistanced every step.

**Per step:**
1. Long reinitialisation every `f_L` (default 12) steps:
   - particle-to-grid reconstruction at cell centres;
   - eikonal redistancing plus quasi-Newton closest-point projection near the interface (particle interpolation,
     Hermite fallback; the projection follows GARM-LS, Li et al. 2023);
   - Hermite semi-Lagrangian eikonal sweeps elsewhere;
   - reseed every lattice slot: the Hermite grid value is blended with the old particles' values, using the normal
     agreement `gamma = n_H . n_P` (e1 = 0.9, e2 = 0.7);
   - reset the long map.
2. Short reinitialisation every `f_S` (default 6) steps after the last long one: `C0` from MAC-averaged unit
   normals (quadratic B-spline derivative), then reset the short map.
3. RK4 advection of positions and `J_s`, then the update of `C` and `g` (in the same kernel).
4. Grid semi-Lagrangian advection and redistancing.
5. Composite: every grid cell with a particle within `p2g_radius * dx` (0.25) of its centre takes the particle
   reconstruction `phi_p + normalize(g_p + C_p d) . d`. Every other cell keeps the grid value.
6. Adaptive seeding (on by default, `seed_min = ppc / 2`; below).

**Pure advection mode** (`reinit_long = reinit_short = 0`). The reinitialisations run only once, at `t = 0`, and
the lattice particles carry the exact initial values along the flow map for the whole run. This is the paper's
"advection only" setting. Adaptive seeding (`seed_min = ppc / 2` by default, the paper's `n_min`) keeps the
stretched band covered.

**Adaptive seeding.** Every step, each band cell (`|phi_g| < band dx`) holding fewer than `seed_min` particles
gets one candidate at a random point. Its value is the weighted mean of the first-order predictions
`phi_p + G_p . d` of the lattice particles in the surrounding 3x3 cells, with the transported gradient
`G_p = J_long^T g0`; a unit normal would underestimate the change of `phi` across a compressed filament and lose
the sign inside it. Its gradient and `C` are the weighted means of `G_p` and `C_p`, its maps start at the
identity. A candidate is rejected without neighbours, beyond the band (mean `v_p / |G_p|` of at least `band dx`),
when the predictions disagree in sign (`|sum w v_p| < seed_coherence sum w |v_p|`), or when no neighbour carries
the predicted sign (the 3D rule, section 4.1, checks the background grid's sign instead, which the advection
protocol loses in thin filaments). The accepted particles (newborns) take idle slots and live until the next long
reinitialisation. By default they only fill holes in the fine reconstruction and never enter the dynamics (a short
reinitialisation refits `C0` of the lattice particles only; a newborn's transported `C` becomes its `C0`), so a
seeded run evolves bitwise the same grid and lattice particles as an unseeded one
(`tests/test_pfmls2d.py::test_seeding_is_an_output_filter`); in the composite they would also fill the grid, but
leave specks of ghost liquid at `t = T` (LeVeque: area +1.9e-4, Hausdorff 18.7 dx).

**Implementation.**
- The velocity is evaluated at the Runge-Kutta stage times (`t, t+dt/2, t+dt/2, t+dt` for the particles).
- Eikonal redistancing uses parallel Jacobi iterations of the first-order upwind update, limited to a band
  (48 iterations at reinitialisation, 16 per step; distances converge within about 34 and 11 cells). The far field
  keeps its sign-correct semi-Lagrangian values, and the input sign is always kept.
- The quasi-Newton projection starts at `x_s - phi n`, stops at a tolerance of `1e-4 dx` and takes at most 64 steps
  of at most 0.02 dx, i.e. 1.28 dx of travel, enough for bands up to about 4 dx.
- Cell lists are built by a counting sort without a per-cell cap and with a deterministic order within each cell,
  so runs are bitwise reproducible and CUDA-graph replay equals eager launches.
- No NaN paths: normalisation returns the zero vector for a zero or non-finite input, and the composite never
  writes a non-finite value into the grid. `scrub_nonfinite()` repairs a state corrupted upstream.
- Areas are marching-squares contours of the 5x fine reconstruction, closed on the domain walls. Shape errors
  (chamfer and Hausdorff) are measured against the exact solution at the final time, for LeVeque also against the
  initial circle at the step that reaches t = T exactly (628 for dt = 0.02).

## 2. Parameter guide

| Parameter | Default | Meaning / advice |
|---|---|---|
| `res`, `dx` | 200, `1/nx` | Grid. |
| `ppc` | 16 | A perfect square of at least 9 (9, 16, 25, ...). 16 is the paper setting. 9 is cheaper but less accurate once the band is deformed (LeVeque full method 99.41% instead of 99.65%), and in advection-only runs the stretched band no longer covers the cell centres (LeVeque returns at 0.912 without seeding, at 0.993 with specks inside the disk with it): use 16 or more there. |
| `band` | 2.0 | Particle band half-width in dx: 1-2 for deforming flows, 3 for rigid motion (Zalesak). Wider bands lose accuracy in stretching flows (LeVeque full method: 99.65% at 2 dx, 99.05% at 4 dx, 96.75% at 6 dx); below 1 dx, cells near a stretched interface are left without particles. The constructor warns above 4 dx. |
| `reinit_long`, `reinit_short` | 12, 6 | Reinitialisation intervals (section 3). |
| `reinit_unit` | `"steps"` | `"time"` gives the intervals in time units. |
| `hessian_scale` | 1.0 | Scale of `C0`. 0.5 is Taylor-consistent and recommended with frequent reinitialisation (section 3). |
| `p2g_radius` | 0.25 | Radius of every particle-to-point reconstruction (composite, P2G, reseed and Newton KNN), in dx, within (0, 0.5). |
| `p2g_weights` | `"uniform"` | `"bspline"` (the Zalesak script's default) lowers the advection-only area noise of the slotted disk by 2-10x. |
| `frozen_time` | False | Evaluate the velocity at the start of the step in all RK stages (first order in time). |
| `newton_max_iter`, `newton_tol`, `newton_step_clamp` | 64, 1e-4 dx, 0.02 dx | Quasi-Newton projection. Raise the cap (e.g. 1000) for bands wider than 4 dx: at band 6, LeVeque ends at 98.74% instead of 96.75%. |
| `eikonal_iterations`, `grid_eikonal_iterations` | 48, 16 | Jacobi iterations of the eikonal guess at reinitialisation and of the per-step grid redistancing. `None` converges on the whole grid (0.6-0.7 ms per step more at 200^2 for both). |
| `use_graph` | True | Capture each step type into a CUDA graph: at 200^2 on an RTX 4090, about 5x faster than eager launches in advection-only mode without seeding and 2.5x for the full method. |
| `seed_min` | `ppc / 2` | Adaptive seeding threshold in particles per band cell (`None` = `ppc / 2`, the paper's `n_min`; 0 = off). In advection-only runs LeVeque keeps 99.5% of its area at maximum stretch (92.0% without) and returns to the same circle; with the full method it changes little, because the long reinitialisation reseeds the band. The free-surface backend keeps it off. `--seed-min` of the benchmark scripts overrides it. |
| `seed_coherence` | 0.9 | Sign agreement required of the neighbours' predictions (0 = no check, which leaves ghost specks at `t = T`). |
| `composite_newborns`, `reconstruct_newborns` | `"off"`, `"fallback"` | How newborns enter the per-step composite and the fine reconstruction: `"off"`, `"fallback"` (only at points that no lattice particle covers) or `"mixed"`. |

Usage notes:
- The options (keyword arguments or `options=PFMLS2DOptions(...)`) and `band` are fixed at construction, because
  they are baked into the captured graphs. The intervals and the flow parameters (`Flow2D.params()`, uploaded
  every step) may change between steps.
- CUDA graphs are cached per step type and per velocity face arrays, keyed on the array addresses: update MAC
  arrays in place or cycle through a few buffers. The cache holds at most 32 graphs and warns when the arrays keep
  changing address.
- `step` and `init_from_sdf` check their inputs (finite `dt >= 0`, matching grid and device, a sign change in the
  initial field) and raise `ValueError` before any work is launched, so a bad call leaves the state intact.
- `snapshot()` / `restore()` (device copies) and `state_dict()` / `load_state_dict()` (host copies) checkpoint the
  state; `reinit_due(dt)` and `step(dt, flow, skip_long_reinit=True)` let a caller drop a scheduled long
  reinitialisation, for example after a rejected step.
- `phi_g` and the particle arrays may be edited between steps; the class docstring says what the next step sees.

## 3. Reinitialisation frequency and `dt`

Each long reinitialisation perturbs the interface a little:

- **Convex shapes shrink.** For a signed distance the extrapolation `phi_p + normalize(g_p + C d) . d` equals
  `phi_p + g.d + d^T C d`, twice the second-order Taylor term, so the reconstruction is biased at every reseed. A
  stationary circle of radius 10 cells loses about 1e-4 of its area per long reinitialisation with
  `hessian_scale = 1`; with `hessian_scale = 0.5` the drift is about 5e-6, of either sign
  (`tests/test_pfmls2d.py::test_half_hessian_reduces_reinit_drift`).
- **Stretched filaments gain area.**

With intervals counted in steps, a smaller `dt` therefore means more reinitialisations per unit time and larger
errors. Area ratio at the end and shape error against the exact solution in dx (LeVeque: `leveque.py --dt 0.00125
--steps 10048`, read at t = T; Zalesak: `zalesak.py --dt 0.00125 --steps 5024`; then `--reinit-unit time
--reinit-long 0.24 --reinit-short 0.12` and `--opt hessian_scale=0.5`):

| case | 12/6 steps at dt = 0.02 | 12/6 steps at dt = 0.00125 | 0.24/0.12 time units | + `hessian_scale = 0.5` |
|---|---|---|---|---|
| LeVeque, t = T | 0.99650 (0.214 / 1.07) | 1.01568 (1.08 / 4.49) | 0.99691 (0.211 / 1.08) | 0.99783 (0.198 / 1.07) |
| Zalesak, 1 revolution | 0.99962 (0.029 / 0.44) | 0.99367 (0.112 / 1.74) | 0.99958 (0.029 / 0.46) | 0.99998 (0.028 / 0.34) |

`hessian_scale = 0.5` alone (still 12/6 steps at dt = 0.00125) removes the convex drift (Zalesak 1.00016) but not
the filament gain (LeVeque 1.029). Recommendations:

- Keep the interval between long reinitialisations near the validated 0.24 time units, or at a comparable
  interface displacement of a few tens of cells: scale step intervals with `1/dt` or use `reinit_unit="time"`.
- Use `hessian_scale = 0.5` whenever the intervals are short.
- The defaults (12/6 steps, scale 1) are the paper's benchmark setting at dt = 0.02.

**Free-surface water.** The free-surface backend (`pfmls.fluid.PFMBackendConfig`) uses 12 / 6 nominal steps of the
water scenes' `dt = 0.1 dx`, converted to time units so that a CFL-limited step does not add reinitialisations. At
200^2 that is a long reinitialisation every 0.006 time units, 40 times more often than in the benchmarks, so the
backend uses `hessian_scale = 0.5`. Other intervals: `PFMBackendConfig(reinit_unit="time", reinit_long=...,
reinit_short=...)`; the water examples were validated with the defaults only.

## 4. 3D: `PFMLevelSet3D`

### 4.1 Summary

The pipeline is the 2D one with `a x a x a` particle lattices (`ppc = a^3`, default 64) and these differences:

- composite and output reconstruction (`mesh`, `volume`, `reconstruct`): the mean particle Taylor value of the
  lattice particles within 0.4 dx (`composite_radius`, `reconstruct_radius`); output points without one take
  seeded particles, then the trilinear background grid; meshes and volumes come from marching cubes;
- short reinitialisation: `C0 = 0.5 grad n` from a weighted least-squares fit of the particle normals within
  0.5 dx;
- long reinitialisation: as in 2D, with a tricubic Hermite fallback and at most 16 quasi-Newton steps within a
  trust radius; the Newton band covers every grid node the reseeding reads;
- positions are clamped to the domain.

**Band storage.** Particle slots are allocated for the band, never for `res^3 ppc`. At every long
reinitialisation the seed cells (centres within `band + sqrt(3) (1/2 - 1/(2a)) + 0.25` cells of the interface)
are compacted with a prefix sum, and each owns `ppc` slots, valid when `|phi| < band dx`. The initial capacity is
`capacity_factor` (2) x the initial need; if the band needs more, the storage grows to `growth_factor` (1.5) x the
need, keeping the particles. A step without seeding and without long reinitialisation never synchronises with the
host. Every `compact_every = 20` steps the particles are gathered into cell order.

**Adaptive seeding** (`seed_min > 0`, off by default): every step, each band cell holding fewer than `seed_min`
particles gets one candidate particle, whose value, normal and `C` are weighted Taylor averages of its neighbours.
Candidates are rejected without neighbours, outside the band, when the neighbours' normals disagree
(`seed_coherence` = 0.9) or when their sign differs from the background grid. By default the accepted particles
(newborns) only fill holes in the output reconstruction and never enter the dynamics, so a seeded run evolves
bitwise the same background grid and lattice particles as an unseeded one
(`tests/test_pfmls3d.py::test_seeding_is_an_output_filter`). The short reinitialisation then skips the newborns
(with `normal_gradient = "particles"`); unlike in 2D, their `C` returns to the `C0` of their birth.

### 4.2 Protocols and options

| protocol | `reinit_long / reinit_short` | `seed_min` | meaning |
|---|---|---|---|
| full method | 12 / 6 | 0 | long reinitialisation with redistancing and reseeding every 12 steps |
| research | 0 / 5 | 16 | long reinitialisation only at t = 0 (one long map for the whole run), short every 5 steps, adaptive seeding |
| advection only | 0 / 0 | 0 | both reinitialisations only at t = 0 |

`sphere3d.py` selects these with `--protocol full|research|advection` (`pfmls.levelset.pfmls3d.PROTOCOLS`, which
stores `seed_min` itself; `pfmls.levelset.pfmls2d.PROTOCOLS` stores a fraction of ppc). Unlike the 2D protocols, which
seed with `seed_min = ppc / 2` (section 1), the 3D full and advection protocols do not seed: the research protocol
is the seeded 3D setting. The 3D-specific options (`PFMLS3DOptions`):

| option | default | notes |
|---|---|---|
| `normal_gradient`, `normal_gradient_radius` | `"particles"`, 0.5 | `C0` from the particle normals within 0.5 dx. The 2D estimate (`"faces"`) smooths over about 2 dx and underestimates the curvature of radii below about 3 dx, so small droplets inflate at every long reinitialisation (radius 1.5 dx, one revolution at 32^3: +6.9% with `"faces"`, +0.53% with `"particles"`). |
| `hessian_scale` | **0.5** | Taylor-consistent (the 2D default is 1.0). |
| `composite_radius`, `reconstruct_radius` | 0.4, 0.4 | With 0.25, about 1.6% of the near-interface points of a deformed band have no particle. |
| `composite_newborns`, `reconstruct_newborns` | `"off"`, `"fallback"` | How newborns enter the composite and the output (`"off"`, `"fallback"` or `"mixed"`). Newborns in the composite feed the seeding and grow ghost liquid. |
| `p2g_radius` | lattice spacing `1 / a` | 0.25 for 64 ppc, 1/3 for 27 ppc. Neighbours at exactly the radius are excluded, so droplets at rest do not drift. |
| `newton_max_iter`, `newton_step_clamp`, `newton_max_step` | 16, 0.02, 1.0 | Trust radius starting at 0.02 dx, doubled while consecutive steps keep their direction, halved when they reverse, at most 1 dx. |
| `eikonal_init` | `"gradient"` | Interface cells start at `min(axis crossing, \|phi\| / \|grad phi\|)`; `"crossing"` overestimates oblique distances by up to `sqrt(3)`. |
| `init_normal_gradient` | True | `C0` from the initial particles before the initial reseed. |
| `newton_band`, `sweep_width` | 4.0, 6.0 | Newton band and width of the trilinear eikonal sweeps before it. |
| `seed_min`, `seed_coherence` | 0, 0.9 | Adaptive seeding. |
| `compact_every`, `cull_distance` | 20, band + 2 | Slot compaction and culling of newborns far from the interface. |
| `capacity_factor`, `growth_factor`, `seed_margin` | 2.0, 1.5, 0.25 | Band storage. |

The remaining options mean the same as in 2D. `ppc` must be a cube of at least 27 (64 is the benchmark value), and
bands wider than 4 dx trigger a warning. `phi_g` may be edited between steps (e.g. by a coupled solver); the edit
enters the next step's advection and the long reinitialisation's fill of cells without particles.

## 5. Limitations

2D:
- **Advection-only mode** is for reversing benchmark flows. Without seeding, particles exist only in the initial
  band and the grid supplies the far-field sign: once a feature is thinner than the grid, its sign is lost and
  cannot be repaired when the flow reverses (LeVeque with T = 16 returns at 0.88 without seeding; with it at 0.998,
  the newborns fill most of the lost core but leave specks, chamfer 5.3 dx; the full method gives 99.62%). `J_s` and `C` are never
  reset, so in flows that do not reverse `C` grows like `|J_s|^2` and dominates the normals.
- **Resolution.** Resolve the smallest feature by at least about 15 cells across its radius (LeVeque full method:
  91.8% at 64^2, 98.2% at 100^2, 99.65% at 200^2), keep the band at 1-3 dx, and do not expect reversibility once
  the flow has made films thinner than a cell (LeVeque with T = 24: 97.44%).
- **Fine reconstruction after strong deformation.** The fine field switches between particle and grid values pixel
  by pixel, which adds small spurious contour loops near the interface (LeVeque full method at step 630: 68 loops,
  each below 0.23 dx^2, net -1.3e-4 of the area). Renders can look slightly rough at the pixel scale.
- **Inflow walls.** The grid's semi-Lagrangian trace clamps to the domain (zero-gradient inflow), so liquid at a
  wall through which the velocity enters gains area (a circle tangent to the top wall under rotation: up to 13%,
  16% with the full method).
  Walls with `u . n = 0` (the free-surface solver and all benchmarks) and outflow walls are not affected.
- **Storage.** Every slot is stored (`ppc * res^2`, about 130 B per slot, 16 B more with adaptive seeding),
  although only 2-17% hold band particles; newborns use the idle slots.

3D:
- **Thin sheets with the full method.** The long reinitialisation cannot keep features thinner than a cell: at
  50^3 the deformed sphere ends one period at 100.87% with a flap (Hausdorff 2.2 dx). The same sphere centred at
  (0.16, 0.5, 0.5), next to a wall where the field pulls its quadrants apart, gains 4.8% in one period at 64^3 and
  1.9% at 128^3. For reversible benchmarks the research or advection protocol returns the shape to within 0.01 dx.
- **Research and advection protocols** are for reversible flows. With the research protocol the newborns
  accumulate until the end of the run (there is no long reinitialisation to reset them), so the storage can grow,
  and they bias the measured volume by O(h^2).
- **Coverage at maximum deformation.** At T/2 the sheet is thinner than a cell at 50^3. Points with no particle
  within 0.4 dx take the background grid, so unseeded shapes measure 4-11% too large there (the research
  protocol's evolved state measures 106.3% at 50^3, 100.97% once its newborns fill the holes). They return to the
  initial volume to within 1e-5 at T.
- **Memory.** About 252 B per slot; 64 ppc with `capacity_factor = 2` gives 640-768 slots per dx^2 of interface.
  Level-set memory at T/2 with the full method (`memory_bytes()`): 0.21 GiB at 50^3, 1.5 GiB at 128^3, 3.7 GiB at
  192^3. A growth briefly needs old and new arrays, and the capacity never shrinks: for large grids use 27 ppc and a
  smaller `capacity_factor`, and call `release_fine()` after measuring.
- **Coupling to your own velocity.** The particle arrays are reallocated when the storage grows, so re-read them
  after `step()` instead of caching them (`phi_g` and the grid arrays are never reallocated). As in 2D, update a
  MAC velocity in place.
