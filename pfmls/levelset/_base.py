"""Dimension-independent parts of :class:`PFMLevelSet2D` and :class:`PFMLevelSet3D`.

The base class holds the reinitialisation schedule, the argument checks of :meth:`step`, ``init_from_sdf`` and
``scrub_nonfinite``, the CUDA-graph cache and the host-copy helpers. The subclasses allocate the dimension-specific
state and implement the step pipeline.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import warp as wp

from ..core.graphs import GraphCache

#: Warning of the graph cache when the velocity arrays change address at every step.
CHURN_WARNING = ("the velocity's face arrays keep changing address, so CUDA graphs are rarely replayed; update the "
                 "MAC arrays in place or reuse a few buffers")

REINIT_UNITS = ("steps", "time")


def checked_sdf(sdf, coords: tuple[np.ndarray, ...], where: str) -> np.ndarray:
    """``sdf(*coords)`` as float32, checked for the inputs' shape and for finite values (``ValueError``)."""
    call = f"sdf({', '.join('xyz'[:len(coords)])})"
    v = np.asarray(sdf(*coords), dtype=np.float32)
    if v.shape != coords[0].shape:
        raise ValueError(f"{call} returned shape {v.shape} for inputs of shape {coords[0].shape} at the {where}; "
                         f"it must be evaluated elementwise")
    bad = int(np.sum(~np.isfinite(v)))
    if bad:
        raise ValueError(f"{call} returned {bad} non-finite values at the {where}")
    return v


def checked_grid_sdf(sdf, shape: tuple[int, ...]) -> np.ndarray:
    """An array of cell-centre values as float32, checked for its shape and for finite values (``ValueError``)."""
    arr = np.asarray(sdf, dtype=np.float32)
    if arr.shape != tuple(shape):
        raise ValueError(f"grid SDF has shape {arr.shape}, expected {tuple(shape)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"grid SDF has {int(np.sum(~np.isfinite(arr)))} non-finite values")
    return arr


def check_sign_change(cell_values: np.ndarray) -> None:
    """``ValueError`` unless the cell values change sign (the level set has an interface inside the grid)."""
    if not (cell_values.min() < 0.0 <= cell_values.max()):
        raise ValueError("the initial level set has no interface inside the grid (no sign change)")


class PFMLevelSetBase:
    """Schedule, argument checks, graph cache and host copies shared by the 2D and 3D level sets.

    Subclasses set ``grid``, ``_options`` and ``_scrub_counter`` and call :meth:`_init_base` from ``__init__``.
    """

    #: band half-width (dx) above which the constructor warns
    WIDE_BAND = 4.0
    #: bound on the cached CUDA graphs (see :class:`pfmls.core.graphs.GraphCache`)
    MAX_GRAPHS = 32

    def _init_base(self, band: float, reinit_long: float, reinit_short: float, reinit_unit: str, device,
                   use_graph: bool, band_note: str) -> None:
        if not band > 0.0:
            raise ValueError(f"band must be > 0, got {band}")
        if band > self.WIDE_BAND:
            warnings.warn(f"band={band} dx: {band_note}", RuntimeWarning, stacklevel=3)
        if reinit_unit not in REINIT_UNITS:
            raise ValueError(f"reinit_unit must be 'steps' or 'time', got {reinit_unit!r}")
        self._band = float(band)
        self.reinit_unit = reinit_unit
        self.reinit_long = reinit_long
        self.reinit_short = reinit_short
        self._check_intervals()
        self.device = wp.get_device(device)
        self._graphs = GraphCache(self.device, self.MAX_GRAPHS, use_graph, CHURN_WARNING)
        self.t = 0.0
        self.step_index = 0
        self._i_long = 0
        self._t_long = 0.0
        self._t_short = 0.0
        self._initialized = False
        self._fine: dict = {}

    # ------------------------------------------------------------------ properties

    @property
    def band(self) -> float:
        """Particle band half-width in dx (fixed at construction)."""
        return self._band

    @property
    def options(self):
        """The (immutable) advanced options."""
        return self._options

    @property
    def use_graph(self) -> bool:
        """Whether steps are captured into CUDA graphs (False on the CPU and after a failed capture)."""
        return self._graphs.enabled

    # ------------------------------------------------------------------ schedule

    def _check_intervals(self) -> None:
        for name in ("reinit_long", "reinit_short"):
            v = getattr(self, name)
            if v < 0:
                raise ValueError(f"{name} must be >= 0, got {v}")
            if self.reinit_unit == "steps" and v != int(v):
                raise ValueError(f"{name} must be an integer number of steps, got {v} (use reinit_unit='time')")

    def _schedule(self, dt: float) -> tuple[bool, bool]:
        """(long due, short due) for the next step of size ``dt``."""
        self._check_intervals()
        if self.reinit_unit == "steps":
            i = self.step_index + 1
            long = self.reinit_long > 0 and i % int(self.reinit_long) == 0
            short = self.reinit_short > 0 and (i - self._i_long) % int(self.reinit_short) == 0
        else:
            t1, tol = self.t + dt, 0.5 * dt
            long = self.reinit_long > 0 and t1 - self._t_long >= self.reinit_long - tol
            short = self.reinit_short > 0 and t1 - self._t_short >= self.reinit_short - tol
        return long, short

    def reinit_due(self, dt: float) -> tuple[bool, bool]:
        """``(long, short)``: the reinitialisations the next :meth:`step` of size ``dt`` will run."""
        long, short = self._schedule(dt)
        return long, long or short

    def _begin_step(self, dt: float, velocity, skip_long_reinit: bool) -> tuple[bool, bool, int, float]:
        """Check the arguments of a step (``ValueError`` before any work) and advance the schedule.

        Returns ``(long, short, i, t1)``: the reinitialisations of the step, its index and its end time.
        """
        if not self._initialized:
            raise RuntimeError("call init_from_sdf() first")
        if not (math.isfinite(dt) and dt >= 0.0):
            raise ValueError(f"dt must be a finite number >= 0, got {dt}")
        g = self.grid
        velocity.check_grid(*g.shape, g.dx)
        for face in velocity.mac_arrays(self.device):
            if face.device != self.device:
                raise ValueError(f"the velocity's arrays live on {face.device}, the level set runs on {self.device}")
        long_due, short_due = self._schedule(dt)
        long = long_due and not skip_long_reinit
        short = long or short_due
        i, t1 = self.step_index + 1, self.t + dt
        if long:
            self._i_long = i
        if long_due:
            self._t_long = t1
        if short:
            self._t_short = t1
        return long, short, i, t1

    def _reset_clock(self) -> None:
        self.t, self.step_index = 0.0, 0
        self._i_long, self._t_long, self._t_short = 0, 0.0, 0.0
        self._graphs.clear()

    def _clock_scalars(self) -> dict:
        return {"t": self.t, "step_index": self.step_index, "i_long": self._i_long, "t_long": self._t_long,
                "t_short": self._t_short, "dx": self.grid.dx}

    def _set_clock(self, s: dict) -> None:
        self.t, self.step_index = float(s["t"]), int(s["step_index"])
        self._i_long, self._t_long, self._t_short = int(s["i_long"]), float(s["t_long"]), float(s["t_short"])
        self._initialized = True

    def _scrub_target(self, counter: wp.array | None) -> wp.array:
        """The counter of :meth:`scrub_nonfinite` (``ValueError`` unless a 1D int32 array with >= 2 entries on the
        level set's device)."""
        if counter is None:
            return self._scrub_counter
        if counter.device != self.device or counter.dtype != wp.int32 or counter.ndim != 1 or counter.shape[0] < 2:
            raise ValueError(f"counter must be a 1D int32 array with >= 2 entries on {self.device}, got "
                             f"{counter.dtype.__name__}{tuple(counter.shape)} on {counter.device}")
        return counter

    # ------------------------------------------------------------------ helpers

    def _launch(self, kernel, dim, inputs):
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)

    def _w_inv_h(self, h: float) -> float:
        """Weight parameter for a reconstruction onto a grid of spacing ``h`` (0 = uniform)."""
        return 0.0 if self._options.p2g_weights == "uniform" else 1.0 / h

    def _host(self, a: wp.array) -> np.ndarray:
        """Host copy of a device array (``.numpy()`` is a zero-copy view on the CPU device)."""
        out = a.numpy()
        return out.copy() if self.device.is_cpu else out

    def phi_grid(self) -> np.ndarray:
        """Background grid level set (cell centres) as a numpy array of the grid's shape (a copy)."""
        return self._host(self.phi_g)
