"""Cache of captured CUDA graphs, keyed by step type.

The level sets (:class:`pfmls.PFMLevelSet2D`, :class:`pfmls.PFMLevelSet3D`) and the 2D free-surface solver run each
kind of step through :meth:`GraphCache.run`: the first occurrence of a key runs eagerly (it compiles and loads the
Warp modules, which cannot happen during a capture), the second is captured into a CUDA graph, and later ones
replay it. Keys contain whatever changes the launched work, e.g. the step type and the addresses of the velocity
arrays, so a caller that reallocates its arrays every step never replays a graph.
"""

from __future__ import annotations

import warnings
from typing import Callable, Hashable

import warp as wp


class GraphCache:
    """CUDA graphs keyed by step type (eager everywhere when disabled or off CUDA).

    Parameters
    ----------
    device:
        Warp device of the captured work.
    max_graphs:
        Bound on the cached graphs (the oldest is dropped) and, times 4, on the remembered keys that ran eagerly
        but were not captured yet.
    enabled:
        Capture at all (ignored on the CPU device).
    churn_warning:
        Message of the ``RuntimeWarning`` issued when the remembered keys reach their bound, i.e. when the keys
        keep changing and graphs are rarely replayed (``None`` = no warning).
    """

    def __init__(self, device, max_graphs: int = 32, enabled: bool = True, churn_warning: str | None = None):
        self.device = wp.get_device(device)
        self.enabled = bool(enabled) and self.device.is_cuda
        self.max_graphs = int(max_graphs)
        self.churn_warning = churn_warning
        #: captured graphs by key (insertion order = age)
        self.graphs: dict = {}
        #: keys that ran eagerly once and are captured at their next occurrence
        self.warm: set = set()

    def __len__(self) -> int:
        return len(self.graphs)

    def clear(self) -> None:
        """Forget every graph and key (call after reallocating arrays that captured work reads)."""
        self.graphs.clear()
        self.warm.clear()

    def run(self, key: Hashable, body: Callable[[], None]) -> None:
        """Run ``body`` (device work only, no host reads): eagerly, captured, or as a replay of the graph of ``key``.

        If a capture fails (e.g. a device without memory-pool support), the cache warns, disables itself and runs
        ``body`` eagerly from then on.
        """
        if not self.enabled:
            body()
            return
        graph = self.graphs.get(key)
        if graph is None:
            if key not in self.warm:
                if len(self.warm) >= 4 * self.max_graphs:
                    self.warm.clear()
                    if self.churn_warning:
                        warnings.warn(self.churn_warning, RuntimeWarning, stacklevel=3)
                self.warm.add(key)
                body()
                return
            try:
                with wp.ScopedCapture(device=self.device) as capture:
                    body()
            except Exception as exc:
                warnings.warn(f"CUDA graph capture failed ({exc}); running without graphs", RuntimeWarning,
                              stacklevel=3)
                self.enabled = False
                self.clear()
                body()
                return
            if len(self.graphs) >= self.max_graphs:
                del self.graphs[next(iter(self.graphs))]
            graph = self.graphs[key] = capture.graph
        wp.capture_launch(graph)
