"""Area and shape-error measurements for 2D level sets (host side, numpy).

* :func:`contour_area` - area of ``{phi < 0}`` as the signed shoelace sum over the marching-squares
  zero contours of ``phi`` (scikit-image), closed along the domain walls.
* :func:`zero_contour` - zero-contour points in physical coordinates.
* :func:`chamfer_hausdorff` - symmetric chamfer (mean of the two mean nearest-neighbour distances)
  and Hausdorff distance between two point clouds (needs scipy).
* Dense reference polylines for the classic benchmarks (Zalesak disk, circles, Distorted flow) and
  :func:`advect_points` to build a reference for any flow by accurate forward integration.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _find_contours(phi: np.ndarray, fully_connected: str = "low", positive_orientation: str = "low"):
    from skimage import measure

    return measure.find_contours(
        np.asarray(phi, dtype=np.float64).T, 0.0, fully_connected=fully_connected,
        positive_orientation=positive_orientation,
    )


def contour_area(phi: np.ndarray, h: float | tuple[float, float], close_boundary: bool = True) -> float:
    """Area of the region ``phi < 0`` for a cell-centred field ``phi[i, j]`` with spacing ``h``.

    Shoelace formula over the zero contours (``fully_connected='low'``,
    ``positive_orientation='high'``). With ``close_boundary`` the field is
    padded by one ghost cell holding ``|phi|`` of the adjacent boundary cell, so a region touching the
    domain boundary is closed exactly on the wall (half a cell beyond the last cell centre); for an
    interface inside the domain the result is unchanged. Without it, contours that reach the
    boundary are open and the result is meaningless.
    """
    hx, hy = (h, h) if np.isscalar(h) else h
    phi = np.asarray(phi, dtype=np.float64)
    offset = 0.0
    if close_boundary:
        padded = np.pad(np.abs(phi), 1, mode="edge")
        padded[1:-1, 1:-1] = phi
        phi, offset = padded, 1.0
    total = 0.0
    for c in _find_contours(phi, "low", "high"):
        # c[:, 0] = row = y index, c[:, 1] = col = x index (field is transposed)
        y, x = c[:, 0] - offset, c[:, 1] - offset
        total += 0.5 * np.sum((x[1:] + x[:-1]) * (y[1:] - y[:-1]))
    return float(total * hx * hy)


def zero_contour(phi: np.ndarray, h: float, origin: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """All zero-contour points of a cell-centred field as an ``(n, 2)`` array of ``(x, y)``."""
    pts = [np.stack([(c[:, 1] + 0.5) * h + origin[0], (c[:, 0] + 0.5) * h + origin[1]], axis=1)
           for c in _find_contours(phi)]
    if not pts:
        return np.zeros((0, 2))
    return np.concatenate(pts, axis=0)


def chamfer_hausdorff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Symmetric chamfer distance ``(mean d(a,B) + mean d(b,A)) / 2`` and Hausdorff distance."""
    from scipy.spatial import cKDTree

    if len(a) == 0 or len(b) == 0:
        return float("inf"), float("inf")
    dab = cKDTree(b).query(a, workers=-1)[0]
    dba = cKDTree(a).query(b, workers=-1)[0]
    return float(0.5 * (dab.mean() + dba.mean())), float(max(dab.max(), dba.max()))


# ---------------------------------------------------------------------------
# Reference polylines
# ---------------------------------------------------------------------------


def circle_points(center=(0.5, 0.75), radius: float = 0.15, n: int = 200_000) -> np.ndarray:
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([center[0] + radius * np.cos(th), center[1] + radius * np.sin(th)], axis=1)


def slotted_disk_points(center=(0.5, 0.75), radius: float = 0.15, slot_width: float = 0.05,
                        slot_top: float = 0.85, n: int = 200_000) -> np.ndarray:
    """Boundary of Zalesak's slotted disk (circle minus the slot opening, slot sides and top)."""
    cx, cy = center
    sl, sr = cx - 0.5 * slot_width, cx + 0.5 * slot_width
    sb = cy - np.sqrt(radius**2 - (0.5 * slot_width) ** 2)
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    px, py = cx + radius * np.cos(th), cy + radius * np.sin(th)
    keep = ~((np.abs(px - cx) < 0.5 * slot_width) & (py < slot_top))
    ys = np.linspace(sb, slot_top, n // 20)
    xs = np.linspace(sl, sr, n // 50)
    return np.concatenate([
        np.stack([px[keep], py[keep]], axis=1),
        np.stack([np.full_like(ys, sl), ys], axis=1),
        np.stack([np.full_like(ys, sr), ys], axis=1),
        np.stack([xs, np.full_like(xs, slot_top)], axis=1),
    ])


def rotate_points(pts: np.ndarray, angle: float, center=(0.5, 0.5)) -> np.ndarray:
    """Rotate points counter-clockwise by ``angle`` about ``center``."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return (pts - np.asarray(center)) @ rot.T + np.asarray(center)


def distorted_points(t: float, u0: float = 0.4, center=(0.5, 0.5), r0: float = 0.0,
                     circle_center=(0.5, 0.75), radius: float = 0.2, n: int = 200_000) -> np.ndarray:
    """Exact solution of ``u = u0 (y - cy, -(x - cx)) / (r + r0)`` applied to a circle.

    Each point rotates clockwise about ``center`` with angular velocity ``u0 / (r + r0)``.
    """
    pts = circle_points(circle_center, radius, n) - np.asarray(center)
    r = np.linalg.norm(pts, axis=1)
    a = -u0 * t / (r + r0)
    c, s = np.cos(a), np.sin(a)
    return np.stack([pts[:, 0] * c - pts[:, 1] * s, pts[:, 0] * s + pts[:, 1] * c], axis=1) + np.asarray(center)


def advect_points(pts: np.ndarray, velocity: Callable[[np.ndarray, np.ndarray, float], tuple],
                  t0: float, t1: float, dt: float = 1.0e-3) -> np.ndarray:
    """Integrate points through ``velocity(x, y, t) -> (u, v)`` from t0 to t1 with classical RK4."""
    n = max(1, int(np.ceil(abs(t1 - t0) / dt)))
    h = (t1 - t0) / n
    x, y = pts[:, 0].astype(np.float64).copy(), pts[:, 1].astype(np.float64).copy()
    t = t0
    for _ in range(n):
        u1, v1 = velocity(x, y, t)
        u2, v2 = velocity(x + 0.5 * h * u1, y + 0.5 * h * v1, t + 0.5 * h)
        u3, v3 = velocity(x + 0.5 * h * u2, y + 0.5 * h * v2, t + 0.5 * h)
        u4, v4 = velocity(x + h * u3, y + h * v3, t + h)
        x += h / 6.0 * (u1 + 2 * u2 + 2 * u3 + u4)
        y += h / 6.0 * (v1 + 2 * v2 + 2 * v3 + v4)
        t += h
    return np.stack([x, y], axis=1)
