"""Volume, mesh and shape-error measurements for 3D level sets (host side, numpy + scikit-image).

* :func:`marching_cubes` - zero-isosurface mesh of a cell-centred field (scikit-image, Lewiner), cropped to
  the interface region and closed along the domain walls, in physical coordinates.
* :func:`mesh_volume` - enclosed volume by the divergence theorem.
* :func:`connected_components`, :func:`largest_component` - split a mesh into connected pieces.
* :func:`point_mesh_distance` - exact point-to-triangle distances (Warp BVH queries).
* :func:`sphere_errors` - symmetric chamfer distance and Hausdorff distance of a mesh to an analytic sphere.
* :func:`write_ply` - binary PLY output without extra dependencies.
"""

from __future__ import annotations

import numpy as np
import warp as wp


def marching_cubes(phi: np.ndarray, h: float, origin=(0.0, 0.0, 0.0), close_boundary: bool = True):
    """Mesh ``(verts (n, 3), faces (m, 3))`` of ``{phi = 0}`` for a cell-centred field ``phi[i, j, k]``.

    Node ``(i, j, k)`` sits at ``origin + (i + 0.5, j + 0.5, k + 0.5) h``. The field is cropped to the
    bounding box of ``{phi < 0}`` (plus one cell) before meshing. With ``close_boundary`` a ghost layer
    holding ``|phi|`` of the adjacent boundary cell closes surfaces that reach a domain wall exactly on
    the wall (half a cell beyond the last cell centre). Faces are oriented counter-clockwise seen from
    outside, so normals point out of ``{phi < 0}`` and :func:`signed_component_volumes` is positive.
    """
    from skimage import measure

    phi = np.asarray(phi, dtype=np.float32)
    mask = phi < 0.0
    empty = (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    if not mask.any():
        return empty
    lo, hi = [], []
    for ax in range(3):
        occupied = np.nonzero(mask.any(axis=tuple(a for a in range(3) if a != ax)))[0]
        lo.append(max(int(occupied[0]) - 1, 0))
        hi.append(min(int(occupied[-1]) + 2, phi.shape[ax]))
    sub = phi[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    pads = [(int(close_boundary and lo[ax] == 0), int(close_boundary and hi[ax] == phi.shape[ax])) for ax in range(3)]
    if any(p for pair in pads for p in pair):
        padded = np.pad(np.abs(sub), pads, mode="edge")
        padded[pads[0][0]:padded.shape[0] - pads[0][1], pads[1][0]:padded.shape[1] - pads[1][1],
               pads[2][0]:padded.shape[2] - pads[2][1]] = sub
        sub = padded
    if min(sub.shape) < 2 or not (sub > 0.0).any():
        return empty
    # "descent" orients the triangles towards decreasing values, i.e. out of {phi < 0} for (i, j, k)-ordered data
    verts, faces, _, _ = measure.marching_cubes(sub, 0.0, gradient_direction="descent")
    shift = np.array([lo[ax] - pads[ax][0] + 0.5 for ax in range(3)])
    return (verts.astype(np.float64) + shift) * h + np.asarray(origin, dtype=np.float64), faces.astype(np.int64)


def mesh_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Enclosed volume ``sum v0 . (v1 x v2) / 6`` of a closed, consistently oriented triangle mesh."""
    if len(faces) == 0:
        return 0.0
    v = np.asarray(verts, dtype=np.float64)
    v0, v1, v2 = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return float(abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()) / 6.0)


def signed_component_volumes(verts: np.ndarray, faces: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Signed enclosed volume of every connected component (outward orientation gives positive values)."""
    v = np.asarray(verts, dtype=np.float64)
    v0, v1, v2 = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    contrib = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0
    return np.bincount(labels[faces[:, 0]], weights=contrib, minlength=int(labels.max()) + 1 if len(labels) else 0)


def connected_components(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Component label of every vertex (connectivity through shared triangle edges)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as cc

    n = len(verts)
    if len(faces) == 0:
        return np.arange(n)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return cc(adj, directed=False)[1]


def largest_component(verts: np.ndarray, faces: np.ndarray):
    """The connected component of largest enclosed volume and a summary of the others.

    Returns ``(verts, faces, info)`` with ``info = {"components", "fragment_volume", "total_volume"}``.
    """
    labels = connected_components(verts, faces)
    vols = signed_component_volumes(verts, faces, labels)
    main = int(np.argmax(np.abs(vols)))
    keep_f = labels[faces[:, 0]] == main
    keep_v = labels == main
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[keep_v] = np.arange(int(keep_v.sum()))
    info = {"components": int(len(vols)), "fragment_volume": float(np.abs(vols).sum() - abs(vols[main])),
            "total_volume": float(abs(vols.sum()))}
    return verts[keep_v], remap[faces[keep_f]], info


@wp.kernel
def _closest_distance(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), max_dist: float, out: wp.array(dtype=float)):
    k = wp.tid()
    p = pts[k]
    q = wp.mesh_query_point_no_sign(mesh, p, max_dist)
    if q.result:
        out[k] = wp.length(p - wp.mesh_eval_position(mesh, q.face, q.u, q.v))
    else:
        out[k] = max_dist


def point_mesh_distance(points: np.ndarray, verts: np.ndarray, faces: np.ndarray, device=None,
                        max_dist: float = 1.0e6) -> np.ndarray:
    """Exact unsigned distance from every point to the triangle mesh ``(verts, faces)``."""
    dev = wp.get_device(device)
    mesh = wp.Mesh(points=wp.array(np.asarray(verts, dtype=np.float32), dtype=wp.vec3, device=dev),
                   indices=wp.array(np.asarray(faces, dtype=np.int32).ravel(), dtype=wp.int32, device=dev))
    pts = wp.array(np.asarray(points, dtype=np.float32), dtype=wp.vec3, device=dev)
    out = wp.zeros(len(pts), dtype=float, device=dev)
    wp.launch(_closest_distance, dim=len(pts), inputs=[mesh.id, pts, float(max_dist), out], device=dev)
    return out.numpy().astype(np.float64)


def fibonacci_sphere(center, radius: float, n: int = 200_000) -> np.ndarray:
    """``n`` nearly uniform points on a sphere."""
    i = np.arange(n) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / n)
    azim = np.pi * (1.0 + 5.0 ** 0.5) * i
    pts = np.stack([np.cos(azim) * np.sin(polar), np.sin(azim) * np.sin(polar), np.cos(polar)], axis=1)
    return np.asarray(center, dtype=np.float64) + radius * pts


def _summary(d_ab: np.ndarray, d_ba: np.ndarray) -> tuple[float, float]:
    return float(0.5 * (d_ab.mean() + d_ba.mean())), float(max(d_ab.max(), d_ba.max()))


def sphere_errors(verts: np.ndarray, faces: np.ndarray, center, radius: float, n_samples: int = 200_000,
                  device=None) -> dict:
    """Chamfer / Hausdorff distance between a mesh and the sphere ``|x - center| = radius``.

    ``surface``: exact distances (mesh vertices -> sphere analytically, sphere samples -> mesh triangles).
    ``points``: the point-cloud version (mesh vertices vs sphere samples, nearest neighbours), which
    overestimates the sphere -> mesh distances by up to half the vertex spacing.
    """
    from scipy.spatial import cKDTree

    if len(faces) == 0:
        return {"surface": (float("inf"),) * 2, "points": (float("inf"),) * 2}
    c = np.asarray(center, dtype=np.float64)
    d_mesh = np.abs(np.linalg.norm(verts - c, axis=1) - radius)
    samples = fibonacci_sphere(c, radius, n_samples)
    d_sph = point_mesh_distance(samples, verts, faces, device)
    d_cloud = cKDTree(verts).query(samples, workers=-1)[0]
    return {"surface": _summary(d_mesh, d_sph), "points": _summary(d_mesh, d_cloud)}


# ---------------------------------------------------------------------------
# Mesh output
# ---------------------------------------------------------------------------


def write_ply(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """Binary little-endian PLY (float32 vertices, int32 triangle indices)."""
    verts = np.asarray(verts, dtype="<f4")
    faces = np.asarray(faces, dtype="<i4")
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    rec = np.empty(len(faces), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    rec["n"] = 3
    rec["i"] = faces
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.tobytes())
        f.write(rec.tobytes())
