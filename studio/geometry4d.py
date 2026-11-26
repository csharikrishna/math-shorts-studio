"""Regular 4D polytopes, plane rotations in R^4, and perspective projection to R^3."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np

PLANE_AXES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2), "xw": (0, 3), "yw": (1, 3), "zw": (2, 3)}
CAMERA_W = 3.0  # the 4D "eye" sits at w = CAMERA_W; points nearer the eye project larger


def polytope(name: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Vertices (N x 4) and edges of a regular 4D polytope, centered at the origin."""
    if name == "tesseract":
        verts = np.array(list(product((-1.0, 1.0), repeat=4)))
        edges = [(i, j) for i, j in combinations(range(len(verts)), 2) if np.sum(verts[i] != verts[j]) == 1]
    elif name == "16-cell":
        verts = np.vstack([np.eye(4), -np.eye(4)]) * np.sqrt(2.0)
        edges = [(i, j) for i, j in combinations(range(8), 2) if not np.allclose(verts[i], -verts[j])]
    elif name == "5-cell":
        # Regular simplex: 5 unit vectors in R^5 summing to zero, mapped onto an orthonormal basis of R^4
        e = np.eye(5) - 1.0 / 5.0
        basis = np.linalg.svd(e)[2][:4]
        verts = e @ basis.T
        verts *= 1.6 / np.linalg.norm(verts[0])
        edges = list(combinations(range(5), 2))
    else:
        raise ValueError(f"unknown polytope {name!r}")
    return verts, edges


def plane_rotation(plane: str, angle: float) -> np.ndarray:
    """4x4 rotation by `angle` radians in the given coordinate plane."""
    i, j = PLANE_AXES[plane]
    rot = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    rot[i, i], rot[i, j], rot[j, i], rot[j, j] = c, -s, s, c
    return rot


def project(points4: np.ndarray, camera_w: float = CAMERA_W) -> np.ndarray:
    """Perspective projection R^4 -> R^3 from a viewpoint on the w axis (the 'shadow' of the object)."""
    w = points4[:, 3]
    factor = 1.0 / (camera_w - w)
    return points4[:, :3] * factor[:, None]


def max_projected_radius(verts: np.ndarray, samples: int = 24) -> float:
    """Largest projected vertex radius over rotations in every plane, for frame-safe scaling."""
    best = 0.0
    for plane in PLANE_AXES:
        for a in np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False):
            pts = project(verts @ plane_rotation(plane, a).T)
            best = max(best, float(np.max(np.linalg.norm(pts, axis=1))))
    return best
