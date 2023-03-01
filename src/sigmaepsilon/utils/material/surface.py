import numpy as np
from numpy import ndarray
from numpy.linalg import inv
from numba import njit, prange

from neumann.linalg import linspace1d
from neumann import clip1d

__cache = True


@njit(nogil=True, cache=__cache)
def layers_of_points(points: np.ndarray, bounds: np.ndarray):
    nL = bounds.shape[0]
    bins = np.zeros((nL + 1,), dtype=points.dtype)
    bins[0] = bounds[0, 0]
    bins[1:] = bounds[:, 1]
    return clip1d(np.digitize(points[:, 2], bins) - 1, 0, nL - 1)


@njit(
    ["f8[:, :](f8[:, :], i8)", "f4[:, :](f4[:, :], i8)"],
    nogil=True,
    parallel=True,
    cache=__cache,
)
def points_of_layers(bounds: np.ndarray, nppl=3):
    nL = bounds.shape[0]
    res = np.zeros((nL, nppl), dtype=bounds.dtype)
    for iL in prange(nL):
        res[iL] = linspace1d(bounds[iL, 0], bounds[iL, 1], nppl)
    return res


@njit(nogil=True, cache=__cache)
def glob_to_loc_layer(point: np.ndarray, bounds: np.ndarray):
    return (point[2] - bounds[0]) / (bounds[1] - bounds[0])


@njit(nogil=True, parallel=True, cache=__cache)
def iso_mindlin_plate_bulk(C: ndarray, t: ndarray, k: float = 5 / 6):
    res = np.zeros((C.shape[0], 5, 5))
    for i in prange(res.shape[0]):
        res[i, :3, :3] = C[i, :3, :3] * (t[i] ** 3 / 12)
        res[i, 3:, 3:] = C[i, 3:, 3:] * t[i]
    res[:, 3:, 3:] *= k
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def iso_membrane_bulk(C: ndarray, t: ndarray):
    res = np.zeros((C.shape[0], 3, 3))
    for i in prange(res.shape[0]):
        res[i, :, :] = C[i, :, :] * t[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def iso_mindlin_shell_bulk(C: ndarray, t: ndarray, k: float = 5 / 6):
    res = np.zeros((C.shape[0], 8, 8))
    res[:, :3, :3] = iso_membrane_bulk(C, t)
    res[:, 3:, 3:] = iso_mindlin_plate_bulk(C, t, k)
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def rotation_matrices_126(angles: np.ndarray):
    """
    Returns transformation matrmatrixices T_126 for each angle.
    Angles are expected in radians.
    """
    nL = len(angles)
    T_126 = np.zeros((nL, 3, 3), dtype=angles.dtype)
    for iL in prange(nL):
        a = angles[iL] * np.pi / 180
        T_126[iL, 0, 0] = np.cos(a) ** 2
        T_126[iL, 0, 1] = np.sin(a) ** 2
        T_126[iL, 0, 2] = -np.sin(2 * a)
        T_126[iL, 1, 0] = T_126[iL, 0, 1]
        T_126[iL, 1, 1] = T_126[iL, 0, 0]
        T_126[iL, 1, 2] = -T_126[iL, 0, 2]
        T_126[iL, 2, 0] = np.cos(a) * np.sin(a)
        T_126[iL, 2, 1] = -T_126[iL, 2, 0]
        T_126[iL, 2, 2] = np.cos(a) ** 2 - np.sin(a) ** 2
    return T_126


@njit(nogil=True, parallel=True, cache=__cache)
def rotation_matrices_45(angles: np.ndarray):
    """
    Returns transformation matrix T_45 for each angle.
    Angles are expected in radians.
    """
    nL = len(angles)
    T_45 = np.zeros((nL, 2, 2), dtype=angles.dtype)
    for iL in prange(nL):
        a = angles[iL] * np.pi / 180
        T_45[iL, 0, 0] = np.cos(a)
        T_45[iL, 0, 1] = -np.sin(a)
        T_45[iL, 1, 1] = T_45[iL, 0, 0]
        T_45[iL, 1, 0] = -T_45[iL, 0, 1]
    return T_45


@njit(nogil=True, parallel=True, cache=__cache)
def material_stiffness_matrices_126(C_126: np.ndarray, angles: np.ndarray):
    """
    Returns the components of the material stiffness matrix C_126
    in the global system.
    """
    nL = len(C_126)
    C_126_g = np.zeros_like(C_126)
    R_126 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=C_126_g.dtype)
    R_126_inv = inv(R_126)
    T_126 = rotation_matrices_126(angles)
    for iL in prange(nL):
        C_126_g[iL] = T_126[iL] @ C_126[iL] @ R_126_inv @ T_126[iL].T @ R_126
    return C_126_g


@njit(nogil=True, parallel=True, cache=__cache)
def material_stiffness_matrices_45(C_45: np.ndarray, angles: np.ndarray):
    """
    Returns the components of the material stiffness matrix C_45
    in the global system.
    """
    nL = len(C_45)
    C_45_g = np.zeros_like(C_45)
    R_45 = np.array([[2, 0], [0, 2]], dtype=C_45_g.dtype)
    R_45_inv = inv(R_45)
    T_45 = rotation_matrices_45(angles)
    for iL in prange(nL):
        C_45_g[iL] = T_45[iL] @ C_45[iL] @ R_45_inv @ T_45[iL].T @ R_45
    return C_45_g
