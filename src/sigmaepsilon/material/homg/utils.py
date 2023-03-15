import numpy as np
from numpy import ndarray
from numba import njit, prange

from neumann.linalg import Tensor2x3, CartesianFrame, ReferenceFrame

__cache = True


def _strain_field_3d_bulk(
    centers: ndarray, *, out: ndarray = None, NSTRE: int = 8
) -> ndarray:
    if out is None:
        out = np.zeros((centers.shape[0], 6, NSTRE), dtype=float)
    else:
        NSTRE = out.shape[-1]
    # case 1 - exx_0
    out[:, 0, 0] = 1.0  # exx
    # case 2 - eyy_0
    out[:, 1, 1] = 1.0  # eyy
    # case 3 - exy_0
    out[:, 5, 2] = 1.0  # exy
    # case 4 - kxx
    out[:, 0, 3] = centers[:, 2]  # exx
    # case 5 - kyy
    out[:, 1, 4] = centers[:, 2]  # eyy
    # case 6 - kxy
    out[:, 5, 5] = centers[:, 2]  # exy
    if NSTRE == 8:
        # case 7 - exz_0
        out[:, 4, 6] = 1.0  # exz
        # case 8 - eyz_0
        out[:, 3, 7] = 1.0  # eyz
    return out


def _expand_strain_arrays_3d(loads: ndarray) -> ndarray:
    nE, _, nS2d = loads.shape
    res = np.zeros((nE, 3, 3, nS2d), dtype=float)
    res[:, 0, 0, :] = loads[:, 0, :]
    res[:, 1, 1, :] = loads[:, 1, :]
    res[:, 2, 2, :] = loads[:, 2, :]
    res[:, 1, 2, :] = loads[:, 3, :] / 2
    res[:, 0, 2, :] = loads[:, 4, :] / 2
    res[:, 0, 1, :] = loads[:, 5, :] / 2
    res[:, 2, 1, :] = loads[:, 3, :] / 2
    res[:, 2, 0, :] = loads[:, 4, :] / 2
    res[:, 1, 0, :] = loads[:, 5, :] / 2
    return res


def _collapse_strain_arrays_3d(loads: ndarray) -> ndarray:
    nE, _, _, nS2d = loads.shape
    res = np.zeros((nE, 6, nS2d), dtype=float)
    res[:, 0, :] = loads[:, 0, 0, :]
    res[:, 1, :] = loads[:, 1, 1, :]
    res[:, 2, :] = loads[:, 2, 2, :]
    res[:, 3, :] = loads[:, 1, 2, :] * 2
    res[:, 4, :] = loads[:, 0, 2, :] * 2
    res[:, 5, :] = loads[:, 0, 1, :] * 2
    return res


def _strain_arrays_3d_to_Mindlin(loads: ndarray) -> ndarray:
    nE, _, nS2d = loads.shape
    res = np.zeros((nE, 8, nS2d), dtype=float)
    res[:, 0, :] = loads[:, 0, :]
    res[:, 1, :] = loads[:, 1, :]
    res[:, 2, :] = loads[:, 5, :]
    res[:, 6, :] = loads[:, 4, :]
    res[:, 7, :] = loads[:, 3, :]
    return res


def _strain_arrays_3d_to_Kirchhoff(loads: ndarray) -> ndarray:
    nE, _, nS2d = loads.shape
    res = np.zeros((nE, 6, nS2d), dtype=float)
    res[:, 0, :] = loads[:, 0, :]
    res[:, 1, :] = loads[:, 1, :]
    res[:, 2, :] = loads[:, 5, :]
    return res


def _transform_3d_strain_loads_to_surfaces(
    loads: ndarray, target: ReferenceFrame
) -> ndarray:
    nS2d = loads.shape[-1]
    arrays = _expand_strain_arrays_3d(loads)
    source = CartesianFrame(dim=3)
    for i in range(nS2d):
        tensors = Tensor2x3(arrays[:, :, :, i], frame=source, bulk=True)
        arrays[:, :, :, i] = tensors.show(target)
    arrays = _collapse_strain_arrays_3d(arrays)
    if nS2d == 8:
        arrays = _strain_arrays_3d_to_Mindlin(arrays)
    else:
        arrays = _strain_arrays_3d_to_Kirchhoff(arrays)
    return arrays


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def _postproc_shell_gauss_dynams(
    cell_dynams: ndarray,  # cell dynams
    weights: ndarray,  # gauss weights
    djac: ndarray,  # jacobian determinants
    out: ndarray,  # ABD or ABDS matrix of shape (6, 6) or (8, 8)
) -> ndarray:
    nE = cell_dynams.shape[0]
    nP = weights.shape[0]
    NSTRE = out.shape[0]
    mindlin = NSTRE == 8
    for iE in range(nE):
        for iP in range(nP):
            w = weights[iP]
            dj = djac[iE, iP]
            for j in prange(NSTRE):
                nxx, nyy, nxy, mxx, myy, mxy, vxz, vyz = cell_dynams[iE, iP, :, j]
                out[0, j] += nxx * w * dj
                out[1, j] += nyy * w * dj
                out[2, j] += nxy * w * dj
                out[3, j] += mxx * w * dj
                out[4, j] += myy * w * dj
                out[5, j] += mxy * w * dj
                if mindlin:
                    out[6, j] += vxz * w * dj
                    out[7, j] += vyz * w * dj
    return out


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def _postproc_3d_gauss_stresses(
    cell_stresses: ndarray,  # cell stresses
    cell_gauss_coords: ndarray,  # cell gauss coordinates
    weights: ndarray,  # gauss weights
    djac: ndarray,  # jacobian determinants
    out: ndarray,  # ABD or ABDS matrix of shape (6, 6) or (8, 8)
) -> ndarray:
    nE, nP = cell_gauss_coords.shape[:2]
    NSTRE = out.shape[0]
    mindlin = NSTRE == 8
    for iE in range(nE):
        for iP in range(nP):
            w = weights[iP]
            z = cell_gauss_coords[iE, iP, 2]
            dj = djac[iE, iP]
            for j in range(NSTRE):
                sxx, syy, _, syz, sxz, sxy = cell_stresses[iE, iP, :, j]
                out[0, j] += sxx * w * dj
                out[1, j] += syy * w * dj
                out[2, j] += sxy * w * dj
                out[3, j] += sxx * z * w * dj
                out[4, j] += syy * z * w * dj
                out[5, j] += sxy * z * w * dj
                if mindlin:
                    out[6, j] += sxz * w * dj
                    out[7, j] += syz * w * dj
    return out


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def _calc_avg_hooke_shell(
    hooke: ndarray,  # cell material stiffness matrices
    areas: ndarray,  # gauss weights
    out: ndarray,  # ABD or ABDS matrix of shape (6, 6) or (8, 8)
) -> ndarray:
    nE = areas.shape()[0]
    for iE in range(nE):
        out += areas[iE] * hooke[iE]
    return out


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def _calc_avg_hooke_3d_to_shell(
    hooke: ndarray,  # cell material stiffness matrices
    cell_gauss_coords: ndarray,  # cell gauss coordinates
    weights: ndarray,  # gauss weights
    djac: ndarray,  # jacobian determinants
    out: ndarray,  # ABD or ABDS matrix of shape (6, 6) or (8, 8)
) -> ndarray:
    C_126 = np.zeros((3, 3), dtype=hooke.dtype)
    C_45 = np.zeros((2, 2), dtype=hooke.dtype)
    nE, nP = cell_gauss_coords.shape[:2]
    NSTRE = out.shape[0]
    mindlin = NSTRE == 8
    for iE in range(nE):
        C11 = hooke[iE, 0, 0]
        C12 = hooke[iE, 0, 1]
        C22 = hooke[iE, 1, 1]
        C66 = hooke[iE, 5, 5]
        C55 = hooke[iE, 4, 4]
        C44 = hooke[iE, 3, 3]
        C_126[0, 0] = C11
        C_126[0, 1] = C12
        C_126[1, 0] = C12
        C_126[1, 1] = C22
        C_126[2, 2] = C66
        if mindlin:
            C_45[0, 0] = C55
            C_45[1, 1] = C44
        for iP in range(nP):
            w = weights[iP]
            z = cell_gauss_coords[iE, iP, 2]
            dj = djac[iE, iP]
            out[:3, :3] += C_126 * w * dj
            out[:3, 3:6] += C_126 * z * w * dj
            out[3:6, :3] += C_126 * z * w * dj
            out[3:6, 3:6] += C_126 * z**2 * w * dj
            if mindlin:
                out[6:, 6:] += C_45 * w * dj
    return out
