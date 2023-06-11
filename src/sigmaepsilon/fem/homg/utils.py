from typing import Union

import numpy as np
from numpy import ndarray
from numba import njit

from neumann.linalg import ReferenceFrame
from sigmaepsilon.material import SmallStrainTensor

__cache = True


def _tr_strains_to_local_frames(
    strains: ndarray,
    global_frame: Union[ndarray, ReferenceFrame],
    local_frames: ndarray,
) -> ndarray:
    if isinstance(global_frame, ndarray):
        source = ReferenceFrame(global_frame)
    elif isinstance(global_frame, ReferenceFrame):
        source = global_frame
    target = ReferenceFrame(local_frames)
    tensor = SmallStrainTensor(strains, frame=source, tensorial=False)
    return tensor.contracted_components(target=target, engineering=True)


def _strain_field_3d_bulk(
    centers: ndarray, *, out: ndarray = None, NSTRE: int = 8
) -> ndarray:
    if out is None:
        out = np.zeros((centers.shape[0], NSTRE, 6), dtype=float)
    else:
        NSTRE = out.shape[-2]
    # case 1 - exx_0
    out[:, 0, 0] = 1.0  # exx
    # case 2 - eyy_0
    out[:, 1, 1] = 1.0  # eyy
    # case 3 - exy_0
    out[:, 2, 5] = 1.0  # exy
    # case 4 - kxx
    out[:, 3, 0] = centers[:, 2]  # exx
    # case 5 - kyy
    out[:, 4, 1] = centers[:, 2]  # eyy
    # case 6 - kxy
    out[:, 5, 5] = centers[:, 2]  # exy
    if NSTRE == 8:
        # case 7 - exz_0
        out[:, 6, 4] = 1.0  # exz
        # case 8 - eyz_0
        out[:, 7, 3] = 1.0  # eyz
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
