from typing import Iterable, Union

import numpy as np
from numpy import ndarray
from numba import njit

from polymesh.utils.space import index_of_closest_point

from sigmaepsilon.fem import FemMesh
from sigmaepsilon.fem.ebc import NodeToNode

__cache = True


def _periodic_essential_bc(
    mesh: FemMesh,
    axis: Union[int, Iterable[int]]
) -> NodeToNode:
    imap = _link_opposite_sides(mesh, axis=[0, 1])
    return NodeToNode(imap)


def _link_opposite_sides(
    mesh: FemMesh,
    axis: Union[int, Iterable[int]] = 0
):
    points = mesh.points()
    coords = points.show()
    bounds = points.bounds()
    links = []
    if isinstance(axis, int):
        axis = [axis]
    assert isinstance(axis, Iterable), \
        "'axis' must be an integer or an 1d iterable of integers."
    for iD in axis:
        bmin, bmax = bounds[iD]
        mask_source = coords[:, iD] < (bmin + 1e-12)
        i_source = np.where(mask_source)[0]
        source = coords[i_source]
        mask_target = coords[:, iD] > (bmax - 1e-12)
        i_target = np.where(mask_target)[0]
        target = coords[i_target]
        i = index_of_closest_point(source, target)
        id_source = points[i_source].id[i]
        id_target = points[i_target].id
        links.append(np.stack([id_source, id_target], axis=1))
    return np.vstack(links)


def _strain_field_mindlin_3d_bulk(
    centers: ndarray,
    out: ndarray = None
) -> ndarray:
    if out is None:
        out = np.zeros((centers.shape[0], 6, 8), dtype=float)
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
    # case 7 - exz_0
    out[:, 4, 6] = 1.0  # exz
    # case 8 - eyz_0
    out[:, 3, 7] = 1.0  # eyz
    return out


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def _postproc_3d_gauss_stresses(
    cell_stresses: ndarray,  # cell stresses
    cell_gauss_coords: ndarray,  # cell gauss coordinates
    weights: ndarray,  # gauss weights
    djac: ndarray,  # jacobian determinants
    out: ndarray  # ABD or ABDS matrix of shape (6, 6) or (8, 8)
) -> ndarray:
    nE, nP = cell_gauss_coords.shape[:2]
    nSTRS = out.shape[0]
    mindlin = nSTRS == 8
    for iE in range(nE):
        for iP in range(nP):
            w = weights[iP]
            z = cell_gauss_coords[iE, iP, 2]
            dj = djac[iE, iP]
            for j in range(nSTRS):
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
def _calc_avg_ABDS_3d(
    hooke: ndarray,  # cell material stiffness matrices
    cell_gauss_coords: ndarray,  # cell gauss coordinates
    weights: ndarray,  # gauss weights
    djac: ndarray,  # jacobian determinants
    out: ndarray  # ABD or ABDS matrix of shape (6, 6) or (8, 8)
) -> ndarray:
    C_126 = np.zeros((3, 3), dtype=hooke.dtype)
    C_45 = np.zeros((2, 2), dtype=hooke.dtype)
    nE, nP = cell_gauss_coords.shape[:2]
    nSTRS = out.shape[0]
    mindlin = nSTRS == 8
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
