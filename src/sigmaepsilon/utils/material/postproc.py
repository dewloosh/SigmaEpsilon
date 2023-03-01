import numpy as np
from numba import njit, prange
from collections import Iterable

from neumann import ascont, atleast1d, atleast2d, atleast3d, atleast4d

from .surface import layers_of_points, points_of_layers
from .mindlin import (
    z_to_shear_factors,
    shear_correction_data,
    material_stiffness_matrices,
)

__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def _pproc_Mindlin_3D(
    ABDS: np.ndarray,
    sfx: np.ndarray,
    sfy: np.ndarray,
    C_126: np.ndarray,
    C_45: np.ndarray,
    bounds: np.ndarray,
    points: np.ndarray,
    e_126_n: np.ndarray,
    c_126: np.ndarray,
    e_45: np.ndarray,
):
    nRHS, nP, _ = e_126_n.shape
    layerinds = layers_of_points(points, bounds)

    # results
    s_126_n = np.zeros((nRHS, nP, 3), dtype=ABDS.dtype)
    s_126_m = np.zeros((nRHS, nP, 3), dtype=ABDS.dtype)
    s_45 = np.zeros((nRHS, nP, 2), dtype=ABDS.dtype)
    e_126_m = np.zeros((nRHS, nP, 3), dtype=ABDS.dtype)
    e_45_new = np.zeros((nRHS, nP, 2), dtype=ABDS.dtype)

    for iP in prange(nP):
        zP = points[iP, 2]
        lP = layerinds[iP]
        sfxz, sfyz = z_to_shear_factors(zP, sfx[lP, :], sfy[lP, :])
        for iRHS in prange(nRHS):
            e_126_m[iRHS, iP] = zP * c_126[iRHS, iP]
            s_126_n[iRHS, iP] = C_126[lP] @ e_126_n[iRHS, iP]
            s_126_m[iRHS, iP] = C_126[lP] @ e_126_m[iRHS, iP]
            s_45[iRHS, iP, 0] = sfxz * ABDS[6, 6] * e_45[iRHS, iP, 0]
            s_45[iRHS, iP, 1] = sfyz * ABDS[7, 7] * e_45[iRHS, iP, 1]
            e_45_new[iRHS, iP, 0] = s_45[iRHS, iP, 0] / C_45[lP, 0, 0]
            e_45_new[iRHS, iP, 1] = s_45[iRHS, iP, 1] / C_45[lP, 1, 1]

    return s_126_n, s_126_m, s_45, e_126_n, e_126_m, e_45_new


@njit(nogil=True, parallel=True, cache=__cache)
def _pproc_Mindlin_3D_s(
    ABDS: np.ndarray,
    sfx: np.ndarray,
    sfy: np.ndarray,
    C_126: np.ndarray,
    C_45: np.ndarray,
    bounds: np.ndarray,
    ppl: int,
    e_126_n: np.ndarray,
    c_126: np.ndarray,
    e_45: np.ndarray,
):
    nRHS, nP, _ = e_126_n.shape
    z = points_of_layers(bounds, ppl)
    nL = z.shape[0]

    # results
    s_126_n = np.zeros((nRHS, nP, nL, ppl, 3), dtype=ABDS.dtype)
    s_126_m = np.zeros((nRHS, nP, nL, ppl, 3), dtype=ABDS.dtype)
    s_45 = np.zeros((nRHS, nP, nL, ppl, 2), dtype=ABDS.dtype)
    e_126_m = np.zeros((nRHS, nP, nL, ppl, 3), dtype=ABDS.dtype)
    e_45_new = np.zeros((nRHS, nP, nL, ppl, 2), dtype=ABDS.dtype)

    for iL in prange(nL):
        for ippl in prange(ppl):
            zP = z[iL, ippl]
            sfxz, sfyz = z_to_shear_factors(zP, sfx[iL, :], sfy[iL, :])
            for iP in prange(nP):
                for iRHS in prange(nRHS):
                    e_126_m[iRHS, iP, iL, ippl] = zP * c_126[iRHS, iP]
                    s_126_n[iRHS, iP, iL, ippl] = C_126[iL] @ e_126_n[iRHS, iP]
                    s_126_m[iRHS, iP, iL, ippl] = C_126[iL] @ e_126_m[iRHS, iP]
                    s_45[iRHS, iP, iL, ippl, 0] = sfxz * ABDS[6, 6] * e_45[iRHS, iP, 0]
                    s_45[iRHS, iP, iL, ippl, 1] = sfyz * ABDS[7, 7] * e_45[iRHS, iP, 1]
                    e_45_new[iRHS, iP, iL, ippl, 0] = (
                        s_45[iRHS, iP, iL, ippl, 0] / C_45[iL, 0, 0]
                    )
                    e_45_new[iRHS, iP, iL, ippl, 1] = (
                        s_45[iRHS, iP, iL, ippl, 1] / C_45[iL, 1, 1]
                    )

    return s_126_n, s_126_m, s_45, e_126_n, e_126_m, e_45_new


@njit(nogil=True, parallel=True, cache=__cache)
def _pproc_Mindlin_3D_rgrid(
    ABDS: np.ndarray,
    coords2d: np.ndarray,
    topo2d: np.ndarray,
    bounds: np.ndarray,
    cell_per_layer: int,
    eshape: tuple,
    sfx: np.ndarray,
    sfy: np.ndarray,
    C_126: np.ndarray,
    C_45: np.ndarray,
    e_126_n: np.ndarray,
    c_126: np.ndarray,
    e_45: np.ndarray,
):
    nNEx, nNEy, nNEz = eshape
    nNE3d = nNEx * nNEy * nNEz
    nRHS = e_126_n.shape[0]
    nL = bounds.shape[0]
    nE2d, nNE2d = topo2d.shape
    nE3d = nE2d * nL * cell_per_layer
    nN3d = nE3d * nNE3d

    ftype = coords2d.dtype
    itype = topo2d.dtype
    s_126_n = np.zeros((nRHS, nN3d, 3), dtype=ftype)
    s_126_m = np.zeros((nRHS, nN3d, 3), dtype=ftype)
    s_45 = np.zeros((nRHS, nN3d, 2), dtype=ftype)
    e_126_m = np.zeros((nRHS, nN3d, 3), dtype=ftype)
    e_45_new = np.zeros((nRHS, nN3d, 2), dtype=ftype)
    coords3d = np.zeros((nN3d, 3), dtype=ftype)
    topo3d = np.zeros((nE3d, nNE3d), dtype=itype)

    # precalculate shear factors
    sfxz = np.zeros((nL, cell_per_layer, nNEz), dtype=ftype)
    sfyz = np.zeros((nL, cell_per_layer, nNEz), dtype=ftype)
    for iL in prange(nL):
        dZ = (bounds[iL, 1] - bounds[iL, 0]) / cell_per_layer
        ddZ = dZ / (nNEz - 1)
        for iCL in prange(cell_per_layer):
            for iNEz in prange(nNEz):
                zP = dZ * iCL + ddZ * iNEz
                sfxz[iL, iCL, iNEz], sfyz[iL, iCL, iNEz] = z_to_shear_factors(
                    zP, sfx[iL, :], sfy[iL, :]
                )

    for iL in prange(nL):
        dZ = (bounds[iL, 1] - bounds[iL, 0]) / cell_per_layer
        ddZ = dZ / (nNEz - 1)
        for iCL in prange(cell_per_layer):
            for iE2d in prange(nE2d):
                iE3d = nE2d * (cell_per_layer * iL + iCL) + iE2d
                for iNEx in prange(nNEx):
                    for iNEy in prange(nNEy):
                        iNE2d = iNEx * nNEy + iNEy
                        iN2d = topo2d[iE2d, iNE2d]
                        for iNEz in prange(nNEz):
                            iNE3d = iNEx * iNEy * nNEz + iNEz
                            iN3d = iE3d * nNE3d + iNE3d
                            topo3d[iE3d, iNE3d] = iN3d
                            coords3d[iN3d, 0] = coords2d[iN2d, 0]
                            coords3d[iN3d, 1] = coords2d[iN2d, 1]
                            coords3d[iN3d, 2] = dZ * iCL + ddZ * iNEz
                            for iRHS in prange(nRHS):
                                e_126_m[iRHS, iN3d] = (
                                    coords3d[iN3d, 2] * c_126[iRHS, iN2d]
                                )
                                s_126_n[iRHS, iN3d] = C_126[iL] @ e_126_n[iRHS, iN2d]
                                s_126_m[iRHS, iN3d] = C_126[iL] @ e_126_m[iRHS, iN3d]
                                s_45[iRHS, iN3d, 0] = (
                                    sfxz[iL, iCL, iNEz]
                                    * ABDS[6, 6]
                                    * e_45[iRHS, iN2d, 0]
                                )
                                s_45[iRHS, iN3d, 1] = (
                                    sfyz[iL, iCL, iNEz]
                                    * ABDS[7, 7]
                                    * e_45[iRHS, iN2d, 1]
                                )
                                e_45_new[iRHS, iN3d, 0] = (
                                    s_45[iRHS, iN3d, 0] / C_45[iL, 0, 0]
                                )
                                e_45_new[iRHS, iN3d, 1] = (
                                    s_45[iRHS, iN3d, 1] / C_45[iL, 1, 1]
                                )

    return s_126_n, s_126_m, s_45, e_126_n, e_126_m, e_45_new


def pproc_Mindlin_3D(
    ABDS: np.ndarray,
    C_126: Iterable,
    C_45: Iterable,
    bounds: Iterable,
    points: np.ndarray,
    strains2d: np.ndarray,
    *args,
    angles: Iterable = None,
    separate=True,
    shear_factors: Iterable = None,
    **kwargs
):
    # formatting
    ABDS = atleast3d(ABDS)
    strains2d = atleast4d(strains2d)
    if isinstance(C_126, np.ndarray):
        C_126 = atleast4d(C_126)
    else:
        C_126 = [atleast3d(C_126[i]) for i in range(len(C_126))]
    if isinstance(C_45, np.ndarray):
        C_45 = atleast4d(C_45)
    else:
        C_45 = [atleast3d(C_45[i]) for i in range(len(C_45))]
    if isinstance(bounds, np.ndarray):
        bounds = atleast3d(bounds)
    else:
        bounds = [atleast2d(bounds[i]) for i in range(len(bounds))]

    # transform material stiffness matrices to global
    if angles is not None:
        C_126_g = np.zeros_like(C_126)
        C_45_g = np.zeros_like(C_45)
        if isinstance(angles, np.ndarray):
            angles = atleast2d(angles)
        else:
            angles = [atleast1d(angles[i]) for i in range(len(angles))]
        for iLHS in range(len(angles)):
            C_126_g[iLHS], C_45_g[iLHS] = material_stiffness_matrices(
                C_126[iLHS], C_45[iLHS], angles[iLHS]
            )
        C_126 = C_126_g
        C_45 = C_45_g
        del C_126_g
        del C_45_g

    # calculate shear factors
    if shear_factors is None:
        shear_factors = []
        for i in range(len(C_126)):
            _, sf = shear_correction_data(ABDS[i], C_126[i], C_45[i], bounds[i])
            shear_factors.append(sf)
    else:
        if isinstance(shear_factors, np.ndarray):
            shear_factors = atleast4d(shear_factors)
        else:
            shear_factors = [
                atleast3d(shear_factors[i]) for i in range(len(shear_factors))
            ]

    nLHS, nRHS, nP, _ = strains2d.shape
    e_126_n = ascont(strains2d[:, :, :, :3])
    s_126_n = np.zeros((nLHS, nRHS, nP, 3), dtype=ABDS.dtype)
    s_126_m = np.zeros((nLHS, nRHS, nP, 3), dtype=ABDS.dtype)
    s_45 = np.zeros((nLHS, nRHS, nP, 2), dtype=ABDS.dtype)
    e_126_m = np.zeros((nLHS, nRHS, nP, 3), dtype=ABDS.dtype)
    e_45 = np.zeros((nLHS, nRHS, nP, 2), dtype=ABDS.dtype)

    for i in range(nLHS):
        sfx_i = ascont(shear_factors[i][:, 0, :])
        sfy_i = ascont(shear_factors[i][:, 1, :])
        e_126_n_i = ascont(strains2d[i, :, :, :3])
        c_126_i = ascont(strains2d[i, :, :, 3:6])
        e_45_i = ascont(strains2d[i, :, :, 6:8])
        (
            s_126_n[i],
            s_126_m[i],
            s_45[i],
            e_126_n[i],
            e_126_m[i],
            e_45[i],
        ) = _pproc_Mindlin_3D(
            ABDS[i],
            sfx_i,
            sfy_i,
            C_126[i],
            C_45[i],
            bounds[i],
            points,
            e_126_n_i,
            c_126_i,
            e_45_i,
        )

    if separate:
        res3d = s_126_n, s_126_m, s_45, e_126_n, e_126_m, e_45
    else:
        res3d = s_126_n + s_126_m, s_45, e_126_n + e_126_m, e_45

    return res3d
