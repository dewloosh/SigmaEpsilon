from numba import njit, prange
import numpy as np
from numpy import ndarray

from ...utils.material.hmh import HMH_M
from .surface import Surface


__cache = True


__all__ = ["Membrane", "DrillingMembrane"]


_NSTRE_ = 3
_NDOFN_ = 2
_NHOOKE_ = 3


@njit(nogil=True, parallel=True, cache=__cache)
def strain_displacement_matrix(dshp: ndarray, jac: ndarray):
    nE = jac.shape[0]
    nP, nN = dshp.shape[:2]
    nTOTV = nN * _NDOFN_
    B = np.zeros((nE, nP, _NSTRE_, nTOTV), dtype=dshp.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            gdshp = dshp[iP] @ np.linalg.inv(jac[iE, iP])
            for i in prange(nN):
                B[iE, iP, 0, 0 + i * _NDOFN_] = gdshp[i, 0]
                B[iE, iP, 1, 1 + i * _NDOFN_] = gdshp[i, 1]
                B[iE, iP, 2, 0 + i * _NDOFN_] = gdshp[i, 1]
                B[iE, iP, 2, 1 + i * _NDOFN_] = gdshp[i, 0]
    return B


@njit(nogil=True, parallel=True, cache=__cache)
def material_strains(strns: ndarray, z: float):
    nE, nP = strns.shape[:2]
    res = np.zeros((nE, nP, _NHOOKE_), dtype=strns.dtype)
    res[:, :, :3] = strns[:, :, :3]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def HMH(estrs: np.ndarray):
    nE, nP = estrs.shape[:2]
    res = np.zeros((nE, nP), dtype=estrs.dtype)
    for iE in prange(nE):
        for jNE in prange(nP):
            res[iE, jNE] = HMH_M(estrs[iE, jNE])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def model_stiffness_iso_homg(C: ndarray, t: ndarray):
    res = np.zeros_like(C)
    for i in prange(res.shape[0]):
        res[i, :, :] = C[i, :, :] * t[i]
    return res


class Membrane(Surface):
    dofs = ("UX", "UY")
    strn = ("exx", "eyy", "exy")

    NDOFN = _NDOFN_
    NSTRE = _NSTRE_

    @classmethod
    def model_stiffness_matrix_iso_homg(cls, C, t, *args, **kwargs):
        return model_stiffness_iso_homg(C, t)

    @classmethod
    def strain_displacement_matrix(cls, *args, dshp=None, jac=None, **kwargs):
        return strain_displacement_matrix(dshp, jac)

    @classmethod
    def HMH(cls, data, *args, **kwargs):
        return HMH(data)

    @classmethod
    def material_strains(cls, model_strains, z, t, *args, **kwargs):
        return material_strains(model_strains, z)


class DrillingMembrane(Membrane):
    dofs = ("UX", "UY", "ROTZ")

    NDOFN = 3
    NSTRE = _NSTRE_
