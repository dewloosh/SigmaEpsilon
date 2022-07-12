# -*- coding: utf-8 -*-
from numba import njit, prange
import numpy as np
from numpy import ndarray

from ...material.utils import HMH_S

from .surface import Surface


__cache = True


__all__ = ['Plate']


_NSTRE_ = 5
_NDOFN_ = 3
_NHOOKE_ = 5


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
                B[iE, iP, 0, 1 + i*_NDOFN_] = -gdshp[i, 0]
                B[iE, iP, 1, 2 + i*_NDOFN_] = -gdshp[i, 1]
                B[iE, iP, 2, 1 + i*_NDOFN_] = -gdshp[i, 1]
                B[iE, iP, 2, 2 + i*_NDOFN_] = -gdshp[i, 0]
                B[iE, iP, 3, 0 + i*_NDOFN_] = gdshp[i, 0]
                B[iE, iP, 3, 1 + i*_NDOFN_] = -1
                B[iE, iP, 4, 0 + i*_NDOFN_] = gdshp[i, 1]
                B[iE, iP, 4, 2 + i*_NDOFN_] = -1
    return B


@njit(nogil=True, parallel=True, cache=__cache)
def material_strains(model_strains: ndarray, z: float, t: float):
    nE, nP = model_strains.shape[:2]
    res = np.zeros((nE, nP, _NHOOKE_), dtype=model_strains.dtype)
    res[:, :, :3] = model_strains[:, :, :3] * z
    for i in prange(nE):
        res[i, :, 3:] = (5/4) * (1 - 4 * (z / t[i])**2) * \
            model_strains[i, :, 3:]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def HMH(estrs: np.ndarray):
    nE, nP = estrs.shape[:2]
    res = np.zeros((nE, nP), dtype=estrs.dtype)
    for iE in prange(nE):
        for jNE in prange(nP):
            res[iE, jNE] = HMH_S(estrs[iE, jNE])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def model_stiffness_iso_homg(C: ndarray, t: ndarray):
    res = np.zeros_like(C)
    for i in prange(res.shape[0]):
        res[i, :3, :3] = C[i, :3, :3] * (t[i]**3 / 12)
        res[i, 3:, 3:] = C[i, 3:, 3:] * (t[i] * 5 / 6)
    return res


class Plate(Surface):
    
    dofs = ('UZ', 'ROTX', 'ROTY')
    
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
        return material_strains(model_strains, z, t)
