# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit, prange

from dewloosh.math.array import atleast2d

from dewloosh.mesh.cells import T3 as Triangle
from dewloosh.mesh.tri.triutils import area_tri

from dewloosh.solid.fem.cells import FiniteElement, ABCFiniteElement as ABC
from dewloosh.solid.fem.utils import topo_to_gnum

from ..model.membrane import Membrane


__cache = True


__all__ = ['Turner']


@njit(nogil=True, cache=__cache)
def _strain_displacement_matrix_(ecoords: np.ndarray):
    (x1, x2, x3), (y1, y2, y3) = ecoords[:, 0], ecoords[:, 1]
    x21, x13, x32 = x2 - x1, x1 - x3, x3 - x2
    y12, y31, y23 = y1 - y2, y3 - y1, y2 - y3
    B = np.zeros((3, 6), dtype=ecoords.dtype)
    B[0, :] = np.array([y23, 0, y31, 0, y12, 0], dtype=ecoords.dtype)
    B[1, :] = np.array([0, x32, 0, x13, 0, x21], dtype=ecoords.dtype)
    B[2, :] = np.array([x32, y23, x13, y31, x21, y12], dtype=ecoords.dtype)
    A2 = 2 * area_tri(ecoords)
    return B / A2


@njit(nogil=True, parallel=True, cache=__cache)
def strain_displacement_matrix(ecoords: ndarray, nE: int, nP: int):
    B = np.zeros((nE, nP, 3, 6), dtype=ecoords.dtype)
    for iE in prange(nE):
        B[iE, 0, :, :] = _strain_displacement_matrix_(ecoords[iE])
        for jP in prange(1, nP):
            B[iE, jP, :, :] = B[iE, 0, :, :]
    return B


@njit(nogil=True, cache=__cache)
def _stiffness_matrix_(C: np.ndarray, ecoords: np.ndarray):
    B = _strain_displacement_matrix_(ecoords)
    return B.T @ C @ B


@njit(nogil=True, parallel=True, cache=__cache)
def stiffness_matrix(C: np.ndarray, ecoords: np.ndarray):
    nE = len(C)
    res = np.zeros((nE, 6, 6), dtype=C.dtype)
    for i in prange(nE):
        res[i, :, :] = _stiffness_matrix_(C[i], ecoords[i])
    return res


class Turner(ABC, Membrane, Triangle, FiniteElement):
    """
    Classical Turner triangle.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def strains_at(self, lcoords, *args,  z=None, topo=None, **kwargs):
        if topo is None:
            topo = self.nodes.to_numpy()
        lcoords = atleast2d(lcoords)
        gnum = topo_to_gnum(topo, self.NDOFN)
        dofsol1d = self.pointdata.dofsol.to_numpy().flatten()
        ecoords = self.local_coordinates(topo=topo)
        nP = lcoords.shape[0]
        nE = ecoords.shape[0]
        B = strain_displacement_matrix(ecoords, nE, nP)
        if z is None:
            # return generalized model strains
            return self.model_strains(dofsol1d, gnum, B)
        else:
            # returns material strains
            t = self.thickness()
            model_strains = self.model_strains(dofsol1d, gnum, B)
            return self.material_strains(model_strains, z, t)

    def stiffness_matrix(self, *args, topo=None, **kwargs):
        ecoords = self.local_coordinates(topo=topo)
        C = self.model_stiffness_matrix()
        return stiffness_matrix(C, ecoords)
