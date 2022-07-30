# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit, prange

from ...math import squeeze
from ...math.linalg.solve import npsolve


@squeeze(True)
def linsolve(LHS: ndarray, RHS: ndarray, *args, **kwarg):
    if len(LHS.shape) > 2:
        return linsolve_Mindlin(LHS, RHS)
    else:
        return linsolve_Kirchhoff(LHS, RHS)


@njit(nogil=True, parallel=True, cache=True)
def linsolve_Kirchhoff(A: ndarray, B: ndarray):
    nLHS, nMN = A.shape
    nRHS = B.shape[0]
    res = np.zeros((nRHS, nLHS, nMN))
    for i in prange(nRHS):
        for j in prange(nLHS):
            for k in prange(nMN):
                res[i, j, k] = B[i, k] / A[j, k]
    return res


@njit(nogil=True, parallel=True, cache=True)
def linsolve_Mindlin(A: ndarray, B: ndarray):
    nLHS, nMN = A.shape[:2]
    nRHS = B.shape[0]
    res = np.zeros((nRHS, nLHS, nMN, 3))
    for i in prange(nRHS):
        for j in prange(nLHS):
            for k in prange(nMN):
                #res[i, j, k] = inv3x3(A[j, k]) @ B[i, k]
                res[i, j, k] = npsolve(A[j, k], B[i, k]) 
    return res