# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit, prange

from neumann import squeeze
from neumann.linalg.solve import npsolve
from neumann.linalg import inv2x2, inv3x3


@squeeze(True)
def linsolve(LHS: ndarray, RHS: ndarray, *args, **kwarg):
    if len(LHS.shape) > 2:
        return linsolve2d(LHS, RHS)
    else:
        pass


@njit(nogil=True, parallel=True, cache=True)
def linsolve_Bernoulli(A: ndarray, B: ndarray):
    """
    Calculates unknowns for Bernoulli Beams.
    """
    nLHS, N = A.shape
    nRHS = B.shape[0]
    res = np.zeros((nRHS, nLHS, N))
    for i in prange(nRHS):
        for j in prange(nLHS):
            for n in prange(N):
                res[i, j, n] = B[i, n] / A[j, n]
    return res


@njit(nogil=True, parallel=True, cache=True)
def linsolve_Timoshenko(A: ndarray, B: ndarray):
    """
    Calculates unknowns for Bernoulli Beams.
    """
    nLHS, nMN = A.shape[:2]
    nRHS = B.shape[0]
    res = np.zeros((nRHS, nLHS, nMN, 2))    
    for i in prange(nRHS):
        for j in prange(nLHS):
            for k in prange(nMN):
                res[i, j, k] = inv2x2(A[j, k]) @ B[i, k]
    return res


@njit(nogil=True, parallel=True, cache=True)
def linsolve2d(A: ndarray, B: ndarray):
    nLHS, nMN = A.shape[:2]
    nV = A.shape[-1]
    nRHS = B.shape[0]
    res = np.zeros((nRHS, nLHS, nMN, nV))
    if nV == 3:
        for i in prange(nRHS):
            for j in prange(nLHS):
                for k in prange(nMN):
                    res[i, j, k] = npsolve(A[j, k], B[i, k])
                    #res[i, j, k] = inv3x3(A[j, k]) @ B[i, k] 
    else:
        for i in prange(nRHS):
            for j in prange(nLHS):
                for k in prange(nMN):
                    res[i, j, k] = npsolve(A[j, k], B[i, k])
                    #res[i, j, k] = inv2x2(A[j, k]) @ B[i, k]
    return res