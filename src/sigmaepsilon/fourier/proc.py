import numpy as np
from numpy import ndarray
from numba import njit, prange

from neumann.linalg import inv2x2, inv3x3


@njit(nogil=True, parallel=True, cache=True)
def linsolve_Bernoulli(A: ndarray, B: ndarray):
    """
    Calculates unknowns for Bernoulli beams.
    """
    nLHS, N = A.shape
    nRHS = B.shape[0]
    res = np.zeros((nLHS, nRHS, N))
    for i in prange(nRHS):
        for j in prange(nLHS):
            for n in prange(N):
                res[j, i, n] = B[i, n] / A[j, n]
    return res


@njit(nogil=True, parallel=True, cache=True)
def linsolve_Timoshenko(A: ndarray, B: ndarray):
    """
    Calculates unknowns for Timoshenko beams.
    """
    nLHS, nMN = A.shape[:2]
    nRHS = B.shape[0]
    res = np.zeros((nLHS, nRHS, nMN, 2))
    for i in prange(nRHS):
        for j in prange(nLHS):
            for k in prange(nMN):
                res[j, i, k] = inv2x2(A[j, k]) @ B[i, k]
    return res


@njit(nogil=True, parallel=True, cache=True)
def linsolve_Mindlin(A: ndarray, B: ndarray):
    """
    Calculates unknowns for Mindlin plates.
    """
    nLHS, nMN = A.shape[:2]
    nRHS = B.shape[0]
    res = np.zeros((nLHS, nRHS, nMN, 3))
    for i in prange(nRHS):
        for j in prange(nLHS):
            for k in prange(nMN):
                res[j, i, k] = inv3x3(A[j, k]) @ B[i, k]
    return res


@njit(nogil=True, parallel=True, cache=True)
def linsolve_Kirchhoff(A: ndarray, B: ndarray):
    """
    Calculates unknowns for Kirchhoff plates.
    """
    nLHS, nMN = A.shape[:2]
    nRHS = B.shape[0]
    res = np.zeros((nLHS, nRHS, nMN))
    for i in prange(nRHS):
        for j in prange(nLHS):
            for k in prange(nMN):
                res[j, i, k] = B[i, k] / A[j, k]
    return res
