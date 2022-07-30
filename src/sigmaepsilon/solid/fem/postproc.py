# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit, prange
__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def approx_element_solution_bulk(dofsol: ndarray, A: ndarray):
    """
    dofsol (nE, nRHS, nDOF * nNE)
    A (nE, nP, nX, nDOF * nNE)
    ---
    (nE, nRHS, nP, nX)
    """
    nE, nP, nX = A.shape[:3]
    nRHS = dofsol.shape[1]
    res = np.zeros((nE, nRHS, nP, nX), dtype=dofsol.dtype)
    for i in prange(nE):
        for j in prange(nP):
            for k in prange(nRHS):
                res[i, k, j, :] = A[i, j] @ dofsol[i, k]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def calculate_internal_forces_bulk(strains: ndarray, D: ndarray):
    """
    strain (nE, nRHS, nP, nX)
    D (nE, nX, nX)
    ---
    (nE, nRHS, nP, nX)
    """
    nE, nRHS, nP, _ = strains.shape
    res = np.zeros_like(strains)
    for i in prange(nE):
        for j in prange(nRHS):
            for k in prange(nP):
                res[i, j, k] = D[i] @ strains[i, j, k]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def explode_kinetic_strains(kstrains: ndarray, nP: int):
    """
    ---
    (nE, nRHS, nP, nSTRE)
    """
    nE, nSTRE, nRHS = kstrains.shape
    res = np.zeros((nE, nRHS, nP, nSTRE), dtype=kstrains.dtype)
    for i in prange(nE):
        for j in prange(nRHS):
            for k in prange(nP):
                for m in prange(nSTRE):
                    res[i, j, k, m] = kstrains[i, m, j]
    return res