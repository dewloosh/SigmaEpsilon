# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit, prange
__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def approx_element_solution_bulk(v: ndarray, A: ndarray):
    """
    Approximates discrete solution over several elements.
    
    Parameters
    ----------
    v : numpy.ndarray
        The discrete values to interpolate as an array of shape 
        (nE, nRHS, nDOF * nNE).
        
    A : numpy.ndarray
        The interpolation matrix of shape (nE, nP, nX, nDOF * nNE).
    
    Returns
    -------
    numpy.ndarray
        An array of shape (nE, nRHS, nP, nX).
        
    """
    nE, nP, nX = A.shape[:3]
    nRHS = v.shape[1]
    res = np.zeros((nE, nRHS, nP, nX), dtype=v.dtype)
    for i in prange(nE):
        for j in prange(nP):
            for k in prange(nRHS):
                res[i, k, j, :] = A[i, j] @ v[i, k]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def calculate_external_forces_bulk(K: ndarray, dofsol: ndarray):
    """
    Returns the external nodal load vectors for several elements.
    
    Parameters
    ----------
    K : numpy.ndarray
        Stiffness matrices of several elements as a 3d numpy 
        array of shape (nE, nEVAB, nEVAB).
    
    dofsol : numpy.ndarray
        Degree of freedom solution for several elements and load cases
        as a 3d numpy array of shape (nE, nRHS, nEVAB).
    
    Note
    ----
    These are the generalized forces that act on the elements.
    
    Returns
    -------
    numpy.ndarray
        3d float array of shape (nE, nRHS, nEVAB), where nE, nRHS and
        nEVAB are the number of elemnts, load cases and element variables.
        
    """
    nE, nRHS, _  = dofsol.shape
    res = np.zeros_like(dofsol)
    for i in prange(nE):
        for j in prange(nRHS):
            res[i, j] = K[i] @ dofsol[i, j]
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


def extrapolate_gauss_data(data, x_in, x_out):
    pass