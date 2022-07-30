import numpy as np
from numpy import ndarray
from numba import njit, prange
__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def nodal_dcm(dcm: ndarray, N=2):
    nE = dcm.shape[0]
    res = np.zeros((nE, 3 * N, 3 * N), dtype=dcm.dtype)
    for i in prange(N):
        _i = i * 3
        i_ = _i + 3
        res[:, _i:i_, _i:i_] = dcm
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def element_dcm(nodal_dcm: ndarray, nNE: int = 2, nDOF: int=6):
    nE = nodal_dcm.shape[0]
    nEVAB = nNE * nDOF
    res = np.zeros((nE, nEVAB, nEVAB), dtype=nodal_dcm.dtype)
    for iNE in prange(nNE):
        i0, i1 = nDOF*iNE, nDOF * (iNE+1)
        for iE in prange(nE):
            res[iE, i0: i1, i0: i1] = nodal_dcm[iE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def element_transformation_matrices(Q: ndarray, nNE: int=2):
    nE = Q.shape[0]
    nEVAB = nNE * 6
    res = np.zeros((nE, nEVAB, nEVAB), dtype=Q.dtype)
    for iE in prange(nE):
        for j in prange(2*nNE):
            res[iE, 3*j : 3*(j+1), 3*j : 3*(j+1)] = Q[iE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def nodal_transformation_matrices(Q: ndarray):
    nE = Q.shape[0]
    res = np.zeros((nE, 6, 6), dtype=Q.dtype)
    for iE in prange(nE):
        for j in prange(2):
            res[iE, 3*j : 3*(j+1), 3*j : 3*(j+1)] = Q[iE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def transform_element_matrices_out(A: ndarray, Q: ndarray):
    res = np.zeros_like(A)
    for iE in prange(res.shape[0]):
        res[iE] = Q[iE].T @ A[iE] @ Q[iE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def transform_element_vectors_out(A: ndarray, Q: ndarray):
    """
    Transforms element load vectors from local to global.
    """
    res = np.zeros_like(A)
    for iE in prange(res.shape[0]):
        res[iE] = Q[iE].T @ A[iE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def transform_element_vectors_out_multi(A: ndarray, Q: ndarray):
    """
    Transforms multiple element load vectors from 
    local to global.
    """
    nE, nP = A.shape[:2]
    res = np.zeros_like(A)
    for iE in prange(nE):
        for iP in prange(nP):
            res[iE, iP] = Q[iE].T @ A[iE, iP]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def transform_element_vectors_in(dofsol: ndarray, Q: ndarray):
    """
    Transforms element dof solutions from global to local.
    """
    res = np.zeros_like(dofsol)
    for i in prange(res.shape[0]):
        res[i] = Q[i] @ dofsol[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def transform_numint_forces_out(data: ndarray, dcm_G_L: ndarray):
    """
    Transforms internal forces of several elements
    and evaluation points from local frames
    to one global frame.
    
    Parameters
    ----------
    data: (nE, nP, nRHS, nDOFN) numpy.ndarray
        nE : number of elements
        nP : number of sampling points
        nRHS : number of datasets (eg. load cases)
        nDOFN : number of dofs of a node
        
    dcm_G_L : (nE, 3, 3) numpy array
        Array of DCM matrices from global to local.
    
    Returns
    -------
    (nE, nP, nRHS, nDOFN) numpy.ndarray
    """
    nE, nP, nC = data.shape[:3]
    res = np.zeros_like(data)
    for iE in prange(nE):
        for iP in prange(nP):
            for iC in prange(nC):
                res[iE, iP, iC, :] = dcm_G_L[iE].T @ data[iE, iP, iC, :]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def transform_numint_forces_in(data: ndarray, dcm_G_L: ndarray):
    """
    Transforms internal forces of several elements
    and numerical integration points form one global frame
    to several local frames.
    
    Parameters
    ----------
    data: (nE, nP, nRHS, nDOF) numpy.ndarray
        nE : number of elements
        nP : number of sampling points
        nRHS : number of datasets (eg. load cases)
        nDOFN : number of dofs of a node
        
    dcm_G_L : (nE, 3, 3) numpy array
        Array of DCM matrices from global to local.
    
    Returns
    -------
    (nE, nP, nRHS, nDOFN) numpy.ndarray   
    """
    nE, nP, nC = data.shape[:3]
    res = np.zeros_like(data)
    for i in prange(nE):
        for j in prange(nP):
            for k in prange(nC):
                res[i, j, k, :] = dcm_G_L[i] @ data[i, j, k, :]
    return res