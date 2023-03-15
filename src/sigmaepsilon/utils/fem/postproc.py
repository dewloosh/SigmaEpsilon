import numpy as np
from numpy import ndarray, swapaxes as swap
from numba import njit, prange
from scipy.interpolate import interp1d

__cache = True


def approx_element_solution_bulk(v: ndarray, A: ndarray) -> ndarray:
    """
    Approximates discrete solution over several elements.

    Parameters
    ----------
    v : numpy.ndarray
        The discrete values to interpolate as an array of shape
        (nE, nRHS, nDOF * nNE).
    A : numpy.ndarray
        The interpolation matrix of shape (nP, nX, nDOF * nNE) for
        constant metric elements, or (nE, nP, nX, nDOF * nNE) for variable
        matric elements.

    Returns
    -------
    numpy.ndarray
        An array of shape (nE, nRHS, nP, nX).
    """
    if len(A.shape) == 4:
        res = _approx_element_solution_bulk_vm_(v, A)
    else:
        res = _approx_element_solution_bulk_cm_(v, A)
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _approx_element_solution_bulk_cm_(v: ndarray, A: ndarray) -> ndarray:
    """
    Approximates discrete solution over several elements.

    Parameters
    ----------
    v : numpy.ndarray
        The discrete values to interpolate as an array of shape
        (nE, nRHS, nDOF * nNE).
    A : numpy.ndarray
        The interpolation matrix of shape (nP, nX, nDOF * nNE).

    Returns
    -------
    numpy.ndarray
        An array of shape (nE, nRHS, nP, nX).
    """
    nP, nX = A.shape[:2]
    nE, nRHS = v.shape[:2]
    res = np.zeros((nE, nRHS, nP, nX), dtype=v.dtype)
    for i in prange(nE):
        for j in prange(nP):
            for k in prange(nRHS):
                res[i, k, j, :] = A[j] @ v[i, k]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _approx_element_solution_bulk_vm_(v: ndarray, A: ndarray) -> ndarray:
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
def calculate_external_forces_bulk(K: ndarray, dofsol: ndarray) -> ndarray:
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
    nE, nRHS, _ = dofsol.shape
    res = np.zeros_like(dofsol)
    for i in prange(nE):
        for j in prange(nRHS):
            res[i, j] = K[i] @ dofsol[i, j]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def calculate_internal_forces_bulk(
    strains: ndarray, D: ndarray, out: ndarray, imap: ndarray
) -> ndarray:
    """
    strain (nE, nRHS, nP, nX)
    D (nE, nX, nX)
    ---
    (nE, nRHS, nP, nX)
    """
    nE, nRHS, nP, _ = strains.shape
    nI = imap.shape[0]
    for i in prange(nE):
        for j in prange(nRHS):
            for k in prange(nP):
                strs = D[i] @ strains[i, j, k]
                for l in prange(nI):
                    out[i, j, k, imap[l]] = strs[l]
    return out


@njit(nogil=True, parallel=True, cache=__cache)
def explode_kinetic_strains(kstrains: ndarray, nP: int) -> ndarray:
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


def extrapolate_gauss_data_1d(gpos: ndarray, gdata: ndarray) -> ndarray:
    gdata = swap(gdata, 0, 2)  # (nDOFN, nE, nP) --> (nP, nE, nDOFN)
    approx = interp1d(gpos, gdata, fill_value="extrapolate", axis=0)

    def inner(*args, **kwargs):
        res = approx(*args, **kwargs)
        return swap(res, 0, 2)  # (nP, nE, nDOFN) --> (nDOFN, nE, nP)

    return inner


@njit(nogil=True, parallel=True, cache=__cache)
def element_dof_solution_bulk(dofsol1d: ndarray, gnum: ndarray) -> ndarray:
    """
    Returns an array that contains degree of freedom solution for
    every node of every element.

    Parameters
    ----------
    dofsol : numpy.ndarray
        A 2d float array of shape (nX, nRHS), where nX and nRHS are
        the number of total variables and right hand sides.
    gnum : numpy.ndarray
        2d integer array of (nE, nEVAB) of global dof numbering
        for several elements, where nE and nEVAB is the number of
        elements and total variables per element.

    Returns
    -------
    numpy.ndarray
        3d float array of shape (nE, nEVAB, nRHS).
    """
    nRHS = dofsol1d.shape[1]
    nE, nEVAB = gnum.shape
    res = np.zeros((nE, nEVAB, nRHS), dtype=dofsol1d.dtype)
    for i in prange(nE):
        for j in prange(nRHS):
            res[i, :, j] = dofsol1d[gnum[i, :], j]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def element_compliances_bulk(K: ndarray, U: ndarray, gnum: ndarray):
    nE, nNE = gnum.shape
    res = np.zeros(nE, dtype=K.dtype)
    for iE in prange(nE):
        for i in range(nNE):
            for j in range(nNE):
                res[iE] += K[iE, i, j] * U[gnum[iE, i]] * U[gnum[iE, j]]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def element_compliances_flat(
    K: ndarray,
    U: ndarray,
    krows: ndarray,
    kcols: ndarray,
    kranges: ndarray,
) -> ndarray:
    nE = len(kranges) - 1
    res = np.zeros(nE, dtype=K.dtype)
    for i in prange(nE):
        _r, r_ = kranges[i], kranges[i + 1]
        res[i] = np.sum(U[krows[_r:r_]] * K[_r:r_] * U[kcols[_r:r_]])
    return res
