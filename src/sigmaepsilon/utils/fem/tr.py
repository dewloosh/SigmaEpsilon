import numpy as np
from numpy import ndarray
from numba import njit, prange

from neumann import repeat_diagonal_2d, ascont

__cache = True


@njit(nogil=True, cache=__cache)
def nodal_dcm(dcm: ndarray, N: int = 2) -> ndarray:
    """
    Assembles the nodal direction cosine matrix of a single node.

    Parameters
    ----------
    dcm : numpy.ndarray
        A 3x3 direction cosine matrix.
    N : int, Optional
        The number of triplets that make up a nodal vector.
        Default is 2, which means 6 dofs per node.

    Returns
    -------
    numpy.ndarray
        A 2d numpy float array.

    Note
    ----
    The template 'dcm' should be the matrix you would use to transform a
    position vector in 3d Euclidean space.
    """
    return repeat_diagonal_2d(dcm, N)


@njit(nogil=True, parallel=True, cache=__cache)
def nodal_dcm_bulk(dcm: ndarray, N: int = 2) -> ndarray:
    """
    Assembles the nodal direction cosine matrices of several cells.

    Parameters
    ----------
    dcm : numpy.ndarray
        A 3x3 direction cosine matrix for each element (4d).
    N : int, Optional
        The number of triplets that make up a nodal vector.
        Default is 2, which means 6 dofs per node.

    Returns
    -------
    numpy.ndarray
        A 3d numpy float array.

    Note
    ----
    Typically, 'dcm' should be the matrix you would use to transform a
    position vector in 3d Euclidean space.
    """
    nE = dcm.shape[0]
    res = np.zeros((nE, 3 * N, 3 * N), dtype=dcm.dtype)
    for i in prange(N):
        _i = i * 3
        i_ = _i + 3
        res[:, _i:i_, _i:i_] = dcm
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def element_dcm(nodal_dcm: ndarray, nNE: int = 2, nDOF: int = 6):
    nEVAB = nNE * nDOF
    res = np.zeros((nEVAB, nEVAB), dtype=nodal_dcm.dtype)
    for iNE in prange(nNE):
        i0, i1 = nDOF * iNE, nDOF * (iNE + 1)
        res[i0:i1, i0:i1] = nodal_dcm
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def element_dcm_bulk(nodal_dcm: ndarray, nNE: int = 2, nDOF: int = 6):
    nE = nodal_dcm.shape[0]
    nEVAB = nNE * nDOF
    res = np.zeros((nE, nEVAB, nEVAB), dtype=nodal_dcm.dtype)
    for iNE in prange(nNE):
        i0, i1 = nDOF * iNE, nDOF * (iNE + 1)
        for iE in prange(nE):
            res[iE, i0:i1, i0:i1] = nodal_dcm[iE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def tr_element_vectors_bulk_multi(
    values: ndarray, dcm: ndarray, invert: bool = False
) -> ndarray:
    """
    Transforms element vectors for multiple cases.

    Parameters
    ----------
    values : numpy.ndarray
        The values to transform as an array of shape (nE, nRHS, nX).
    dcm : numpy.ndarray
        The direct cosine matrix of the transformation as an
        array of shape (nE, nX).
    invert : bool, Optional
        If True, the DCM matrices are transposed before transformation.
        This makes this function usable in both directions.
        Default is False.

    Returns
    -------
    numpy.ndarray
        An array with the same shape as 'values'.
    """
    res = np.zeros_like(values)
    if not invert:
        for iE in prange(res.shape[0]):
            for jRHS in prange(res.shape[1]):
                res[iE, jRHS] = dcm[iE] @ values[iE, jRHS]
    else:
        for iE in prange(res.shape[0]):
            for jRHS in prange(res.shape[1]):
                res[iE, jRHS] = dcm[iE].T @ values[iE, jRHS]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def tr_element_matrices_bulk(K: ndarray, dcm: ndarray, invert: bool = False):
    """
    Transforms element stiffness matrices.
    """
    res = np.zeros_like(K)
    if not invert:
        for iE in prange(res.shape[0]):
            res[iE] = dcm[iE] @ K[iE] @ dcm[iE].T
    else:
        for iE in prange(res.shape[0]):
            res[iE] = dcm[iE].T @ K[iE] @ dcm[iE]
    return res


def tr_nodal_loads_bulk(nodal_loads: ndarray, dcm: ndarray) -> ndarray:
    """
    Transforms discrete nodal loads to the global frame.

    Parameters
    ----------
    nodal_loads : numpy.ndarray
        A 3d array of shape (nE, nEVAB, nRHS).

    Returns
    -------
    numpy.ndarray
        A numpy array of shape (nE, nEVAB, nRHS).

    """
    nodal_loads = np.swapaxes(nodal_loads, 1, 2)
    # (nE, nNE * nDOF, nRHS) -> (nE, nRHS, nNE * nDOF)
    nodal_loads = ascont(nodal_loads)
    nodal_loads = tr_element_vectors_bulk_multi(nodal_loads, dcm)
    nodal_loads = np.swapaxes(nodal_loads, 1, 2)
    # (nE, nRHS, nNE * nDOF) -> (nE, nNE * nDOF, nRHS)
    return nodal_loads
