from numba import njit, prange
import numpy as np
from numpy import ndarray

__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def strain_displacement_matrix_timoshenko_bulk(dshp: ndarray, jac: ndarray) -> ndarray:
    """
    Returns the matrix expressing the relationship of generalized strains
    and generalized displacements of a Timoshenko beam element. The order
    of strain components is
        0 : normal strain x
        1 : shear strain xy
        2 : shear strain xz
        3 : curvature x
        4 : curvature y
        5 : curvature z

    Parameters
    ----------
    dshp: numpy.ndarray
        First derivatives for every node (nNE) and integration
        point (nP) as a 2d array of shape (nP, nNE).
    jac: numpy.ndarray
        Jacobian determinants for every element (nE) and integration point (nP) as
        a 2d array of shape (nE, nP).

    Returns
    -------
    numpy.ndarray
        Approximation coefficients for every generalized strain and every
        shape function as a 2d float array of shape (nSTRE=6, nNE * nDOF=6).
    """
    nDOFN, nSTRE = 6, 6
    nE = jac.shape[0]
    nP, nNE = dshp.shape[:2]
    nTOTV = nNE * nDOFN
    B = np.zeros((nE, nP, nSTRE, nTOTV), dtype=dshp.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            gdshp = dshp[iP] / jac[iE, iP]
            for i in prange(nNE):
                B[iE, iP, 0, 0 + i * nDOFN] = gdshp[i]
                B[iE, iP, 1, 1 + i * nDOFN] = gdshp[i]
                B[iE, iP, 1, 5 + i * nDOFN] = -1
                B[iE, iP, 2, 2 + i * nDOFN] = gdshp[i]
                B[iE, iP, 2, 4 + i * nDOFN] = 1
                B[iE, iP, 3, 3 + i * nDOFN] = gdshp[i]
                B[iE, iP, 4, 4 + i * nDOFN] = gdshp[i]
                B[iE, iP, 5, 5 + i * nDOFN] = gdshp[i]
    return B