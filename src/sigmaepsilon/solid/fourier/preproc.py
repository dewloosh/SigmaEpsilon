# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange
from numpy import ndarray, pi as PI

from neumann import squeeze
from neumann.array import atleast3d


@squeeze(True)
def lhs_Navier(size: tuple, shape: tuple, *args, D: ndarray,
               S: ndarray = None, model: str = 'mindlin', **kwargs):
    """
    Returns coefficient matrix for a Navier solution, for a single or 
    multiple left-hand sides.

    Parameters
    ----------
    size : tuple
        Tuple of floats, containing the sizes of the rectagle.

    shape : tuple
        Tuple of integers, containing the number of harmonic terms
        included in both directions.

    D : numpy.ndarray
        2d or 3d float array of bending stiffnesses.

    S : numpy.ndarray, Optional
        2d or 3d float array of shear stiffnesses. Default is None.

    squeeze : boolean, optional
        Removes single-dimensional entries from the shape of the 
        resulting array. Default is True.

    Returns
    -------
    numpy.ndarray
        3d or 4d float array of coefficients. The shape depends on
        the shape of the input parameters.
        
    """
    if model.lower() in ['mindlin', 'm']:
        return lhs_Mindlin(size, shape, atleast3d(D), atleast3d(S))
    elif model.lower() in ['kirchhoff', 'k']:
        return lhs_Kirchhoff(size, shape, atleast3d(D))


@njit(nogil=True, parallel=True, cache=True)
def lhs_Mindlin(size: tuple, shape: tuple, D: np.ndarray, S: np.ndarray):
    """
    JIT compiled function, that returns coefficient matrices for a Navier 
    solution for multiple left-hand sides.

    Parameters
    ----------
    size : tuple
        Tuple of floats, containing the sizes of the rectagle.

    shape : tuple
        Tuple of integers, containing the number of harmonic terms
        included in both directions.

    D : numpy.ndarray
        3d float array of bending stiffnesses.

    S : numpy.ndarray
        3d float array of shear stiffnesses.

    Returns
    -------
    numpy.ndarray
        4d float array of coefficients.
        
    """
    Lx, Ly = size
    nLHS = D.shape[0]
    M, N = shape
    res = np.zeros((nLHS, M * N, 3, 3), dtype=D.dtype)
    for iLHS in prange(nLHS):
        D11, D12, D22, D66 = D[iLHS, 0, 0], D[iLHS, 0, 1], \
            D[iLHS, 1, 1], D[iLHS, 2, 2]
        S44, S55 = S[iLHS, 0, 0], S[iLHS, 1, 1]
        for m in prange(1, M + 1):
            for n in prange(1, N + 1):
                iMN = (m - 1) * N + n - 1
                res[iLHS, iMN, 0, 0] = -PI**2*D22*n**2 / \
                    Ly**2 - PI**2*D66*m**2/Lx**2 - S44
                res[iLHS, iMN, 0, 1] = PI**2*D12*m * \
                    n/(Lx*Ly) + PI**2*D66*m*n/(Lx*Ly)
                res[iLHS, iMN, 0, 2] = PI*S44*n/Ly
                res[iLHS, iMN, 1, 0] = -PI**2*D12*m * \
                    n/(Lx*Ly) - PI**2*D66*m*n/(Lx*Ly)
                res[iLHS, iMN, 1, 1] = PI**2*D11*m**2 / \
                    Lx**2 + PI**2*D66*n**2/Ly**2 + S55
                res[iLHS, iMN, 1, 2] = PI*S55*m/Lx
                res[iLHS, iMN, 2, 0] = -PI*S44*n/Ly
                res[iLHS, iMN, 2, 1] = PI*S55*m/Lx
                res[iLHS, iMN, 2, 2] = PI**2*S44 * \
                    n**2/Ly**2 + PI**2*S55*m**2/Lx**2
    return res


@njit(nogil=True, parallel=True, cache=True)
def lhs_Kirchhoff(size: tuple, shape: tuple, D: np.ndarray):
    """
    JIT compiled function, that returns coefficient matrices for a Navier 
    solution for multiple left-hand sides.

    Parameters
    ----------
    size : tuple
        Tuple of floats, containing the sizes of the rectagle.

    shape : tuple
        Tuple of integers, containing the number of harmonic terms
        included in both directions.

    D : numpy.ndarray
        3d float array of bending stiffnesses.

    Returns
    -------
    numpy.ndarray
        2d float array of coefficients.
        
    """
    Lx, Ly = size
    nLHS = D.shape[0]
    M, N = shape
    res = np.zeros((nLHS, M * N), dtype=D.dtype)
    for iLHS in prange(nLHS):
        D11, D12, D22, D66 = D[iLHS, 0, 0], D[iLHS, 0, 1], \
            D[iLHS, 1, 1], D[iLHS, 2, 2]
        for m in prange(1, M + 1):
            for n in prange(1, N + 1):
                iMN = (m - 1) * N + n - 1
                res[iLHS, iMN] = PI**4*D11*m**4/Lx**4 + \
                    2*PI**4*D12*m**2*n**2/(Lx**2*Ly**2) + \
                        PI**4*D22*n**4/Ly**4 + \
                            4*PI**4*D66*m**2*n**2/(Lx**2*Ly**2)
    return res
