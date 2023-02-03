import numpy as np
from numpy import ndarray
from numba import njit, prange

__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def calc_beam_stresses_2d(forces: ndarray, factors: ndarray):
    """
    Returns the stresses in the order s11, s22, s33, s12, s13, s23.

    Parameters
    ----------
    forces : numpy.ndarray
        A 2d float array of shape (nX, nSTRE) where 'nX' is the number of records
        in the data and 'nSTRE' is the number of internal force components.
    factors (nS, nSTRE, nPS)
        Factors for every stress (nS) and internal force (nSTRE) component
        at a nPS number of points of a section.

    Returns
    -------
    numpy.ndarray
        A 3d array of stresses of shape (nX, nPS, nS).
    """
    nX, nSTRE = forces.shape
    nS, _, nPS = factors.shape
    res = np.zeros((nX, nPS, nS))
    for iX in prange(nX):
        for iPS in prange(nPS):
            for iS in prange(nS):
                for iSTRE in range(nSTRE):
                    res[iX, iPS, iS] += forces[iX, iSTRE] * factors[iS, iSTRE, iPS]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def calc_beam_stresses_4d(forces: ndarray, factors: ndarray):
    """
    Returns the stresses in the order s11, s22, s33, s12, s13, s23.

    Parameters
    ----------
    forces : numpy.ndarray
        A 4d float array of shape (nE, nP, nSTRE, nRHS), representing
        all internal forces (nSTRE) for multiple elements(nE), load cases(nRHS)
        and evaluation points (nP).
    factors (nS, nSTRE, nPS)
        Factors for every stress (nS) and internal force (nSTRE) component
        at a nPS number of points in the section.

    Returns
    -------
    numpy.ndarray
        A 5d array of stresses of shape (nE, nP, nPS, nS, nRHS).
    """
    nE, nP, nSTRE, nRHS = forces.shape
    nS, _, nPS = factors.shape
    res = np.zeros((nE, nP, nPS, nS, nRHS))
    for iE in prange(nE):
        for iP in prange(nP):
            for iPS in prange(nPS):
                for iS in prange(nS):
                    for iRHS in prange(nRHS):
                        for iSTRE in range(nSTRE):
                            res[iE, iP, iPS, iS, iRHS] += (
                                forces[iE, iP, iSTRE, iRHS] * factors[iS, iSTRE, iPS]
                            )
    return res
