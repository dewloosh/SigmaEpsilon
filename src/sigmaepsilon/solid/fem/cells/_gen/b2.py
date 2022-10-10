# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit, prange
__cache = True


@njit(nogil=True, cache=__cache)
def shape_function_values(x, L):
    """
    Evaluates the shape functions at a point x in the range [-1, 1].
    """
    return np.array([
        [
            0.5 - 0.5 * x,
            0.25 * x**3 - 0.75 * x + 0.5,
            0.25 * x**3 - 0.75 * x + 0.5,
            0.5 - 0.5 * x,
            -0.125 * L * (x - 1)**2 * (x + 1),
            0.125 * L * (x - 1)**2 * (x + 1)
        ],
        [
            0.5 * x + 0.5,
            -0.25 * x**3 + 0.75 * x + 0.5,
            -0.25 * x**3 + 0.75 * x + 0.5,
            0.5 * x + 0.5,
            -0.125 * L * (x - 1) * (x + 1)**2,
            0.125 * L * (x - 1) * (x + 1)**2
        ]
    ])


@njit(nogil=True, cache=__cache)
def shape_function_derivatives_1(x, L):
    """
    Evaluates the first derivatives of the shape
    functions at a point x in the range [-1, 1].
    """
    return np.array([
        [
            -0.500000000000000,
            0.75 * x**2 - 0.75,
            0.75 * x**2 - 0.75,
            -0.500000000000000,
            L * (-0.375 * x**2 + 0.25 * x + 0.125),
            0.375 * L * (x - 1) * (1.0 * x + 0.333333333333333)
        ],
        [
            0.500000000000000,
            0.75 - 0.75 * x**2,
            0.75 - 0.75 * x**2,
            0.500000000000000,
            L * (-0.375 * x**2 - 0.25 * x + 0.125),
            0.375 * L * (1.0 * x - 0.333333333333333) * (x + 1)
        ]
    ])


@njit(nogil=True, cache=__cache)
def shape_function_derivatives_2(x, L):
    """
    Evaluates the second derivatives of the shape
    functions at a point x in the range [-1, 1].
    """
    return np.array([
        [
            0,
            1.5 * x,
            1.5 * x,
            0,
            L * (0.25 - 0.75 * x),
            L * (0.75 * x - 0.25)
        ],
        [
            0,
            -1.5 * x,
            -1.5 * x,
            0,
            -L * (0.75 * x + 0.25),
            L * (0.75 * x + 0.25)
        ]
    ])


@njit(nogil=True, cache=__cache)
def shape_function_derivatives_3(x, L):
    """
    Evaluates the third derivatives of the shape
    functions at a point x in the range [-1, 1].
    """
    return np.array([
        [
            0,
            1.50000000000000,
            1.50000000000000,
            0,
            -0.75 * L,
            0.75 * L
        ],
        [
            0,
            -1.50000000000000,
            -1.50000000000000,
            0,
            -0.75 * L,
            0.75 * L
        ]
    ])


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_values_bulk(x: ndarray, L: ndarray):
    """
    Evaluates the shape functions at several points
    in the range [-1, 1].

    Parameters
    ----------
    x : 1d numpy float array
        The points of interest in the range [-1, -1]

    Returns
    -------
    numpy float array of shape (nE, nP, nNE, nDOF=6)
    """
    nP = x.shape[0]
    nE = L.shape[0]
    res = np.zeros((nE, nP, 2, 6), dtype=x.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            res[iE, iP] = shape_function_values(x[iP], L[iE])
    return res


@njit(nogil=True, cache=__cache)
def shape_function_derivatives(x, L):
    """
    Evaluates the derivatives of the shape
    functions at a point x in the range [-1, 1].

    Parameters
    ----------
    x : float
        The point of interest in the range [-1, -1]

    djac : float
        Determinant of the Jacobi matrix of local-global transformation
        between the master elment and the actual element.
        Default is 1.0.

    Returns
    -------
    numpy float array of shape (nNE, nDOF=6, 3)
    """
    res = np.zeros((2, 6, 3))
    res[:, :, 0] = shape_function_derivatives_1(x, L)
    res[:, :, 1] = shape_function_derivatives_2(x, L)
    res[:, :, 2] = shape_function_derivatives_3(x, L)
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_derivatives_bulk(x: ndarray, L: ndarray):
    """
    Evaluates the derivatives of the shape
    functions at several points in the range [-1, 1].

    Returns
    -------
    dshp (nE, nP, nNE, nDOF=6, 3)
    """
    nP = x.shape[0]
    nE = L.shape[0]
    res = np.zeros((nE, nP, 2, 6, 3), dtype=x.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            res[iE, iP] = shape_function_derivatives(x[iP], L[iE])
    return res
