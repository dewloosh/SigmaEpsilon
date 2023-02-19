import numpy as np
from numpy import ndarray
from numba import njit, prange

__cache = True


@njit(nogil=True, cache=__cache)
def shape_function_values(x: float, L: float = 2) -> ndarray:
    """
    Evaluates the shape functions at a point x in the range [-1, 1].

    Parameters
    ----------
    x : float
        A point in the range [-1, 1].
    L : float, Optional
        Length of the beam element. Default is 2.

    Returns
    -------
    numpy.ndarray
        A 2d float array.
    """
    return np.array(
        [
            [
                0.5 * x * (x - 1),
                x**2 * (0.75 * x + 1.0) * (x - 1) ** 2,
                x**2 * (0.75 * x + 1.0) * (x - 1) ** 2,
                0.5 * x * (x - 1),
                -0.125 * L * x**2 * (x - 1) ** 2 * (x + 1),
                0.125 * L * x**2 * (x - 1) ** 2 * (x + 1),
            ],
            [
                1.0 - 1.0 * x**2,
                1.0 * (x - 1) ** 2 * (x + 1) ** 2,
                1.0 * (x - 1) ** 2 * (x + 1) ** 2,
                1.0 - 1.0 * x**2,
                L * x * (-0.5 * x**4 + 1.0 * x**2 - 0.5),
                0.5 * L * x * (x - 1) ** 2 * (x + 1) ** 2,
            ],
            [
                0.5 * x * (x + 1),
                x**2 * (1.0 - 0.75 * x) * (x + 1) ** 2,
                x**2 * (1.0 - 0.75 * x) * (x + 1) ** 2,
                0.5 * x * (x + 1),
                -0.125 * L * x**2 * (x - 1) * (x + 1) ** 2,
                0.125 * L * x**2 * (x - 1) * (x + 1) ** 2,
            ],
        ]
    )


@njit(nogil=True, cache=__cache)
def shape_function_values_L(x: float) -> ndarray:
    """
    Evaluates the Lagrangian shape functions at a point x in the range [-1, 1].

    Parameters
    ----------
    x : float
        A point in the range [-1, 1].

    Returns
    -------
    numpy.ndarray
        A 2d float array.
    """
    return np.array([0.5 * x * (x - 1), 1.0 - 1.0 * x**2, 0.5 * x * (x + 1)])


@njit(nogil=True, cache=__cache)
def shape_function_derivatives_1(x: float, L: float = 2) -> ndarray:
    """
    Evaluates the first derivatives of the shape
    functions at a point x in the range [-1, 1].

    Parameters
    ----------
    x : float
        A point in the range [-1, 1].
    L : float, Optional
        Length of the beam element. Default is 2.

    Returns
    -------
    numpy.ndarray
        A 2d float array.
    """
    return np.array(
        [
            [
                1.0 * x - 0.5,
                3.75 * x * (x - 1) * (1.0 * x - 0.533333333333333) * (x + 1),
                3.75 * x * (x - 1) * (1.0 * x - 0.533333333333333) * (x + 1),
                1.0 * x - 0.5,
                L * x * (-0.625 * x**3 + 0.5 * x**2 + 0.375 * x - 0.25),
                0.625 * L * x * (x - 1) * (1.0 * x**2 + 0.2 * x - 0.4),
            ],
            [
                -2.0 * x,
                4.0 * x * (x**2 - 1),
                4.0 * x * (x**2 - 1),
                -2.0 * x,
                L * (-2.5 * x**4 + 3.0 * x**2 - 0.5),
                L * (2.5 * x**4 - 3.0 * x**2 + 0.5),
            ],
            [
                1.0 * x + 0.5,
                -3.75 * x * (x - 1) * (1.0 * x + 0.533333333333333) * (x + 1),
                -3.75 * x * (x - 1) * (1.0 * x + 0.533333333333333) * (x + 1),
                1.0 * x + 0.5,
                0.625 * L * x * (x + 1) * (-1.0 * x**2 + 0.2 * x + 0.4),
                L * x * (0.625 * x**3 + 0.5 * x**2 - 0.375 * x - 0.25),
            ],
        ]
    )


@njit(nogil=True, cache=__cache)
def shape_function_derivatives_2(x: float, L: float = 2) -> ndarray:
    """
    Evaluates the second derivatives of the shape
    functions at a point x in the range [-1, 1].

    Parameters
    ----------
    x : float
        A point in the range [-1, 1].
    L : float, Optional
        Length of the beam element. Default is 2.

    Returns
    -------
    numpy.ndarray
        A 2d float array.
    """
    return np.array(
        [
            [
                1.00000000000000,
                15.0 * x**3 - 6.0 * x**2 - 7.5 * x + 2.0,
                15.0 * x**3 - 6.0 * x**2 - 7.5 * x + 2.0,
                1.00000000000000,
                L * (-2.5 * x**3 + 1.5 * x**2 + 0.75 * x - 0.25),
                L * (2.5 * x**3 - 1.5 * x**2 - 0.75 * x + 0.25),
            ],
            [
                -2.00000000000000,
                12.0 * x**2 - 4.0,
                12.0 * x**2 - 4.0,
                -2.00000000000000,
                L * x * (6.0 - 10.0 * x**2),
                L * x * (10.0 * x**2 - 6.0),
            ],
            [
                1.00000000000000,
                -15.0 * x**3 - 6.0 * x**2 + 7.5 * x + 2.0,
                -15.0 * x**3 - 6.0 * x**2 + 7.5 * x + 2.0,
                1.00000000000000,
                L * (-2.5 * x**3 - 1.5 * x**2 + 0.75 * x + 0.25),
                L * (2.5 * x**3 + 1.5 * x**2 - 0.75 * x - 0.25),
            ],
        ]
    )


@njit(nogil=True, cache=__cache)
def shape_function_derivatives_3(x: float, L: float = 2) -> ndarray:
    """
    Evaluates the third derivatives of the shape
    functions at a point x in the range [-1, 1].

    Parameters
    ----------
    x : float
        A point in the range [-1, 1].
    L : float, Optional
        Length of the beam element. Default is 2.

    Returns
    -------
    numpy.ndarray
        A 2d float array.
    """
    return np.array(
        [
            [
                0,
                45.0 * x**2 - 12.0 * x - 7.5,
                45.0 * x**2 - 12.0 * x - 7.5,
                0,
                L * (-7.5 * x**2 + 3.0 * x + 0.75),
                L * (7.5 * x**2 - 3.0 * x - 0.75),
            ],
            [
                0,
                24.0 * x,
                24.0 * x,
                0,
                L * (6.0 - 30.0 * x**2),
                L * (30.0 * x**2 - 6.0),
            ],
            [
                0,
                -45.0 * x**2 - 12.0 * x + 7.5,
                -45.0 * x**2 - 12.0 * x + 7.5,
                0,
                L * (-7.5 * x**2 - 3.0 * x + 0.75),
                L * (7.5 * x**2 + 3.0 * x - 0.75),
            ],
        ]
    )


@njit(nogil=True, cache=__cache)
def shape_function_derivatives_L(x: float) -> ndarray:
    """
    Evaluates the first derivatives of the shape
    functions at a point x in the range [-1, 1].

    Parameters
    ----------
    x : float
        A point in the range [-1, 1].

    Returns
    -------
    numpy.ndarray
        A 2d float array.
    """
    return np.array([1.0 * x - 0.5, -2.0 * x, 1.0 * x + 0.5])


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_values_bulk(x: ndarray, L: ndarray) -> ndarray:
    """
    Evaluates the shape functions at several points
    in the range [-1, 1].

    Parameters
    ----------
    x : numpy.ndarray
        1d array of floats in the range [-1, -1].
    L : numpy.ndarray
        Lengths of the beam elements. Default is None.

    Returns
    -------
    numpy.ndarray
        3d float array of shape (nP, nNE=3, nDOF=6).
    """
    nP = x.shape[0]
    nE = L.shape[0]
    res = np.zeros((nE, nP, 3, 6), dtype=x.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            res[iE, iP] = shape_function_values(x[iP], L[iE])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_values_multi_L(x: ndarray) -> ndarray:
    """
    Evaluates the Lagrangian shape functions at several points
    in the range [-1, 1].

    Parameters
    ----------
    x : numpy.ndarray
        1d array of floats in the range [-1, -1].

    Returns
    -------
    numpy.ndarray
        3d float array of shape (nP, nNE=3).
    """
    nP = x.shape[0]
    res = np.zeros((nP, 3), dtype=x.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_values_L(x[iP])
    return res


@njit(nogil=True, cache=__cache)
def shape_function_derivatives(x: float, L: float = 2) -> ndarray:
    """
    Evaluates the derivatives of the shape
    functions at a point x in the range [-1, 1].

    Parameters
    ----------
    x : float
        A point in the range [-1, 1].
    L : float, Optional
        Length of the beam element. Default is 2.

    Returns
    -------
    numpy.ndarray
        3d float array of shape (nNE=3, nDOF=6, 3).
    """
    res = np.zeros((3, 6, 3))
    res[:, :, 0] = shape_function_derivatives_1(x, L)
    res[:, :, 1] = shape_function_derivatives_2(x, L)
    res[:, :, 2] = shape_function_derivatives_3(x, L)
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_derivatives_bulk(x: ndarray, L: ndarray) -> ndarray:
    """
    Evaluates the derivatives of the shape
    functions at several points in the range [-1, 1].

    Parameters
    ----------
    x : numpy.ndarray
        1d array of floats in the range [-1, -1].
    L : numpy.ndarray
        Lengths of the beam elements. Default is None.

    Returns
    -------
    numpy.ndarray
        5d float array of shape (nE, nP, nNE=3, nDOF=6, 3).
    """
    nP = x.shape[0]
    nE = L.shape[0]
    res = np.zeros((nE, nP, 3, 6, 3), dtype=x.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            res[iE, iP] = shape_function_derivatives(x[iP], L[iE])
    return res


@njit(nogil=True, cache=__cache)
def shape_function_derivatives_multi_L(x: ndarray) -> ndarray:
    """
    Evaluates the derivatives of the Lagrangian shape
    functions at multiple points x in the range [-1, 1].

    Parameters
    ----------
    x : numpy.ndarray
        1d array of floats in the range [-1, -1].

    Returns
    -------
    numpy.ndarray
        2d float array of shape (nP, nNE=3), where nP is the number
        of coordinates in 'x', nNE is the number of nodes of the cell.
    """
    nP = x.shape[0]
    res = np.zeros((nP, 3), dtype=x.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_derivatives_L(x[iP])
    return res
