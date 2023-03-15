from typing import Iterable, Union, Tuple

import numpy as np
from numba import njit, prange
from numpy import ndarray, sin, cos, ndarray, pi as PI

from neumann import atleast1d, atleast2d, atleast3d


def lhs_Navier(
    size: Union[float, Tuple[float]],
    shape: Union[int, Tuple[int]],
    *,
    D: Union[float, ndarray],
    S: Union[float, ndarray] = None,
    **kw
) -> ndarray:
    """
    Returns coefficient matrices for a Navier solution, for a single or
    multiple left-hand sides.

    Parameters
    ----------
    size : Union[float, Tuple[float]]
        The size of the problem. Scalar for a beam, 2-tuple for a plate.
    shape : Union[int, Tuple[int]]
        The number of harmonic terms used. Scalar for a beam, 2-tuple for a plate.
    D : Union[float, ndarray]
        2d or 3d float array of bending stiffnesses for a plate, scalar or 1d float array
        for a beam.
    S : Union[float, ndarray], Optional
        2d or 3d float array of shear stiffnesses for a plate, scalar or 1d float array
        for a beam. Only for Mindlin-Reissner plates and Euler-Bernoulli beams.
        plates. Default is None.

    Note
    ----
    Shear stiffnesses must include shear correction.

    Returns
    -------
    numpy.ndarray
        The coefficients as an array. See the documentation of the corresponding
        function for further details.

    See Also
    --------
    :func:`lhs_Navier_Mindlin`
    :func:`lhs_Navier_Kirchhoff`
    :func:`lhs_Navier_Bernoulli`
    :func:`lhs_Navier_Timoshenko`
    """
    if isinstance(shape, Iterable):  # plate problem
        if S is None:
            return lhs_Navier_Kirchhoff(size, shape, atleast3d(D))
        else:
            return lhs_Navier_Mindlin(size, shape, atleast3d(D), atleast3d(S))
    else:  # beam problem
        if S is None:
            return lhs_Navier_Bernoulli(size, shape, atleast1d(D))
        else:
            return lhs_Navier_Timoshenko(size, shape, atleast1d(D), atleast1d(S))


@njit(nogil=True, parallel=True, cache=True)
def lhs_Navier_Mindlin(size: tuple, shape: tuple, D: ndarray, S: ndarray) -> ndarray:
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

    Note
    ----
    The shear stiffness values must include the shear correction factor.

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
        D11, D12, D22, D66 = D[iLHS, 0, 0], D[iLHS, 0, 1], D[iLHS, 1, 1], D[iLHS, 2, 2]
        S44, S55 = S[iLHS, 0, 0], S[iLHS, 1, 1]
        for m in prange(1, M + 1):
            for n in prange(1, N + 1):
                iMN = (m - 1) * N + n - 1
                res[iLHS, iMN, 0, 0] = (
                    -(PI**2) * D22 * n**2 / Ly**2
                    - PI**2 * D66 * m**2 / Lx**2
                    - S44
                )
                res[iLHS, iMN, 0, 1] = PI**2 * D12 * m * n / (
                    Lx * Ly
                ) + PI**2 * D66 * m * n / (Lx * Ly)
                res[iLHS, iMN, 0, 2] = PI * S44 * n / Ly
                res[iLHS, iMN, 1, 0] = -(PI**2) * D12 * m * n / (
                    Lx * Ly
                ) - PI**2 * D66 * m * n / (Lx * Ly)
                res[iLHS, iMN, 1, 1] = (
                    PI**2 * D11 * m**2 / Lx**2
                    + PI**2 * D66 * n**2 / Ly**2
                    + S55
                )
                res[iLHS, iMN, 1, 2] = PI * S55 * m / Lx
                res[iLHS, iMN, 2, 0] = -PI * S44 * n / Ly
                res[iLHS, iMN, 2, 1] = PI * S55 * m / Lx
                res[iLHS, iMN, 2, 2] = (
                    PI**2 * S44 * n**2 / Ly**2 + PI**2 * S55 * m**2 / Lx**2
                )
    return res


@njit(nogil=True, parallel=True, cache=True)
def lhs_Navier_Kirchhoff(size: tuple, shape: tuple, D: ndarray) -> ndarray:
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
        D11, D12, D22, D66 = D[iLHS, 0, 0], D[iLHS, 0, 1], D[iLHS, 1, 1], D[iLHS, 2, 2]
        for m in prange(1, M + 1):
            for n in prange(1, N + 1):
                iMN = (m - 1) * N + n - 1
                res[iLHS, iMN] = (
                    PI**4 * D11 * m**4 / Lx**4
                    + 2 * PI**4 * D12 * m**2 * n**2 / (Lx**2 * Ly**2)
                    + PI**4 * D22 * n**4 / Ly**4
                    + 4 * PI**4 * D66 * m**2 * n**2 / (Lx**2 * Ly**2)
                )
    return res


@njit(nogil=True, parallel=True, cache=True)
def lhs_Navier_Bernoulli(L: float, N: int, EI: ndarray) -> ndarray:
    """
    JIT compiled function, that returns coefficient matrices for a Navier
    solution for multiple left-hand sides.

    Parameters
    ----------
    L : float
        The length of the beam.
    N : int
        The number of harmonic terms.
    EI : numpy.ndarray
        1d float array of bending stiffnesses.

    Returns
    -------
    numpy.ndarray
        2d float array of coefficients.
    """
    nLHS = EI.shape[0]
    res = np.zeros((nLHS, N), dtype=EI.dtype)
    for iLHS in prange(nLHS):
        for n in prange(1, N + 1):
            res[iLHS, n - 1] = PI**4 * EI[iLHS] * n**4 / L**4
    return res


@njit(nogil=True, parallel=True, cache=True)
def lhs_Navier_Timoshenko(L: float, N: int, EI: ndarray, GA: ndarray) -> ndarray:
    """
    JIT compiled function, that returns coefficient matrices for a Navier
    solution for multiple left-hand sides.

    Parameters
    ----------
    L : float
        The length of the beam.
    N : int
        The number of harmonic terms.
    EI : numpy.ndarray
        1d float array of bending stiffnesses.
    GA : numpy.ndarray
        1d float array of shear stiffnesses.

    Note
    ----
    The shear stiffness values must include the shear correction factor.

    Returns
    -------
    numpy.ndarray
        4d float array of coefficients.
    """
    nLHS = EI.shape[0]
    res = np.zeros((nLHS, N, 2, 2), dtype=EI.dtype)
    for iLHS in prange(nLHS):
        for n in prange(1, N + 1):
            iN = n - 1
            c1 = PI * n / L
            c2 = PI**2 * n**2 / L**2
            res[iLHS, iN, 0, 0] = c2 * GA[iLHS]
            res[iLHS, iN, 0, 1] = -c1 * GA[iLHS]
            res[iLHS, iN, 1, 0] = res[iLHS, iN, 0, 1]
            res[iLHS, iN, 1, 1] = GA[iLHS] + c2 * EI[iLHS]
    return res


@njit(nogil=True, parallel=True, cache=True)
def rhs_Bernoulli(coeffs: ndarray, L: float) -> ndarray:
    """
    Calculates unknowns for Bernoulli Beams.
    """
    nRHS, N = coeffs.shape[:2]
    res = np.zeros((nRHS, N))
    c = PI / L
    for i in prange(nRHS):
        for n in prange(N):
            res[i, n] = coeffs[i, n, 0] + coeffs[i, n, 1] * c * (n + 1)
    return res


def rhs_line_const(L: float, N: int, v: ndarray, x: ndarray) -> ndarray:
    """
    Returns coefficients for constant loads over line segments in
    the order [f, m].
    """
    return _line_const_(L, N, atleast2d(x), atleast2d(v))


@njit(nogil=True, parallel=True, cache=True)
def _line_const_(L: float, N: int, x: ndarray, values: ndarray) -> ndarray:
    nR = values.shape[0]
    rhs = np.zeros((nR, N, 2), dtype=x.dtype)
    for iR in prange(nR):
        for n in prange(1, N + 1):
            iN = n - 1
            xa, xb = x[iR]
            f, m = values[iR]
            c = PI * n / L
            rhs[iR, iN, 0] = (2 * f / (PI * n)) * (cos(c * xa) - cos(c * xb))
            rhs[iR, iN, 1] = (2 * m / (PI * n)) * (sin(c * xb) - sin(c * xa))
    return rhs


def rhs_rect_const(size: tuple, shape: tuple, x: ndarray, v: ndarray) -> ndarray:
    """
    Returns coefficients for constant loads over rectangular patches
    in the order [f, mx, my].
    """
    return _rect_const_(size, shape, atleast2d(v), atleast3d(x))


@njit(nogil=True, cache=True)
def __rect_const__(
    size: tuple,
    m: int,
    n: int,
    xc: float,
    yc: float,
    w: float,
    h: float,
    values: ndarray,
) -> ndarray:
    Lx, Ly = size
    f, mx, my = values
    return np.array(
        [
            16
            * f
            * sin((1 / 2) * PI * m * w / Lx)
            * sin(PI * m * xc / Lx)
            * sin((1 / 2) * PI * h * n / Ly)
            * sin(PI * n * yc / Ly)
            / (PI**2 * m * n),
            16
            * mx
            * sin((1 / 2) * PI * m * w / Lx)
            * sin((1 / 2) * PI * h * n / Ly)
            * sin(PI * n * yc / Ly)
            * cos(PI * m * xc / Lx)
            / (PI**2 * m * n),
            16
            * my
            * sin((1 / 2) * PI * m * w / Lx)
            * sin(PI * m * xc / Lx)
            * sin((1 / 2) * PI * h * n / Ly)
            * cos(PI * n * yc / Ly)
            / (PI**2 * m * n),
        ]
    )


@njit(nogil=True, parallel=True, cache=True)
def _rect_const_(
    size: tuple, shape: tuple, values: ndarray, points: ndarray
) -> ndarray:
    nR = values.shape[0]
    M, N = shape
    rhs = np.zeros((nR, M * N, 3), dtype=points.dtype)
    for iR in prange(nR):
        xmin, ymin = points[iR, 0]
        xmax, ymax = points[iR, 1]
        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        w = np.abs(xmax - xmin)
        h = np.abs(ymax - ymin)
        for m in prange(1, M + 1):
            for n in prange(1, N + 1):
                mn = (m - 1) * N + n - 1
                rhs[iR, mn, :] = __rect_const__(size, m, n, xc, yc, w, h, values[iR])
    return rhs


def rhs_conc_1d(L: float, N: int, v: ndarray, x: ndarray) -> ndarray:
    return _conc1d_(L, N, atleast2d(v), atleast1d(x))


@njit(nogil=True, parallel=True, cache=True)
def _conc1d_(L: tuple, N: tuple, values: ndarray, points: ndarray) -> ndarray:
    nRHS = values.shape[0]
    c = 2 / L
    rhs = np.zeros((nRHS, N, 2), dtype=points.dtype)
    PI = np.pi
    for iRHS in prange(nRHS):
        x = points[iRHS]
        f, m = values[iRHS]
        Sx = PI * x / L
        for n in prange(1, N + 1):
            i = n - 1
            rhs[iRHS, i, 0] = c * f * sin(n * Sx)
            rhs[iRHS, i, 1] = c * m * cos(n * Sx)
    return rhs


def rhs_conc_2d(size: tuple, shape: tuple, v: ndarray, x: ndarray) -> ndarray:
    return _conc2d_(size, shape, atleast2d(v), atleast2d(x))


@njit(nogil=True, parallel=True, cache=True)
def _conc2d_(size: tuple, shape: tuple, values: ndarray, points: ndarray) -> ndarray:
    nRHS = values.shape[0]
    Lx, Ly = size
    c = 4 / Lx / Ly
    M, N = shape
    rhs = np.zeros((nRHS, M * N, 3), dtype=points.dtype)
    PI = np.pi
    for iRHS in prange(nRHS):
        x, y = points[iRHS]
        fz, mx, my = values[iRHS]
        Sx = PI * x / Lx
        Sy = PI * y / Ly
        for m in prange(1, M + 1):
            for n in prange(1, N + 1):
                mn = (m - 1) * N + n - 1
                rhs[iRHS, mn, 0] = c * fz * sin(m * Sx) * sin(n * Sy)
                rhs[iRHS, mn, 1] = c * mx * cos(m * Sx) * sin(n * Sy)
                rhs[iRHS, mn, 2] = c * my * sin(m * Sx) * cos(n * Sy)
    return rhs
