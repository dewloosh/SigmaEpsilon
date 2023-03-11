import numpy as np
from numpy import ndarray
from numba import njit, prange

__all__ = []

__cache__ = True
__NDOFN__ = 6


@njit(nogil=True, parallel=True, cache=__cache__)
def global_shape_function_derivatives_Bernoulli_bulk(dshp: ndarray, jac: ndarray):
    """
    Calculates derivatives of shape functions w.r.t. the global axes
    using derivatives along local axes evaulated at some points in
    the interval [-1, 1], and jacobians of local-to-global mappings.

    Parameters
    ----------
    dshp: numpy.ndarray
        A float array of shape (nE, nP, nNE, nDOF=6, 3)
        Derivatives of shape functions evaluated at an 'nP' number of
        points of an 'nE' number of elements.
    jac: numpy.ndarray
        A float array of shape (nE, nP, 1, 1)
        Jacobi determinants, evaluated for each point in each cell.

    Returns
    -------
    numpy.ndarray
        5d float array of shape (nE, nP, nNE, nDOF=6, 3).
    """
    nE, nP, nNE = dshp.shape[:3]
    res = np.zeros((nE, nP, nNE, __NDOFN__, 3), dtype=dshp.dtype)
    for iE in prange(nE):
        for jP in prange(nP):
            J = jac[iE, jP, 0, 0]
            res[iE, jP, :, :, 0] = dshp[iE, jP, :, :, 0] / J
            res[iE, jP, :, :, 1] = dshp[iE, jP, :, :, 1] / J**2
            res[iE, jP, :, :, 2] = dshp[iE, jP, :, :, 2] / J**3
    return res


@njit(nogil=True, parallel=True, cache=__cache__)
def shape_function_matrix_Bernoulli(shp: ndarray, gdshp: ndarray):
    """
    Returns the shape function matrix of a single Bernoulli beam.

    The input contains evaluations of the shape functions and at least
    the first global derivatives of the shape functions.

    Parameters
    ----------
    shp: numpy.ndarray
        A float array of shape (nNE, nDOF=6), being the shape
        functions evaluated at a single point of a single cell.
    gdshp: numpy.ndarray
        A float array of shape (nNE, nDOF=6, 3), being derivatives of shape
        functions evaluated at a single point of a single cell.

    Returns
    -------
    numpy.ndarray
        2d float array of shape (nDOF, nDOF * nNODE).

    Notes
    -----
    The approximation applies a mixture of Lagrange(L) and Hermite(H)
    polynomials, in the order [L, H, H, L, H, H].

    See Also
    --------
    :func:`shape_function_matrix_bulk`
    """
    nNE = shp.shape[0]
    res = np.zeros((__NDOFN__, nNE * __NDOFN__), dtype=gdshp.dtype)
    for iN in prange(nNE):
        di = iN * __NDOFN__
        # u_s
        res[0, 0 + di] = shp[iN, 0]
        # v_s
        res[1, 1 + di] = shp[iN, 1]
        res[1, 5 + di] = shp[iN, 5]
        # w_s
        res[2, 2 + di] = shp[iN, 2]
        res[2, 4 + di] = shp[iN, 4]
        # \heta_x
        res[3, 3 + di] = shp[iN, 3]
        # \theta_y
        res[4, 2 + di] = -gdshp[iN, 2, 0]
        res[4, 4 + di] = -gdshp[iN, 4, 0]
        # \theta_z
        res[5, 1 + di] = gdshp[iN, 1, 0]
        res[5, 5 + di] = gdshp[iN, 5, 0]
    return res


@njit(nogil=True, parallel=True, cache=__cache__)
def shape_function_matrix_Bernoulli_bulk(shp: ndarray, gdshp: ndarray):
    """
    Returns the shape function matrix for several Bernoulli beams.

    The input contains evaluations of the shape functions and at least
    the first global derivatives of the shape functions.

    Parameters
    ----------
    shp: numpy.ndarray
        A float array of shape (nE, nP, nNE, nDOF=6), being the shape
        functions evaluated at a nP number of points of several cells.
    dshp: numpy.ndarray
        A float array of shape (nE, nP, nNE, nDOF=6, 3), being derivatives
        of shape functions evaluated at a nP number of points of several cells.

    Returns
    -------
    numpy.ndarray
        2d float array of shape (nE, nP, nDOF, nDOF * nNODE).

    See Also
    --------
    :func:`shape_function_matrix`
    """
    nE, nP, nNE = gdshp.shape[:3]
    res = np.zeros((nE, nP, __NDOFN__, nNE * __NDOFN__), dtype=gdshp.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            res[iE, iP] = shape_function_matrix_Bernoulli(shp[iE, iP], gdshp[iE, iP])
    return res


@njit(nogil=True, parallel=True, cache=__cache__)
def _shape_function_matrix_Bernoulli_L(shp: ndarray):
    """
    Returns the shape function matrix for a line.
    The input contains evaluations of the shape functions.

    shp (nNE, nDOF=6)
    ---
    (nDOF, nDOF * nNODE)
    """
    nNE = shp.shape[0]
    res = np.zeros((__NDOFN__, nNE * __NDOFN__), dtype=shp.dtype)
    for iN in prange(nNE):
        di = iN * __NDOFN__
        res[0, 0 + di] = shp[iN, 0]
        res[1, 1 + di] = shp[iN, 1]
        res[2, 2 + di] = shp[iN, 2]
        res[3, 3 + di] = shp[iN, 3]
        res[4, 2 + di] = shp[iN, 4]
        res[5, 1 + di] = shp[iN, 5]
    return res


@njit(nogil=True, parallel=True, cache=__cache__)
def _shape_function_matrix_L_multi(shp: ndarray):
    """
    Returns the shape function matrix for a line.
    The input contains evaluations of the shape functions.

    shp (nP, nNE, nDOF=6)
    ---
    (nP, nDOF, nDOF * nNODE)
    """
    nP, nNE = shp.shape[:2]
    res = np.zeros((nP, __NDOFN__, nNE * __NDOFN__), dtype=shp.dtype)
    for iP in prange(nP):
        res[iP] = _shape_function_matrix_Bernoulli_L(shp[iP])
    return res


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache__)
def body_load_vector_Bernoulli(
    values: ndarray, shp: ndarray, gdshp: ndarray, djac: ndarray, w: ndarray
):
    """
    Input values are assumed to be evaluated at multiple (nG) Gauss points of
    multiple (nE) cells.

    Parameters
    ----------
    values: numpy.ndarray
        A 3d float array of body loads of shape (nE, nRHS, nNE * 6)
        for several elements and load cases.
    shp: numpy.ndarray
        A float array of shape (nE, nG, nNE, nDOF=6), being the shape
        functions evaluated at a nG number of Gauss points of several cells.
    djac: numpy.ndarray
        A float array of shape (nE, nG), being jacobian determinants
        evaluated at the Gauss points of several cells.
    w: numpy.ndarray
        1d float array of weights for an nG number of Gauss points,
        with a shape of (nG,).

    Returns
    -------
    numpy.ndarray
        3d float array of shape (nE, nNE * 6, nRHS).
    """
    nRHS = values.shape[1]
    nE, nG, nNE = shp.shape[:3]
    NH = shape_function_matrix_Bernoulli_bulk(
        shp, gdshp
    )  # (nE, nG, nDOF, nDOF * nNODE)
    NL = _shape_function_matrix_L_multi(shp[0])  # (nG, nDOF, nDOF * nNODE)
    res = np.zeros((nE, nNE * __NDOFN__, nRHS), dtype=values.dtype)
    for iG in range(nG):
        for iRHS in prange(nRHS):
            for iE in prange(nE):
                res[iE, :, iRHS] += (
                    NH[iE, iG].T @ NL[iG] @ values[iE, iRHS, :] * djac[iE, iG] * w[iG]
                )
    return res


@njit(nogil=True, parallel=True, cache=__cache__)
def lumped_mass_matrices_direct_Bernoulli(
    dens: ndarray,
    lengths: ndarray,
    areas: ndarray,
    topo: ndarray,
    alpha: float = 1 / 20,
) -> ndarray:
    """
    Returns the diagonal values of the directly lumped mass matrix for several
    Bernoulli beam elements, sharing the total masses equivalently among nodes.

    Implementation is based on the lecture notes of Carlos A. Felippa [1].

    Parameters
    ----------
    dens : numpy.ndarray
        1d numpy float array of material densities.
    lengths : numpy.ndarray
        1d numpy float array of element lengths.
    areas : numpy.ndarray
        1d numpy float array of element areas.
    topo : numpy.ndarray
        2d numpy integer array of node indices referencing the global pointcloud.
    alpha : float, Optional
        A nonnegative parameter, typically between 0 and 1/50 (see notes).
        Default is 1/20.

    Returns
    -------
    numpy.ndarray
        A 2d float array of shape (nE, 6 * nNE), where nE and nNE are the number of
        elements and nodes per element.

    Notes
    -----
    [1] : "The choice of α has been argued in the FEM literature over several decades,
    but the whole discussion is largely futile. Matching the angular momentum of
    the beam element gyrating about its midpoint gives α = −1/24. This violates the
    positivity condition stated in [1, 32.2.4]. It follows that the best possible
    α — as opposed to possible best — is zero. This choice gives, however, a singular
    mass matrix, which is undesirable in scenarios where a mass-inverse appears."

    References
    ----------
    .. [1] Introduction to Finite Element Methods, Carlos A. Felippa.
           Department of Aerospace Engineering Sciences and Center for
           Aerospace Structures, University of Colorado. 2004.
    """
    nE, nNE = topo.shape
    diags = np.zeros((nE, 6 * nNE), dtype=dens.dtype)
    for iE in prange(nE):
        li = lengths[iE]
        vi = li * areas[iE]
        di = dens[iE]
        for jNE in prange(nNE):
            diags[iE, jNE * 6 : jNE * 6 + 3] = di * vi / nNE
            diags[iE, jNE * 6 + 3 : jNE * 6 + 6] = di * vi * alpha * li**2
    return diags
