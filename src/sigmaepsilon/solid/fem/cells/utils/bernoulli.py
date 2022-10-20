# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit, prange

__all__ = []

__cache__ = True
__NDOFN__ = 6


@njit(nogil=True, parallel=True, cache=__cache__)
def global_shape_function_derivatives_bulk(dshp: ndarray, jac: ndarray):
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
        5d float array of shape (nE, nP, nNE, nDOF=6, 3)

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
def shape_function_matrix(shp: ndarray, gdshp: ndarray):
    """
    Returns the shape function matrix for a Bernoulli beam.

    The input contains evaluations of the shape functions and at lest 
    the first global derivatives of the shape functions. 

    shp (nNE, nDOF=6)
    gdshp (nNE, nDOF=6, 3)
    ---
    (nDOF, nDOF * nNODE)

    Notes
    -----
    The approximation applies a mixture of Lagrange(L) and Hermite(H) 
    polynomials, in the order [L, H, H, L, H, H].
    
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
        res[4, 2 + di] = - gdshp[iN, 2, 0]
        res[4, 4 + di] = - gdshp[iN, 4, 0]
        # \theta_z
        res[5, 1 + di] = gdshp[iN, 1, 0]
        res[5, 5 + di] = gdshp[iN, 5, 0]
    return res


@njit(nogil=True, parallel=True, cache=__cache__)
def shape_function_matrix_bulk(shp: ndarray, gdshp: ndarray):
    """
    In
    --
    shp (nE, nP, nNE, nDOF=6)
    gdshp (nE, nP, nNE, nDOF=6, 3)

    Out
    ---
    (nE, nP, nDOF, nDOF * nNODE)
    """
    nE, nP, nNE = gdshp.shape[:3]
    res = np.zeros((nE, nP, __NDOFN__, nNE * __NDOFN__), dtype=gdshp.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            res[iE, iP] = shape_function_matrix(shp[iE, iP], gdshp[iE, iP])
    return res


@njit(nogil=True, parallel=True, cache=__cache__)
def shape_function_matrix_L(shp: ndarray):
    """
    Returns the shape function matrix for a line.

    The input contains evaluations of the shape functions. 

    In
    --
    shp (nNE, nDOF=6)

    Out
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
def shape_function_matrix_L_multi(shp: ndarray):
    """
    Returns the shape function matrix for a line.

    The input contains evaluations of the shape functions. 

    In
    --
    shp (nP, nNE, nDOF=6)

    Out
    ---
    (nP, nDOF, nDOF * nNODE)
    """
    nP, nNE = shp.shape[:2]
    res = np.zeros((nP, __NDOFN__, nNE * __NDOFN__), dtype=shp.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_matrix_L(shp[iP])
    return res


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache__)
def body_load_vector_bulk(values: ndarray, shp: ndarray, gdshp: ndarray,
                          djac: ndarray, w: ndarray):
    """
    Input values are assumed to be evaluated at multiple (nG) Gauss points of
    multiple (nE) cells.

    values (nE, nRHS, nNE * 6)
    djac (nE, nG)
    shp (nE, nG, nNE=2, nDOF=6)
    w (nG)
    ---
    (nE, nNE * 6, nRHS)
    """
    nRHS = values.shape[1]
    nE, nG, nNE = shp.shape[:3]
    NH = shape_function_matrix_bulk(shp, gdshp)  # (nE, nG, nDOF, nDOF * nNODE)
    NL = shape_function_matrix_L_multi(shp[0])  # (nG, nDOF, nDOF * nNODE)
    res = np.zeros((nE, nNE * __NDOFN__, nRHS), dtype=values.dtype)
    for iG in range(nG):
        for iRHS in prange(nRHS):
            for iE in prange(nE):
                res[iE, :, iRHS] += NH[iE, iG].T @ NL[iG] @ values[iE,
                                                                   iRHS, :] * djac[iE, iG] * w[iG]
    return res


@njit(nogil=True, parallel=True, cache=__cache__)
def calculate_element_forces_bulk(dofsol: ndarray, B: ndarray,
                                  D: ndarray, shp: ndarray,
                                  gdshp: ndarray, body_forces: ndarray):
    """
    Calculates internal forces from dof solution.
    """
    nE = dofsol.shape[0]
    nP = B.shape[1]
    res = np.zeros((__NDOFN__, nE, nP), dtype=dofsol.dtype)
    for i in prange(nE):
        for j in prange(nP):
            pyy, pzz = body_forces[i, 0, 4:] * shp[j, 0] + \
                body_forces[i, 1, 4:] * shp[j, 1]
            N, T, My, Mz = D[i] @ B[i, j] @ dofsol[i]
            res[0, i, j] = N
            # Vy
            res[1, i, j] = -D[i, 3, 3] * (
                gdshp[i, j, 2, 2] * dofsol[i, 1] +
                gdshp[i, j, 3, 2] * dofsol[i, 5] +
                gdshp[i, j, 4, 2] * dofsol[i, 7] +
                gdshp[i, j, 5, 2] * dofsol[i, 11]) - pzz
            # Vz
            res[2, i, j] = -D[i, 2, 2] * (
                gdshp[i, j, 6, 2] * dofsol[i, 2] +
                gdshp[i, j, 7, 2] * dofsol[i, 4] +
                gdshp[i, j, 8, 2] * dofsol[i, 8] +
                gdshp[i, j, 9, 2] * dofsol[i, 10]) + pyy
            res[3, i, j] = T
            res[4, i, j] = My
            res[5, i, j] = Mz
    return res


@njit(nogil=True, parallel=True, cache=__cache__)
def interpolate_element_data_bulk(edata: ndarray, N: ndarray):
    """
    N (nE, nP, nDOF, nDOF * nNODE)  shape function matrix
    edata (nE, nDOF * nNODE)  element data
    """
    nE, nP = N.shape[:2]
    res = np.zeros(N.shape[:3], dtype=N.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            res[iE, iP, :] = N[iE, iP] @ edata[iE]
    return res


@njit(nogil=True, parallel=True, cache=__cache__)
def lumped_mass_matrices_direct(dens: ndarray, lengths: ndarray, areas: ndarray,
                                topo: ndarray, alpha: float = 1/20) -> ndarray:
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
        A 3d float array of shape (nE, 6 * nNE), where nE and nNE are the number of
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
            diags[iE, jNE*6: jNE*6 + 3] = di * vi / nNE
            diags[iE, jNE*6 + 3: jNE*6 + 6] = di * vi * alpha * li**2
    return diags
