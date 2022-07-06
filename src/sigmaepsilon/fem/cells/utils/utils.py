# -*- coding: utf-8 -*-
from typing import Callable
from numba import njit, prange
import numpy as np
from numpy import ndarray

from dewloosh.math.linalg import inv

from dewloosh.mesh.utils import cells_coords

__cache = True


@njit(nogil=True, cache=__cache)
def element_coordinates(coords: ndarray, topo: np.ndarray):
    return cells_coords(coords, topo)


def element_centers(ecoords: np.ndarray):
    return np.mean(ecoords, axis=1)


@njit(nogil=True, cache=__cache)
def loc_to_glob(pcoord: np.ndarray, gcoords: np.ndarray, shpfnc: Callable):
    """Local to global transformation for a single cell and point.

    Returns global coordinates of a point in an element, provided the global
    corodinates of the points of the element, an array of parametric
    coordinates and a function to evaluate the shape functions.

    Parameters
    ----------
    gcoords : (nNE, nD) ndarray
        2D array containing coordinates for every node of a single element.
            nNE : number of vertices of the element
            nD : number of dimensions of the model space

    pcoords : (nDP, ) ndarray
        1D array of parametric coordinates for a single point.
            nDP : number of dimensions of the parametric space

    shpfnc : Callable
        A function that evaluates shape function values at a point,
        specified with parametric coordinates.

    Returns
    -------
    (nD, ) ndarray
        Global cooridnates of the specified point.

    Notes
    -----
    'shpfnc' must be a numba-jitted function, that accepts a 1D array of
    exactly nDP number of components.
    """
    return gcoords.T @ shpfnc(pcoord)


@njit(nogil=True, parallel=True, cache=__cache)
def loc_to_glob_bulk(pcoords: np.ndarray, gcoords: np.ndarray,
                     shpfnc: Callable):
    """Local to global transformation for multiple cells and points.

    Returns global coordinates of points expressed in parametric coordinates.

    Parameters
    ----------
    gcoords : (nE, nNE, nD) ndarray
        3D array containing coordinates for every node of every element.
            nE : number of elements
            nNE : number of vertices of the element
            nD : number of dimensions of the model space

    pcoords : (nP, nDP) ndarray
        2D array of parametric coordinates for several points.
            nP : number of points
            nDP : number of dimensions of the parametric space

    shpfnc : Callable
        A function that evaluates shape function values at a point,
        specified with parametric coordinates.

    Returns
    -------
    (nE, nP, nD) ndarray
        Global cooridnates of the specified points in the specified elements.
        Dummy axes are collapsed, so the result may be 1, 2 or 3 dimensional,
        depending on the shapes of the input arrays.

    Notes
    -----
    'shpfnc' must be a numba-jitted function, that accepts a 1D array of
    exactly nDP number of components.
    """
    nE, nNE, nD = gcoords.shape
    nP, _ = pcoords.shape
    res = np.zeros((nE, nP, nD), dtype=gcoords.dtype)
    for iP in prange(nP):
        shp = shpfnc(pcoords[iP])
        for iE in prange(nE):
            for iD in prange(nD):
                for iNE in prange(nNE):
                    res[iE, iP, iD] += gcoords[iE, iNE, iD] * shp[iNE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def glob_to_loc(coord: np.ndarray, gcoords: np.ndarray, lcoords: np.ndarray,
                monomsfnc: Callable):
    """Global to local transformation for a single point and cell.

    Returns local coordinates of a point in an element, provided the global
    corodinates of the points of the element, an array of global
    coordinates and a function to evaluate the monomials of the shape
    functions.

    Parameters
    ----------
    gcoords : (nNE, nD) ndarray
        2D array containing coordinates for every node of a single element.
            nNE : number of vertices of the element
            nD : number of dimensions of the model space

    coord : (nD, ) ndarray
        1D array of global coordinates for a single point.
            nD : number of dimensions of the model space

    lcoords : (nNE, nDP) ndarray
        2D array of local coordinates of the parametric element.
            nNE : number of vertices of the element
            nDP : number of dimensions of the parametric space

    monomsfnc : Callable
        A function that evaluates monomials of the shape functions at a point
        specified with parametric coordinates.

    Returns
    -------
    (nDP, ) ndarray
        Parametric cooridnates of the specified point.

    Notes
    -----
    'shpfnc' must be a numba-jitted function, that accepts a 1D array of
    exactly nDP number of components.
    """
    nNE = gcoords.shape[0]
    monoms = np.zeros((nNE, nNE), dtype=coord.dtype)
    for i in prange(nNE):
        monoms[:, i] = monomsfnc(gcoords[i])
    coeffs = inv(monoms)
    shp = coeffs @ monomsfnc(coord)
    return lcoords.T @ shp


@njit(nogil=True, parallel=True, cache=__cache)
def glob_to_loc_bulk(coords: np.ndarray, gcoords: np.ndarray,
                     lcoords: np.ndarray, monomsfnc: Callable):
    """Global to local transformation for multiple cells and points.

    Returns local coordinates of a point in an element, provided the global
    corodinates of the points of the element, an array of global
    coordinates and a function to evaluate the monomials of the shape
    functions.

    Parameters
    ----------
    gcoords : (nE, nNE, nD) ndarray
        3D array containing coordinates for every node of several elements.
            nE : number of elements
            nNE : number of vertices of the element
            nD : number of dimensions of the model space

    coords : (nP, nD) ndarray
        2D array of global coordinates for several points.
            nP : number of points
            nD : number of dimensions of the model space

    lcoords : (nNE, nDP) ndarray
        2D array of local coordinates of the parametric element.
            nNE : number of vertices of the element
            nDP : number of dimensions of the parametric space

    monomsfnc : Callable
        A function that evaluates monomials of the shape functions at a point
        specified with parametric coordinates.

    Returns
    -------
    (nP, nE, nDP) ndarray
        Parametric cooridnates of the specified points in all elements.

    Notes
    -----
    'shpfnc' must be a numba-jitted function, that accepts a 1D array of
    exactly nDP number of components, where nDP is the number
    of paramatric cooridnate dimensions.
    """
    nE, nNE, _ = gcoords.shape
    nP = len(coords)
    nDP = lcoords.shape[1]
    lcoords = lcoords.T
    res = np.zeros((nP, nE, nDP), dtype=gcoords.dtype)
    monoms = np.zeros((nNE, nNE), dtype=gcoords.dtype)
    shp = np.zeros(nNE, dtype=gcoords.dtype)
    coeffs = np.zeros_like(monoms)
    for iE in prange(nE):
        for iNE in prange(nNE):
            monoms[:, iNE] = monomsfnc(gcoords[iE, iNE])
        coeffs[:, :] = inv(monoms)
        for iP in prange(nP):
            shp[:] = coeffs @ monomsfnc(coords[iP])
            res[iP, iE, :] = lcoords @ shp
    return res


@njit(nogil=True, cache=__cache)
def pip(coord: np.ndarray, gcoords: np.ndarray, lcoords: np.ndarray,
        monomsfnc: Callable, shpfnc: Callable, tol=1e-12):
    """Point-in-polygon test for a single cell and point.

    Performs the point-in-poligon test for a single point and cell.
    False means that the point is outside of the cell, True means the point
    is either inside, or on the boundary of the cell.

    Parameters
    ----------
    gcoords : (nNE, nD) ndarray
        2D array containing coordinates for every node of a single element.
            nNE : number of vertices of the element
            nD : number of dimensions of the model space

    coord : (nD, ) ndarray
        1D array of global coordinates for a single point.
            nD : number of dimensions of the model space

    lcoords : (nNE, nDP) ndarray
        2D array of local coordinates of the parametric element.
            nNE : number of vertices of the element
            nDP : number of dimensions of the parametric space

    monomsfnc : Callable
        A function that evaluates monomials of the shape functions at a point
        specified with parametric coordinates.

    shpfnc : Callable
        A function that evaluates shape function values at a point,
        specified with parametric coordinates.

    Returns
    -------
    bool
        False if points is outside of the cell, True otherwise.

    Notes
    -----
    'shpfnc'  and 'monomsfnc' must be numba-jitted functions, that accept
    a 1D array of exactly nDP number of components, where nDP is the number
    of paramatric cooridnate dimensions.
    """
    limit = 1 + tol
    loc = glob_to_loc(coord, gcoords, lcoords, monomsfnc)
    return np.all(shpfnc(loc) <= limit)


@njit(nogil=True, parallel=True, cache=__cache)
def int_domain(points: np.ndarray, qpos: np.ndarray, qweight: np.ndarray,
               dshpfnc: Callable, weight: float = 1.0):
    """
    Returns the measure (length, area or volume in 1d, 2d and 3d) of a 
    single domain.
    """
    volume = 0.
    points = np.transpose(points)
    nG = len(qweight)
    for iG in prange(nG):
        jac = points @ dshpfnc(qpos[iG])
        djac = np.linalg.det(jac)
        volume += qweight[iG] * djac
    return volume * weight


@njit(nogil=True, parallel=True, cache=__cache)
def int_domains(ecoords: np.ndarray, qpos: np.ndarray,
                qweight: np.ndarray, dshpfnc: Callable):
    """
    Returns the measure (length, area or volume in 1d, 2d and 3d) of 
    several domains.
    """
    nE = ecoords.shape[0]
    res = np.zeros(nE, dtype=ecoords.dtype)
    nG = len(qweight)
    for iG in prange(nG):
        dshp = dshpfnc(qpos[iG])
        for i in prange(nE):
            jac = ecoords[i].T @ dshp
            djac = np.linalg.det(jac)
            res[i] += qweight[iG] * djac
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def stiffness_matrix(C: np.ndarray, ecoords: np.ndarray, qpos: np.ndarray,
                     qweight: np.ndarray, dshpfnc: Callable,
                     sdmfnc: Callable, nDOFN=3):
    nTOTV = len(ecoords) * nDOFN
    res = np.zeros((nTOTV, nTOTV), dtype=C.dtype)
    points = np.transpose(ecoords)
    nG = len(qweight)
    for iG in prange(nG):
        dshp = dshpfnc(qpos[iG])
        jac = points @ dshp
        djac = np.linalg.det(jac) * qweight[iG]
        B = sdmfnc(jac, dshp)
        res += B.T @ C @ B * djac
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def stiffness_matrix_bulk(C: np.ndarray, ecoords: np.ndarray,
                          qpos: np.ndarray, qweight: np.ndarray,
                          dshpfnc: Callable, sdmfnc: Callable, nDOFN=3):
    nE, nNODE, _ = ecoords.shape
    nTOTV = nNODE * nDOFN
    res = np.zeros((nE, nTOTV, nTOTV), dtype=C.dtype)
    nG = len(qweight)
    for iG in prange(nG):
        dshp = dshpfnc(qpos[iG])
        for iE in prange(nE):
            jac = np.transpose(ecoords[iE]) @ dshp
            djac = np.linalg.det(jac) * qweight[iG]
            B = sdmfnc(jac, dshp)
            res[iE] += B.T @ C[iE] @ B * djac
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def strain_displacement_matrix_bulk2(B: ndarray, djac: ndarray, w: ndarray):
    nE, nG, nSTRE, nTOTV = B.shape
    res = np.zeros((nE, nSTRE, nTOTV), dtype=B.dtype)
    for iG in range(nG):
        for iE in prange(nE):
            res[iE] += B[iE, iG] * djac[iE, iG] * w[iG]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def unit_strain_load_vector_bulk(D: ndarray, B: ndarray):
    nE, nSTRE, nTOTV = B.shape
    res = np.zeros((nE, nTOTV, nSTRE), dtype=D.dtype)
    for iE in prange(nE):
        res[iE] = B[iE].T @ D[iE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def strain_load_vector_bulk(BTD: ndarray, strains: ndarray):
    nE, nTOTV, nSTRE = BTD.shape
    nE, nRHS, nSTRE = strains.shape
    res = np.zeros((nE, nTOTV, nRHS), dtype=BTD.dtype)
    for iE in prange(nE):
        for iRHS in prange(nRHS):
            res[iE, :, iRHS] = BTD[iE] @ strains[iE, iRHS, :]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def stiffness_matrix_bulk2(D: ndarray, B: ndarray, djac: ndarray, w: ndarray):
    nE, nG = djac.shape
    nTOTV = B.shape[-1]
    res = np.zeros((nE, nTOTV, nTOTV), dtype=D.dtype)
    for iG in range(nG):
        for iE in prange(nE):
            res[iE] += B[iE, iG].T @ D[iE] @ B[iE, iG] * djac[iE, iG] * w[iG]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def mass_matrix_bulk(N: ndarray, dens: ndarray, weights: ndarray,
                     djac: ndarray, w: ndarray):
    """
    Evaluates the mass matrix of several elements from evaluations at the
    Gauss points of the elements. The densities are assumed to be constant
    over the elements, and be represented by the 1d float numpy array 'dens'.

    Parameters
    ----------
    N : 4d numpy float array
        Evaluations of the shape function matrix according to the type of 
        the element for every cell and Gauss point.

    dens : 1d numpy float array
        1d float array of densities of several elements.

    weights : 1d numpy float array
        Array of weighting factors for the cells (eg. areas, volumes) with 
        the same shape as 'dens'. 

    w : 1d numpy float array
        Weights of the Gauss points

    djac : 2d numpy float array
        Jacobi determinants for every element and Gauss point.

    nE : number of elements
    nG : number of Gauss points
    nNE : number of nodes per element
    nDOF : number of dofs per node

    """
    nE, nG = djac.shape
    nTOTV = N.shape[-1]
    res = np.zeros((nE, nTOTV, nTOTV), dtype=N.dtype)
    for iG in range(nG):
        for iE in prange(nE):
            res[iE] += dens[iE] * weights[iE] * \
                N[iE, iG].T @ N[iE, iG] * djac[iE, iG] * w[iG]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def compliance_bulk(K_bulk: np.ndarray, dofsol1d: np.ndarray,
                    gnum: np.ndarray):
    """
    Returns the total compliance of the structure, that is
        u.T @ K @ u
    """
    nE, nTOTV, _ = K_bulk.shape
    res = np.zeros(nE, dtype=K_bulk.dtype)
    for iE in prange(nE):
        for i in prange(nTOTV):
            for j in prange(nTOTV):
                res[iE] += K_bulk[iE, i, j] * \
                    dofsol1d[gnum[iE, i]] * dofsol1d[gnum[iE, j]]
    return res


@njit(nogil=True, cache=__cache)
def elastic_strain_energy_bulk(K_bulk: np.ndarray, dofsol1d: np.ndarray,
                               gnum: np.ndarray):
    """
    Returns the total elastic strain energy of the structure, that is 
        0.5 * u.T @ K @ u
    """
    return compliance_bulk(K_bulk, dofsol1d, gnum) / 2


@njit(nogil=True, parallel=True, cache=__cache)
def approx_dof_solution(dofsol1d: np.ndarray, gnum: np.ndarray,
                        shpmat: np.ndarray):
    nDOFN, nTOTV = shpmat.shape
    res = np.zeros((nDOFN), dtype=dofsol1d.dtype)
    for j in prange(nDOFN):
        for k in prange(nTOTV):
            res[j] += shpmat[j, k] * dofsol1d[gnum[k]]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def approx_dof_solution_bulk(dofsol1d: np.ndarray, gnum: np.ndarray,
                             shpmat: np.ndarray):
    nE = len(gnum)
    nDOFN, nTOTV = shpmat.shape
    res = np.zeros((nE, nDOFN), dtype=dofsol1d.dtype)
    for i in prange(nE):
        for j in prange(nDOFN):
            for k in prange(nTOTV):
                res[i, j] += shpmat[j, k] * dofsol1d[gnum[i, k]]
    return res


@njit(nogil=True, cache=__cache)
def element_dof_solution(dofsol1d: np.ndarray, gnum: np.ndarray):
    return dofsol1d[gnum]


@njit(nogil=True, parallel=True, cache=__cache)
def element_dof_solution_bulk(dofsol1d: ndarray, gnum: ndarray):
    nE, nEVAB = gnum.shape
    res = np.zeros((nE, nEVAB), dtype=dofsol1d.dtype)
    for i in prange(nE):
        res[i, :] = dofsol1d[gnum[i, :]]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def avg_nodal_data_bulk(data: np.ndarray, topo: np.ndarray):
    nE, nNE, nD = data.shape
    nP = np.max(topo) + 1
    res = np.zeros((nP, nD), dtype=data.dtype)
    count = np.zeros((nP), dtype=topo.dtype)
    for iE in prange(nE):
        for jNE in prange(nNE):
            res[topo[iE, jNE]] += data[iE, jNE]
            count[topo[iE, jNE]] += 1
    for iP in prange(nP):
        res[iP] = res[iP] / count[iP]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def stresses_at_point_bulk(C: np.ndarray, dofsol1d: np.ndarray,
                           ecoords: np.ndarray, gnum: np.ndarray,
                           dshp: np.ndarray, sdmfnc: Callable):
    nE = len(gnum)  # number of elements
    nSTRE = C.shape[-1]  # number of generalized forces
    esol = element_dof_solution_bulk(dofsol1d, gnum)
    res = np.zeros((nE, nSTRE), dtype=C.dtype)
    for i in prange(nE):
        jac = ecoords[i].T @ dshp
        B = sdmfnc(jac, dshp)
        res[i] = C[i] @ B @ esol[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def stresses_at_cells_nodes(C: np.ndarray, dofsol1d: np.ndarray,
                            ecoords: np.ndarray, gnum: np.ndarray,
                            dshp: np.ndarray, sdmfnc: Callable):
    nE, nNODE, _ = ecoords.shape  # numbers of elements and element nodes
    nSTRE = C.shape[-1]  # number of generalized forces
    res = np.zeros((nE, nNODE, nSTRE), dtype=C.dtype)
    for i in prange(nNODE):
        res[:, i, :] = stresses_at_point_bulk(C, dofsol1d, ecoords, gnum,
                                              dshp[i], sdmfnc)
    return res


if __name__ == '__main__':
    pass
