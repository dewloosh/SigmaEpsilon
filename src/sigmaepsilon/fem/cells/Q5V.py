import numpy as np
from numpy import ndarray
from numba import njit, prange

from polymesh.cells import Q9 as Quadrilateral
from polymesh.utils.tri import area_tri, glob_to_nat_tri, loc_to_nat_tri
from polymesh.utils import nodal_distribution_factors

from .elem import FiniteElement
from ..material.membrane import Membrane
from ...utils.fem.fem import topo_to_gnum


__cache = True


__all__ = ["Q5MV"]


topoT2 = np.array(
    [[0, 1, 3, 4, 8, 7], [1, 2, 3, 5, 6, 8], [0, 1, 2, 4, 5, 8], [0, 2, 3, 8, 6, 7]]
)
ndfT2 = nodal_distribution_factors(topoT2, np.ones(4))


@njit(nogil=True, parallel=True, cache=__cache)
def dofmapT2():
    """Returns the indices of dofs in each subtriangle relative
    to the index array of active nodes `activeT2`."""
    topoT2a = np.array([[0, 4, 3], [1, 2, 4], [0, 1, 4], [4, 2, 3]])
    res = np.zeros((4, 6), dtype=np.int64)
    for i in prange(4):
        for j in prange(3):
            for k in prange(2):
                res[i, j * 2 + k] = topoT2a[i, j] * 2 + k
    return res


@njit(nogil=True, cache=__cache)
def Turner_to_Veubeke():
    """Transformation matrix of nodal variables in the
    direction Turner -> Veubeke."""
    return (
        np.array(
            [
                [1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1],
            ]
        )
        / 2
    )


@njit(nogil=True, cache=__cache)
def shp_Veubeke_a(acoord: np.ndarray):
    A1, A2, A3 = acoord
    return np.array([A1 + A2 - A3, -A1 + A2 + A3, A1 - A2 + A3], dtype=acoord.dtype)


@njit(nogil=True, cache=__cache)
def shp_Veubeke_g(gcoord: np.ndarray, ecoords: np.ndarray):
    return shp_Veubeke_a(glob_to_nat_tri(gcoord, ecoords))


@njit(nogil=True, cache=__cache)
def shp_Veubeke_p(pcoord: np.ndarray):
    return shp_Veubeke_a(loc_to_nat_tri(pcoord))


@njit(nogil=True, parallel=True, cache=__cache)
def nodal_approximation_matrix_V6():
    """Returns a matrix of nodal approximation coefficients.
    The coefficients in the ith row describe the approximation
    at node i as a linear combination of values at every node.
    At a compatible element, this is an identity matrix."""
    pcoords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
    )
    res = np.zeros((6, 6), dtype=pcoords.dtype)
    for i in prange(6):
        res[i, 3:] = shp_Veubeke_p(pcoords[i])
    return res


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def nodal_approximation_matrix_T2():
    """Returns a matrix of nodal approximation coefficients
    for the T2 variant, for the whole element."""
    nshpV6 = nodal_approximation_matrix_V6()
    res = np.zeros((9, 9), dtype=nshpV6.dtype)
    for iE in prange(4):
        for iN in prange(6):
            i = topoT2[iE, iN]
            ndf = ndfT2[iE, iN]
            for jN in prange(6):
                j = topoT2[iE, jN]
                res[i, j] += nshpV6[iN, jN] * ndf
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def nodal_approximation_matrix_T2_bulk(ndf: ndarray):
    """Returns a matrix of nodal approximation coefficients for all elements."""
    nE = ndf.shape[0]
    nappr = nodal_approximation_matrix_T2()
    res = np.zeros((nE, 9, 9), dtype=nappr.dtype)
    for iE in prange(nE):
        for i in prange(9):
            for j in prange(9):
                res[iE, i, j] = nappr[i, j] * ndf[iE, i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def approximation_matrix_T2_bulk(ndf: ndarray):
    """Returns a matrix of approximation coefficients for all elements."""
    nE = ndf.shape[0]
    nappr = nodal_approximation_matrix_T2()
    res = np.zeros((nE, 18, 18), dtype=nappr.dtype)
    for iE in prange(nE):
        for i in prange(9):
            for j in prange(9):
                for ii in prange(2):
                    for jj in prange(2):
                        res[iE, i * 2 + ii, j * 2 + jj] = nappr[i, j] * ndf[iE, i]
    return res


@njit(nogil=True, cache=__cache)
def _stiffness_matrix_V_(C: ndarray, ecoords: ndarray):
    (x1, x2, x3), (y1, y2, y3) = ecoords[:, 0], ecoords[:, 1]
    x12, x31, x23 = x1 - x2, x3 - x1, x2 - x3
    y21, y13, y32 = y2 - y1, y1 - y3, y3 - y2
    A = area_tri(ecoords)
    B = np.zeros((3, 6), dtype=C.dtype)
    B[0, :] = np.array([y21, 0, y32, 0, y13, 0], dtype=C.dtype)
    B[1, :] = np.array([0, x12, 0, x23, 0, x31], dtype=C.dtype)
    B[2, :] = np.array([x12, y21, x23, y32, x31, y13], dtype=C.dtype)
    return B.T @ C @ B / A


@njit(nogil=True, parallel=True, cache=__cache)
def stiffness_matrix_V(C: np.ndarray, ecoords: np.ndarray):
    nE = len(C)
    res = np.zeros((nE, 6, 6), dtype=C.dtype)
    for i in prange(nE):
        res[i, :, :] = _stiffness_matrix_V_(C[i], ecoords[i])
    return res


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def stiffness_matrix_T2(C: ndarray, ecoords: ndarray):
    nE = len(C)
    dmapT2 = dofmapT2()
    res = np.zeros((nE, 10, 10), dtype=C.dtype)
    C = C / 2
    for i in prange(4):
        Ki = stiffness_matrix_V(C, ecoords[:, topoT2[i, :3], :])
        for j in prange(6):
            for k in prange(6):
                res[:, dmapT2[i, j], dmapT2[i, k]] += Ki[:, j, k]
    return res


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def transfer_passive_loads_T2(cellloadsT2: ndarray):
    """(2) Transforms loads on passive nodes to active nodes
    according to the transformation described by `Turner_to_Veubeke`."""
    nE = cellloadsT2.shape[0]
    res = np.zeros_like(cellloadsT2)
    T_to_V = Turner_to_Veubeke()
    for iE in prange(nE):
        for iSE in prange(4):
            loads_Turner = np.ravel(cellloadsT2[iE, iSE, :3, :])
            loads_Veubeke = T_to_V @ loads_Turner
            loads_Veubeke = np.reshape(loads_Veubeke, (3, 2))
            for i in prange(3, 6):
                for j in prange(2):
                    res[iE, iSE, i, j] += cellloadsT2[iE, iSE, i, j]
                    res[iE, iSE, i, j] += loads_Veubeke[i - 3, j]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def distribute_nodal_data_T2(celldata: ndarray):
    """(1) Distributes master element data to subelement level."""
    nE, _, nD = celldata.shape
    res = np.zeros((nE, 4, 6, nD))
    for iE in prange(nE):
        for iSE in prange(4):
            for iSNE in prange(6):
                res[iE, iSE, iSNE] = celldata[iE, topoT2[iSE, iSNE]] * ndfT2[iSE, iSNE]
    return res


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def collect_nodal_dataT2(celldataT2: ndarray):
    """(3) Collects master element data from subelement level."""
    nE, _, _, nD = celldataT2.shape
    res = np.zeros((nE, 9, nD), dtype=celldataT2.dtype)
    for iE in prange(nE):
        for iSE in prange(4):
            for iNSE in prange(6):
                res[iE, topoT2[iSE, iNSE]] += celldataT2[iE, iSE, iNSE]
    return res


@njit(nogil=True, cache=__cache)
def nodal_data_T2(celldata: ndarray):
    """
    Handles loads defined on passive nodes in 3 steps:
    (1) Distribute master element data further to subelement level.
    (2) On every subelement, transforms passive loads to uniform
        body loads and integrates to active nodes.
    (3) Collects master element data from subelement level.
    """
    celldataT2 = distribute_nodal_data_T2(celldata)
    celldataT2 = transfer_passive_loads_T2(celldataT2)
    return collect_nodal_dataT2(celldataT2)


class Q5MV(Quadrilateral, Membrane, FiniteElement):
    def stiffness_matrix(self, *args, topo=None, **kwargs):
        topo = self.nodes.to_numpy() if topo is None else topo
        ecoords = self.local_coordinates(topo=topo)
        C = self.model_stiffness_matrix()
        return stiffness_matrix_T2(C, ecoords)

    def global_dof_numbering(self, *args, topo=None, **kwargs):
        topo = self.nodes.to_numpy() if topo is None else topo
        return topo_to_gnum(topo[:, 4:], self.NDOFN)

    def distribute_nodal_data(self, data, key):
        super().distribute_nodal_data(data, key)
        if key == "loads":
            celldata = self._wrapped["loads"].to_numpy()
            self._wrapped["loads"] = nodal_data_T2(celldata)

    def approximation_matrix(self, *args, **kwargs):
        return approximation_matrix_T2_bulk(self.ndf.to_numpy())

    def nodal_approximation_matrix(self, *args, **kwargs):
        return nodal_approximation_matrix_T2_bulk(self.ndf.to_numpy())
