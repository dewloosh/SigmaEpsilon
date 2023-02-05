"""
The ANDES template.
"""
import numpy as np
from numpy import ndarray
from numba import njit, prange

from neumann.utils import repeat1d
from polymesh.cells import T3 as Triangle
from polymesh.utils.tri import lcoords_tri, shp_tri_loc

from ..material.membrane import DrillingMembrane

from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC

__cache = True


__all__ = ["LSTMR"]


def sig_ANDES(key: str = "OPT", C: ndarray = None):
    if key == "OPT":
        # OPTIMAL ANDES TEMPLATE
        sig = sig_ANDES_opt(C)
    elif key == "ALL88-3I":
        # Allman 88 element integrated by 3-point interior rule
        sig = (
            1,
            4 / 9,
            1 / 12,
            5 / 12,
            1 / 2,
            0,
            1 / 3,
            -1 / 3,
            -1 / 12,
            -1 / 2,
            -5 / 12,
        )
    elif key == "ALL88-3M":
        # Allman 88 element integrated by 3 midpoint rule
        sig = (1, 4 / 9, 1 / 4, 5 / 4, 3 / 2, 0, 1, -1, -1 / 4, -3 / 2, -5 / 4)
    elif key == "ALL88-LS":
        # Allman 88 element least-square strain fit
        sig = (
            1,
            4 / 9,
            3 / 20,
            3 / 4,
            9 / 10,
            0,
            3 / 5,
            -3 / 5,
            -3 / 20,
            -9 / 10,
            -3 / 4,
        )
    elif key == "LST-Ret":
        # Retrofitted LST with alpha=4/3
        sig = (4 / 3, 1 / 2, 2 / 3, -2 / 3, 0, 0, -4 / 3, 4 / 3, -2 / 3, 0, 2 / 3)
    elif key == "CST":
        # Constant strain triangle CST-3/6C
        sig = (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    else:
        raise ValueError("Invalid signature.")
    return np.array(sig).astype(float)


@njit(nogil=True, cache=__cache)
def sig_ANDES_opt(C: ndarray):
    E11, E12, E13, _, E22, E23, _, _, E33 = C.flatten()
    Edet = (
        E11 * E22 * E33
        + 2 * E12 * E13 * E23
        - E11 * E23**2
        - E22 * E13**2
        - E33 * E12**2
    )
    W = (
        -6 * E12**3
        + 5 * E11**2 * E22
        - 5 * E12**2 * E22
        - E22 * (75 * E13**2 + 14 * E13 * E23 + 3 * E23**2)
        + 2 * E12 * (7 * E13**2 + 46 * E13 * E23 + 7 * E23**2)
        - E11
        * (
            5 * E12**2
            + 3 * E13**2
            - 6 * E12 * E22
            - 5 * E22**2
            + 14 * E13 * E23
            + 75 * E23**2
        )
        + (
            3 * E11**2
            + 82 * E11 * E22
            + 3 * E22**2
            - 4 * (6 * E12**2 + 5 * E13**2 - 6 * E13 * E23 + 5 * E23**2)
        )
        * E33
        + 4 * (5 * E11 - 6 * E12 + 5 * E22) * E33**2
    )
    E11C11avg = W / (128 * Edet)
    b0 = max([2 / E11C11avg - 3 / 2, 1 / 100])
    return np.array([3 / 2, b0, 1, 2, 1, 0, 1, -1, -1, -1, -2], dtype=C.dtype)


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def sig_ANDES_opt_bulk(C: ndarray):
    nE = len(C)
    res = np.zeros((nE, 11), dtype=C.dtype)
    for i in prange(nE):
        res[i] = sig_ANDES_opt(C[i])
    return res


@njit(nogil=True, cache=__cache)
def _stiffness_data_(ecoords, h, sig):
    (x1, x2, x3), (y1, y2, y3) = ecoords[:, 0], ecoords[:, 1]
    ab, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9 = sig
    x12, x21, x13, x31, x23, x32 = x1 - x2, x2 - x1, x1 - x3, x3 - x1, x2 - x3, x3 - x2
    y12, y21, y13, y31, y23, y32 = y1 - y2, y2 - y1, y1 - y3, y3 - y1, y2 - y3, y3 - y2
    LL21 = x21**2 + y21**2
    LL32 = x32**2 + y32**2
    LL13 = x13**2 + y13**2
    A = (y21 * x13 - x21 * y13) / 2
    A2, A4 = 2 * A, 4 * A
    L = (
        np.array(
            [
                [y23, 0, x32],
                [0, x32, y23],
                [
                    ab * y23 * (y13 - y21) / 6,
                    ab * x32 * (x31 - x12) / 6,
                    ab * (x31 * y13 - x12 * y21) / 3,
                ],
                [y31, 0, x13],
                [0, x13, y31],
                [
                    ab * y31 * (y21 - y32) / 6,
                    ab * x13 * (x12 - x23) / 6,
                    ab * (x12 * y21 - x23 * y32) / 3,
                ],
                [y12, 0, x21],
                [0, x21, y12],
                [
                    ab * y12 * (y32 - y13) / 6,
                    ab * x21 * (x23 - x31) / 6,
                    ab * (x23 * y32 - x31 * y13) / 3,
                ],
            ],
            dtype=ecoords.dtype,
        )
        * h
        / 2
    )
    Tru = (
        np.array(
            [
                [x32, y32, A4, x13, y13, 0, x21, y21, 0],
                [x32, y32, 0, x13, y13, A4, x21, y21, 0],
                [x32, y32, 0, x13, y13, 0, x21, y21, A4],
            ],
            dtype=ecoords.dtype,
        )
        / A4
    )
    Te = np.array(
        [
            [y23 * y13 * LL21, y31 * y21 * LL32, y12 * y32 * LL13],
            [x23 * x13 * LL21, x31 * x21 * LL32, x12 * x32 * LL13],
            [
                (y23 * x31 + x32 * y13) * LL21,
                (y31 * x12 + x13 * y21) * LL32,
                (y12 * x23 + x21 * y32) * LL13,
            ],
        ],
        dtype=ecoords.dtype,
    ) / (4 * A**2)
    Q1 = (
        np.array(
            [
                [b1 / LL21, b2 / LL21, b3 / LL21],
                [b4 / LL32, b5 / LL32, b6 / LL32],
                [b7 / LL13, b8 / LL13, b9 / LL13],
            ],
            dtype=ecoords.dtype,
        )
        * A2
        / 3
    )
    Q2 = (
        np.array(
            [
                [b9 / LL21, b7 / LL21, b8 / LL21],
                [b3 / LL32, b1 / LL32, b2 / LL32],
                [b6 / LL13, b4 / LL13, b5 / LL13],
            ],
            dtype=ecoords.dtype,
        )
        * A2
        / 3
    )
    Q3 = (
        np.array(
            [
                [b5 / LL21, b6 / LL21, b4 / LL21],
                [b8 / LL32, b9 / LL32, b7 / LL32],
                [b2 / LL13, b3 / LL13, b1 / LL13],
            ],
            dtype=ecoords.dtype,
        )
        * A2
        / 3
    )
    return A, L, Te, b0, Q1, Q2, Q3, Tru


@njit(nogil=True, cache=__cache)
def _stiffness_matrix_LSTMR(C, ecoords, h, sig):
    A, L, Te, b0, Q1, Q2, Q3, Tru = _stiffness_data_(ecoords, h, sig)
    V = h * A
    Q4, Q5, Q6 = (Q1 + Q2) / 2, (Q3 + Q2) / 2, (Q1 + Q3) / 2
    Kb = (L @ C @ L.T) / V
    Cn = Te.T @ (C @ Te)
    Kr = (3 * b0 * V / 4) * (Q4.T @ Cn @ Q4 + Q5.T @ Cn @ Q5 + Q6.T @ Cn @ Q6)
    Kh = Tru.T @ Kr @ Tru
    return Kb, Kh


@njit(nogil=True, cache=__cache)
def __stiffness_matrix__(C, ecoords, h, sig):
    Kb, Kh = _stiffness_matrix_LSTMR(C, ecoords, h, sig)
    return Kb + Kh


@njit(nogil=True, parallel=True, cache=__cache)
def stiffness_matrix(C, ecoords, h, sig):
    nE = len(C)
    res = np.zeros((nE, 9, 9), dtype=C.dtype)
    for i in prange(nE):
        res[i, :, :] = __stiffness_matrix__(C[i], ecoords[i], h[i], sig[i])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def stiffness_matrix_LSQ_4_12(
    C: ndarray,
    ecoords: ndarray,
    h: ndarray,
    sig: ndarray,
    dmaps: ndarray,
    subtopo: ndarray,
):
    nE = len(C)
    dmax = dmaps.max() + 1
    res = np.zeros((nE, dmax, dmax), dtype=C.dtype)
    for i in range(subtopo.shape[0]):
        K_i_bulk = stiffness_matrix(C, ecoords[:, subtopo[i], :], h, sig)
        for j in prange(K_i_bulk.shape[1]):
            for k in prange(K_i_bulk.shape[2]):
                res[:, dmaps[i, j], dmaps[i, k]] += K_i_bulk[:, j, k]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def approx_strains(pcoord, ecoords, h, sig, dofsol1d, gnum):
    nE = len(gnum)
    res = np.zeros((nE, 3), dtype=ecoords.dtype)
    A1, A2, A3 = shp_tri_loc(pcoord)
    for i in prange(nE):
        A, L, Te, _, Q1, Q2, Q3, Tru = _stiffness_data_(ecoords[i], h[i], sig[i])
        uE = dofsol1d[gnum]
        res[i, :] = (1 / (h[i] * A)) * (L.T @ uE) + 3 * (
            Te @ (A1 * Q1 + A2 * Q2 + A3 * Q3) @ Tru @ uE
        ) / 2
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def strain_displacement_matrix(pcoord, ecoords, h, sig):
    nE = ecoords.shape[0]
    res = np.zeros((nE, 3, 9), dtype=ecoords.dtype)
    A1, A2, A3 = shp_tri_loc(pcoord)
    for i in prange(nE):
        A, L, Te, _, Q1, Q2, Q3, Tru = _stiffness_data_(ecoords[i], h[i], sig[i])
        res[i, :] = (1 / (h[i] * A)) * L.T + 3 * (
            Te @ (A1 * Q1 + A2 * Q2 + A3 * Q3) @ Tru
        ) / 2
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def approx_stresses(pcoord, C, ecoords, h, sig, dofsol1d, gnum):
    e = approx_strains(pcoord, ecoords, h, sig, dofsol1d, gnum)
    nE = len(e)
    res = np.zeros((nE, 3), dtype=C.dtype)
    for i in prange(nE):
        res[i, :] = C[i] @ e[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def stresses_at_nodes(C, ecoords, h, sig, dofsol1d, gnum):
    lcoords = lcoords_tri()
    nE = len(C)
    res = np.zeros((nE, 3, 3), dtype=C.dtype)
    for i in prange(3):
        res[:, i, :] = approx_stresses(lcoords[i], C, ecoords, h, sig, dofsol1d, gnum)
    return res


class T3_ALL88_3I(ABC, DrillingMembrane, Triangle, FiniteElement):
    """
    Allman 88 element integrated by 3-point interior rule, constructed
    using the ANDES template of Carmelo Militello and Carlos A. Felippa.

    References
    ----------
    .. [1] Carlos A. Felippa, A study of optimal membrane triangles with
       drilling freedoms, Computer Methods in Applied Mechanics and Engineering,
       Volume 192, Issues 16-18, 2003, Pages 2125-2168, ISSN 0045-7825,
       https://doi.org/10.1016/S0045-7825(03)00253-6.
    """

    version = "ALL88-3I"
    qrule = "full"
    quadrature = {
        "full": (np.array([[1 / 3, 1 / 3]]), np.array([1 / 2])),
    }

    def _elastic_stiffness_matrix_(
        self, *, transform: bool = True, **kwargs
    ) -> ndarray:
        ec = kwargs.get("_ec", None)
        if ec is None:
            _frames = kwargs.get("_frames", self.frames)
            if _frames is not None:
                ec = self.local_coordinates(target=_frames)
            else:
                _topo = kwargs.get("_topo", self.topology().to_numpy())
                ec = self.points_of_cells(topo=_topo)
        D = self.model_stiffness_matrix()
        sig = np.array(sig_ANDES(key=self.version), dtype=float)
        sig = repeat1d(sig, D.shape[0]).reshape(D.shape[0], 11)
        h = np.ones((D.shape[0]), dtype=float)
        K = stiffness_matrix(D, ec, h, sig)
        return self._transform_coeff_matrix_(K) if transform else K

    @classmethod
    def strain_displacement_matrix(cls, pcoord, ecoords):
        # FIXME : not tested, probably wrong
        nE = ecoords.shape[0]
        sig = np.array(sig_ANDES(key=cls.version), dtype=float)
        sig = repeat1d(sig, nE).reshape(nE, 11)
        h = np.ones((nE), dtype=float)
        return strain_displacement_matrix(pcoord, ecoords, h, sig)

    def _strain_load_vector_(self, values: ndarray) -> ndarray:
        # FIXME : returns a zero, but shape-correct array
        nE, _, nRHS = values.shape
        nNE = self.NNODE
        nDOF = self.NDOFN
        nodal_loads = np.zeros((nE, nNE * nDOF, nRHS))
        return nodal_loads  # (nE, nTOTV, nRHS)
