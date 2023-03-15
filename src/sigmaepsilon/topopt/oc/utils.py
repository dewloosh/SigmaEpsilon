from typing import Union

from numba import njit, prange
import numpy as np
from numpy import ndarray
from numba import types as nbtypes
from numba.typed import Dict as nbDict

from neumann.linalg.sparse import csr_matrix

__cache = True

nbint64 = nbtypes.int64
nbint64A = nbint64[:]
nbfloat64A = nbtypes.float64[:]


@njit(nogil=True, cache=__cache)
def get_filter_factors(
    centers: ndarray, neighbours: Union[nbDict, ndarray], r_max: float
):
    nE = len(centers)
    res = nbDict.empty(
        key_type=nbint64,
        value_type=nbfloat64A,
    )
    for iE in range(nE):
        res[iE] = r_max - norms(centers[neighbours[iE]] - centers[iE])
    return res


@njit(nogil=True, cache=__cache)
def get_filter_factors_csr(centers: ndarray, neighbours: csr_matrix, r_max: float):
    nE = len(centers)
    res = nbDict.empty(
        key_type=nbint64,
        value_type=nbfloat64A,
    )
    for iE in range(nE):
        ni, _ = neighbours.row(iE)
        res[iE] = r_max - norms(centers[ni] - centers[iE])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def norms(a: ndarray):
    nI = len(a)
    res = np.zeros(nI, dtype=a.dtype)
    for iI in prange(len(a)):
        res[iI] = np.dot(a[iI], a[iI])
    return np.sqrt(res)


@njit(nogil=True, parallel=True, cache=__cache)
def norm(a: ndarray):
    return np.sqrt(np.dot(a, a))


@njit(nogil=True, parallel=True, cache=__cache)
def weighted_stiffness_bulk(K: ndarray, weights: ndarray):
    nE = len(weights)
    res = np.zeros_like(K, dtype=K.dtype)
    for i in prange(nE):
        res[i, :] = weights[i] * K[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def weighted_stiffness_flat(K: ndarray, weights: ndarray, kranges: ndarray):
    nE = len(weights)
    res = np.zeros_like(K, dtype=K.dtype)
    for i in prange(nE):
        _r, r_ = kranges[i], kranges[i + 1]
        res[_r:r_] = weights[i] * K[_r:r_]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def weighted_stiffness_1d_regular(data: ndarray, weights: ndarray):
    nE = len(weights)
    dsize = int(len(data) / nE)
    res = np.zeros_like(data, dtype=data.dtype)
    for i in prange(nE):
        res[i * dsize : (i + 1) * dsize] = (
            data[i * dsize : (i + 1) * dsize] * weights[i]
        )
    return res


def element_stiffness_ranges(kshape: tuple) -> ndarray:
    nE = kshape[0]
    if len(kshape) == 3 and isinstance(kshape[1], int):
        res = np.zeros(nE + 1, dtype=int)
        res[1:] = np.cumsum(np.full(nE, kshape[1] * kshape[2]))
    elif isinstance(kshape[1], ndarray):
        res = np.zeros(nE + 1, dtype=int)
        res[1:] = np.cumsum(kshape[1])
    else:
        raise ValueError(f"Invalid shape {kshape}")
    return res
