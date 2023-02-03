from typing import Union

from numba import njit, prange
import numpy as np
from numpy import ndarray
from numba import types as nbtypes
from numba.typed import Dict as nbDict

from neumann.linalg.sparse import csr_matrix, JaggedArray

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
def weighted_stiffness_flat(K: ndarray, weights: ndarray, 
                            kranges: ndarray):
    nE = len(weights)
    res = np.zeros_like(K, dtype=K.dtype)
    for i in prange(nE):
        _r, r_ = kranges[i], kranges[i + 1]
        res[_r : r_] = weights[i] * K[_r : r_]
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


@njit(nogil=True, parallel=True, cache=__cache)
def compliances_bulk(K: ndarray, U: ndarray, gnum: ndarray):
    nE, nNE = gnum.shape
    res = np.zeros(nE, dtype=K.dtype)
    for iE in prange(nE):
        for i in prange(nNE):
            for j in prange(nNE):
                res[iE] += K[iE, i, j] * U[gnum[iE, i]] * U[gnum[iE, j]]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def compliances_flat(K: ndarray, U: ndarray, kcols: ndarray, 
                     kranges: ndarray) -> ndarray:
    nE = len(kranges) - 1
    res = np.zeros(nE, dtype=K.dtype)
    for i in prange(nE):
        _r, r_ = kranges[i], kranges[i + 1]
        res[i] = np.sum(K[_r : r_] * U[kcols[_r : r_]])
    return res


def element_stiffness_ranges(K_bulk: Union[ndarray, JaggedArray]) -> ndarray:
    nE = len(K_bulk)
    if isinstance(K_bulk, ndarray):
        assert len(K_bulk.shape) == 3, "Mismatch in shape"
        res = np.zeros(nE + 1, dtype=int)
        res[1:] = np.cumsum(np.full(nE, K_bulk.shape[1] * K_bulk.shape[2]))
    elif isinstance(K_bulk, JaggedArray):
        res = np.zeros(nE + 1, dtype=int)
        res[1:] = np.cumsum(K_bulk.widths())
    else:
        raise TypeError(f"Invalid type {type(K_bulk)}")
    return res