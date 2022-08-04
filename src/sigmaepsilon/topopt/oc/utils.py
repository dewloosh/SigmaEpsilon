# -*- coding: utf-8 -*-
from typing import Union
from numba import njit, prange
import numpy as np
from numpy import ndarray
from numba import types as nbtypes
from numba.typed import Dict as nbDict

from sigmaepsilon.math.linalg.sparse.csr import csr_matrix
from sigmaepsilon.solid.fem.utils import irows_icols_bulk_filtered

__cache = True

nbint64 = nbtypes.int64
nbint64A = nbint64[:]
nbfloat64A = nbtypes.float64[:]


def filter_stiffness(K_bulk: np.ndarray, edofs: np.ndarray,
                     vals: np.ndarray, *args, tol=1e-12, **kwargs):
    inds = np.where(vals > tol)[0]
    rows, cols = irows_icols_bulk_filtered(edofs, inds)
    return K_bulk[inds, :, :].flatten(), (rows.flatten(), cols.flatten())


@njit(nogil=True, cache=__cache)
def get_filter_factors(centers: np.ndarray, neighbours: Union[nbDict, ndarray], 
                       r_max: float):
    nE = len(centers)
    res = nbDict.empty(
        key_type=nbint64,
        value_type=nbfloat64A,
    )
    for iE in range(nE):
        res[iE] = r_max - norms(centers[neighbours[iE]] - centers[iE])
    return res


@njit(nogil=True, cache=__cache)
def get_filter_factors_csr(centers: np.ndarray, neighbours: csr_matrix,
                           r_max: float):
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
def norms(a: np.ndarray):
    nI = len(a)
    res = np.zeros(nI, dtype=a.dtype)
    for iI in prange(len(a)):
        res[iI] = np.dot(a[iI], a[iI])
    return np.sqrt(res)


@njit(nogil=True, parallel=True, cache=__cache)
def norm(a: np.ndarray):
    return np.sqrt(np.dot(a, a))


@njit(nogil=True, parallel=True, cache=__cache)
def weighted_stiffness_bulk(K: np.ndarray, weights: np.ndarray):
    nE = len(weights)
    res = np.zeros_like(K, dtype=K.dtype)
    for i in prange(nE):
        res[i, :] = weights[i] * K[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def weighted_stiffness_1d(data: np.ndarray, weights: np.ndarray):
    nE = len(weights)
    dsize = int(len(data)/nE)
    res = np.zeros_like(data, dtype=data.dtype)
    for i in prange(nE):
        res[i*dsize: (i+1)*dsize] = data[i*dsize: (i+1)*dsize] * weights[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def compliances_bulk(K: np.ndarray, U: np.ndarray, gnum: np.ndarray):
    nE, nNE = gnum.shape
    res = np.zeros(nE, dtype=K.dtype)
    for iE in prange(nE):
        for i in prange(nNE):
            for j in prange(nNE):
                res[iE] += K[iE, i, j] * U[gnum[iE, i]] * U[gnum[iE, j]]
    return res
