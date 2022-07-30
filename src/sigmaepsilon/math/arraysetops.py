# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from awkward import Array as akarray
from numba import njit
from numba.core import types
from numba.typed import Dict
from typing import Union
import awkward as ak

from .array import flatten2dC
from .array import count_cols

__cache = True


ArrayLike = Union[ndarray, akarray]
i64 = types.int64
i64A = types.int64[:]
i64A2 = types.int64[:, :]


def unique2d(arr: ArrayLike, return_index=False,
             return_inverse=False, return_counts=False):
    if isinstance(arr, ndarray):
        unique, counts, inverse, indices = unique2d_njit(arr)
        res = [unique, ]
        if return_index:
            res.append(indices)
        if return_inverse:
            res.append(inverse)
        if return_counts:
            res.append(counts)
        return res if len(res) > 1 else res[0]
    elif isinstance(arr, akarray):
        assert arr.ndim == 2, "Only 2 dimensional awkward arrays are supported!"
        if hasattr(arr, 'is_jagged'):
            if not arr.is_jagged():
                return unique2d(arr.to_numpy(), return_index=return_index,
                                return_inverse=return_inverse,
                                return_counts=return_counts)
        flatarr = ak.flatten(arr).to_numpy()
        unique, counts, inverse, indices = unique1d_njit(flatarr)
        res = [unique, ]
        if return_index:
            res.append(indices)
        if return_inverse:
            res.append(ak.unflatten(inverse, count_cols(arr)))
        if return_counts:
            res.append(counts)
        return res if len(res) > 1 else res[0]


@njit(nogil=True, parallel=False, fastmath=False, cache=__cache)
def unique1d_njit(flatdata: ndarray):
    unique = np.unique(flatdata)
    nF = len(flatdata)
    nU = len(unique)
    counts = np.zeros(nU, dtype=np.int64)
    inverse = np.zeros(nF, dtype=np.int64)
    imap = Dict.empty(key_type=i64, value_type=i64)
    for i in range(nU):
        imap[unique[i]] = i
    for i in range(nF):
        counts[imap[flatdata[i]]] += 1
        inverse[i] = imap[flatdata[i]]
    indices = Dict.empty(key_type=i64, value_type=i64A)
    for i in range(nU):
        indices[unique[i]] = np.zeros((counts[i]), dtype=np.int64)
    counts[:] = 0
    for iF in range(nF):
        i = inverse[iF]
        indices[unique[i]][counts[i]] = iF
        counts[i] += 1
    return unique, counts, inverse, indices


@njit(nogil=True, parallel=False, fastmath=False, cache=__cache)
def unique2d_njit(data: ndarray):
    flatdata = flatten2dC(data)
    unique = np.unique(flatdata)
    nF = len(flatdata)
    nU = len(unique)
    counts = np.zeros(nU, dtype=np.int64)
    inverse = np.zeros(nF, dtype=np.int64)
    imap = Dict.empty(key_type=i64, value_type=i64)
    for i in range(nU):
        imap[unique[i]] = i
    for i in range(nF):
        counts[imap[flatdata[i]]] += 1
        inverse[i] = imap[flatdata[i]]
    inverse = inverse.reshape(data.shape)
    indices = Dict.empty(key_type=i64, value_type=i64A2)
    for i in range(nU):
        indices[unique[i]] = np.zeros((counts[i], 2), dtype=np.int64)
    counts[:] = 0
    nR, nC = data.shape
    for r in range(nR):
        for c in range(nC):
            i = inverse[r, c]
            indices[unique[i]][counts[i], 0] = r
            indices[unique[i]][counts[i], 1] = c
            counts[i] += 1
    return unique, counts, inverse, indices

"""
@njit(nogil=True, parallel=False, fastmath=False, cache=__cache)
def unique2d_njit_jagged(data: akarray):
    return ak.flatten(data)
"""