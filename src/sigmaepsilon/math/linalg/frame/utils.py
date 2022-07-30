# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange

__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def transpose_dcm_multi(dcm: np.ndarray):
    N = dcm.shape[0]
    res = np.zeros_like(dcm)
    for i in prange(N):
        res[i] = dcm[i].T
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def dcm_n(dcm: np.ndarray, n: int):
    res = np.zeros((3 * n, 3 * n), dtype=dcm.dtype)
    for i in prange(n):
        _i = i * 3
        i_ = _i + 3
        res[_i:i_, _i:i_] = dcm
    return res