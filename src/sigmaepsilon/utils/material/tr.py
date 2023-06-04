import numpy as np
from numpy import ndarray
from numba import njit, prange


__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def _tr_stresses_3d_bulk_multi(arr: ndarray, dcm: ndarray):
    nE, nP, nRHS = arr.shape[:3]
    res = np.zeros_like(arr)
    for iE in prange(nE):
        for iP in prange(nP):
            for iRHS in prange(nRHS):
                res[iE, iP, iRHS, :, :] = (
                    dcm[iE, iP] @ arr[iE, iP, iRHS, :, :] @ dcm[iE, iP].T
                )
    return res
