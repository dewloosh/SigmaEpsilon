from numba import njit, prange
import numpy as np
from numba.typed import Dict as nbDict

from neumann.linalg.sparse.csr import csr_matrix as csr

__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def sensitivity_filter(
    sens: np.ndarray, dens: np.ndarray, neighbours: nbDict, factors: dict
):
    nE = len(dens)
    res = np.zeros_like(sens)
    inds = np.arange(nE).astype(np.int64)
    for iE in prange(nE):
        adj = neighbours[inds[iE]]
        res[inds[iE]] = np.sum(factors[inds[iE]] * dens[adj] * sens[adj]) / (
            dens[inds[iE]] * np.sum(factors[inds[iE]])
        )
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def sensitivity_filter_csr(
    sens: np.ndarray, dens: np.ndarray, neighbours: csr, factors: dict
):
    nE = len(dens)
    res = np.zeros_like(sens)
    for iE in prange(nE):
        adj, _ = neighbours.row(iE)
        res[iE] = np.sum(factors[iE] * dens[adj] * sens[adj]) / (
            dens[iE] * np.sum(factors[iE])
        )
    return res
