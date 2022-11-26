# -*- coding: utf-8 -*-
from scipy.sparse import coo_matrix as coo_np
from scipy.sparse import csr_matrix as csr_np
from scipy.sparse import csc_matrix as csc_np
from scipy.sparse import spmatrix
from scipy.sparse import isspmatrix as isspmatrix_np
import numpy as np
from numpy import ndarray
from numba import njit, prange
from numba.typed import Dict as nbDict
from numba import types as nbtypes

from neumann.array import minmax

__cache = True

nbint64 = nbtypes.int64


def index_mappers(*args, **kwargs):
    if isinstance(args[0], np.ndarray):
        return index_mappers_bulk(*args, **kwargs)
    elif isspmatrix_np(args[0]):
        return index_mappers_sparse(*args, **kwargs)
    raise TypeError


def index_mappers_bulk(inds: np.ndarray, N: int = None, return_trivial: bool = False,
                       return_inverse: bool = False):
    imin, imax = minmax(inds)
    loc_to_glob = array_to_dict(np.unique(inds), 0)
    N = imax if N is None else N
    glob_to_loc = invert_mapping(loc_to_glob) if return_inverse else None
    if imin == 0 and imax == (len(loc_to_glob)-1) == (N - 1):
        if not return_trivial:
            loc_to_glob = None
            glob_to_loc = None
    if return_inverse:
        return loc_to_glob, glob_to_loc
    return loc_to_glob


def index_mappers_sparse(A: spmatrix, return_trivial: bool = False,
                         return_inverse: bool = False):
    if isinstance(A, coo_np):
        inds = A.row
    elif isinstance(A, csr_np) or isinstance(A, csc_np):
        inds = A.indices
    else:
        raise NotImplementedError
    return index_mappers_bulk(inds, A.shape[0], return_trivial, return_inverse)


def box_dof_numbering(gnum: np.ndarray, glob_to_loc: nbDict = None):
    if glob_to_loc is None:
        return gnum
    elif isinstance(glob_to_loc, nbDict):
        return box_indices2d(gnum, glob_to_loc)
    elif isinstance(glob_to_loc, ndarray):
        pass
    raise NotImplementedError


def box_spmatrix(spmat: spmatrix, glob_to_loc: nbDict = None):
    if isinstance(spmat, coo_np):
        return box_coo(spmat, glob_to_loc)
    elif isinstance(spmat, csr_np) or isinstance(spmat, csc_np):
        return box_csr(spmat, glob_to_loc)
    else:
        raise NotImplementedError('Unknown type : {}'.format(type(spmat)))


def box_coo(coo: coo_np, glob_to_loc: nbDict = None, sum_duplicates=True):
    if sum_duplicates:
        coo.sum_duplicates()
    if glob_to_loc is None:
        return coo
    elif isinstance(glob_to_loc, nbDict):
        N = len(glob_to_loc)
        coo.row = box_indices1d(coo.row, glob_to_loc)
        coo.col = box_indices1d(coo.col, glob_to_loc)
        coo._shape = (N, N)
        return coo
    elif isinstance(glob_to_loc, ndarray):
        """
        inds = np.where(glob_to_loc[coo.row] >= 0)[0]
        coo.row = coo.row[inds]
        coo.col = coo.col[inds]
        coo.data = coo.data[inds]
        pass
        """
    raise NotImplementedError


def box_csr(csr: csr_np, glob_to_loc: nbDict = None):
    """
    FIXME. This fails.
    """
    raise RuntimeWarning('FIXME. THIS FAILS!')
    if glob_to_loc is None:
        return csr
    elif isinstance(glob_to_loc, nbDict):
        N = len(glob_to_loc)
        csr.indices, csr.indptr = \
            _box_csr_data(csr.indices, csr.indptr, glob_to_loc)
        csr._shape = (N, N)
        return csr
    raise NotImplementedError


@njit(nogil=True, parallel=True, cache=__cache)
def _box_csr_data(indices: np.ndarray, indptr: np.ndarray,
                  glob_to_loc: nbDict):
    """
    FIXME. This fails.
    """
    raise RuntimeWarning('FIXME. THIS FAILS!')
    N = len(glob_to_loc)
    indices_ = np.zeros(indices.shape[0], dtype=indices.dtype)
    indptr_ = np.zeros(N + 1, dtype=indptr.dtype)
    widths_ = np.zeros(N, dtype=indptr.dtype)
    for i in prange(indices.shape[0]):
        indices_[i] = glob_to_loc[indices[i]]
    for i in prange(N):
        widths_[i] = indptr[i+1] - indptr[i]
    indptr_[1:] = widths_
    return indices_, indptr_


def box_csc(csc: csc_np, glob_to_loc: nbDict = None):
    return box_csr(csc, glob_to_loc)


def box_rhs(f: np.ndarray, loc_to_glob: nbDict = None):
    if loc_to_glob is None:
        return f
    elif isinstance(loc_to_glob, nbDict):
        return _box(f, loc_to_glob)


def unbox_lhs(u: np.ndarray, loc_to_glob: nbDict = None, N: int = None):
    if loc_to_glob is None:
        return u
    elif isinstance(loc_to_glob, nbDict):
        if N is None:
            N = np.max(list(loc_to_glob.values())) + 1
        if len(u.shape) == 1:
            u = u.reshape(len(u), 1)
        return _unbox(u, loc_to_glob, N)


@njit(nogil=True, parallel=True, cache=__cache)
def _unbox(a: np.ndarray, loc_to_glob: nbDict, N: int):
    res = np.zeros((N, a.shape[1]), dtype=a.dtype)
    for i in prange(len(loc_to_glob)):
        res[loc_to_glob[i], :] = a[i, :]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _box(a: np.ndarray, loc_to_glob: nbDict):
    N = len(loc_to_glob)
    res = np.zeros((N, a.shape[1]), dtype=a.dtype)
    for i in prange(N):
        res[i, :] = a[loc_to_glob[i], :]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def box_indices1d(indices: np.ndarray, glob_to_loc: nbDict):
    """
    Returns an 1d index array, renumbered according to the 
    mapping introduced by the mapping `glob_to_loc`.
    """
    res = np.zeros_like(indices)
    for i in prange(res.shape[0]):
        res[i] = glob_to_loc[indices[i]]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def box_indices2d(indices: np.ndarray, glob_to_loc: nbDict):
    res = np.zeros_like(indices)
    for i in prange(res.shape[0]):
        for j in prange(res.shape[1]):
            res[i, j] = glob_to_loc[indices[i, j]]
    return res


@njit(nogil=True, cache=__cache)
def array_to_dict(a: np.ndarray, firstindex: int = 0):
    res = nbDict.empty(key_type=nbint64, value_type=nbint64)
    for i in range(len(a)):
        res[firstindex + i] = a[i]
    return res


@njit(nogil=True, cache=__cache)
def invert_mapping(imap: nbDict):
    res = nbDict.empty(key_type=nbint64, value_type=nbint64)
    for i in imap:
        res[imap[i]] = i
    return res
