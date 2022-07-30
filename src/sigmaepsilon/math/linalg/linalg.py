# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange, guvectorize as guv
from numpy import ndarray
__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def vpath(p1: ndarray, p2: ndarray, n: int):
    nD = len(p1)
    dist = p2 - p1
    length = np.linalg.norm(dist)
    s = np.linspace(0, length, n)
    res = np.zeros((n, nD), dtype=p1.dtype)
    d = dist / length
    for i in prange(n):
        res[i] = p1 + s[i] * d
    return res


@njit(nogil=True, cache=__cache)
def solve(A, b):
    return np.linalg.solve(A, b)


@njit(nogil=True, cache=__cache)
def inv(A: ndarray):
    return np.linalg.inv(A)


@njit(nogil=True, cache=__cache)
def matmul(A: ndarray, B: ndarray):
    return A @ B


@njit(nogil=True, cache=__cache)
def ATB(A: ndarray, B: ndarray):
    return A.T @ B


@njit(nogil=True, cache=__cache)
def matmulw(A: ndarray, B: ndarray, w: float = 1.0):
    return w * (A @ B)


@njit(nogil=True, cache=__cache)
def ATBA(A: ndarray, B: ndarray):
    return A.T @ B @ A


@njit(nogil=True, cache=__cache)
def ATBAw(A: ndarray, B: ndarray, w: float = 1.0):
    return w * (A.T @ B @ A)


@guv(['(f8[:, :], f8)'], '(n, n) -> ()', nopython=True, cache=__cache)
def det3x3(A, res):
    res = A[0, 0] * A[1, 1] * A[2, 2] - A[0, 0] * A[1, 2] * A[2, 1] \
        - A[0, 1] * A[1, 0] * A[2, 2] + A[0, 1] * A[1, 2] * A[2, 0] \
        + A[0, 2] * A[1, 0] * A[2, 1] - A[0, 2] * A[1, 1] * A[2, 0]


@guv(['(f8[:, :], f8)'], '(n, n) -> ()', nopython=True, cache=__cache)
def det2x2(A, res):
    res = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]


@guv(['(f8[:, :], f8[:, :])'], '(n, n) -> (n, n)', nopython=True, cache=__cache)
def adj3x3(A, res):
    res[0, 0] = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    res[0, 1] = -A[0, 1] * A[2, 2] + A[0, 2] * A[2, 1]
    res[0, 2] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    res[1, 0] = -A[1, 0] * A[2, 2] + A[1, 2] * A[2, 0]
    res[1, 1] = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    res[1, 2] = -A[0, 0] * A[1, 2] + A[0, 2] * A[1, 0]
    res[2, 0] = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    res[2, 1] = -A[0, 0] * A[2, 1] + A[0, 1] * A[2, 0]
    res[2, 2] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]


@guv(['(f8[:, :], f8[:, :])'], '(n, n) -> (n, n)', nopython=True, cache=__cache)
def inv3x3u(A, res):
    det = A[0, 0] * A[1, 1] * A[2, 2] - A[0, 0] * A[1, 2] * A[2, 1] \
        - A[0, 1] * A[1, 0] * A[2, 2] + A[0, 1] * A[1, 2] * A[2, 0] \
        + A[0, 2] * A[1, 0] * A[2, 1] - A[0, 2] * A[1, 1] * A[2, 0]
    res[0, 0] = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    res[0, 1] = -A[0, 1] * A[2, 2] + A[0, 2] * A[2, 1]
    res[0, 2] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    res[1, 0] = -A[1, 0] * A[2, 2] + A[1, 2] * A[2, 0]
    res[1, 1] = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    res[1, 2] = -A[0, 0] * A[1, 2] + A[0, 2] * A[1, 0]
    res[2, 0] = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    res[2, 1] = -A[0, 0] * A[2, 1] + A[0, 1] * A[2, 0]
    res[2, 2] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    res /= det


@njit(nogil=True, cache=__cache)
def inv3x3(A):
    res = np.zeros_like(A)
    det = A[0, 0] * A[1, 1] * A[2, 2] - A[0, 0] * A[1, 2] * A[2, 1] \
        - A[0, 1] * A[1, 0] * A[2, 2] + A[0, 1] * A[1, 2] * A[2, 0] \
        + A[0, 2] * A[1, 0] * A[2, 1] - A[0, 2] * A[1, 1] * A[2, 0]
    res[0, 0] = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    res[0, 1] = -A[0, 1] * A[2, 2] + A[0, 2] * A[2, 1]
    res[0, 2] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    res[1, 0] = -A[1, 0] * A[2, 2] + A[1, 2] * A[2, 0]
    res[1, 1] = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    res[1, 2] = -A[0, 0] * A[1, 2] + A[0, 2] * A[1, 0]
    res[2, 0] = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    res[2, 1] = -A[0, 0] * A[2, 1] + A[0, 1] * A[2, 0]
    res[2, 2] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    res /= det
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def inv3x3_bulk(A):
    res = np.zeros_like(A)
    for i in prange(A.shape[0]):
        det = A[i, 0, 0] * A[i, 1, 1] * A[i, 2, 2] - A[i, 0, 0] * A[i, 1, 2] \
            * A[i, 2, 1] - A[i, 0, 1] * A[i, 1, 0] * A[i, 2, 2] \
            + A[i, 0, 1] * A[i, 1, 2] * A[i, 2, 0] + A[i, 0, 2] \
            * A[i, 1, 0] * A[i, 2, 1] - A[i, 0, 2] * A[i, 1, 1] * A[i, 2, 0]
        res[i, 0, 0] = A[i, 1, 1] * A[i, 2, 2] - A[i, 1, 2] * A[i, 2, 1]
        res[i, 0, 1] = -A[i, 0, 1] * A[i, 2, 2] + A[i, 0, 2] * A[i, 2, 1]
        res[i, 0, 2] = A[i, 0, 1] * A[i, 1, 2] - A[i, 0, 2] * A[i, 1, 1]
        res[i, 1, 0] = -A[i, 1, 0] * A[i, 2, 2] + A[i, 1, 2] * A[i, 2, 0]
        res[i, 1, 1] = A[i, 0, 0] * A[i, 2, 2] - A[i, 0, 2] * A[i, 2, 0]
        res[i, 1, 2] = -A[i, 0, 0] * A[i, 1, 2] + A[i, 0, 2] * A[i, 1, 0]
        res[i, 2, 0] = A[i, 1, 0] * A[i, 2, 1] - A[i, 1, 1] * A[i, 2, 0]
        res[i, 2, 1] = -A[i, 0, 0] * A[i, 2, 1] + A[i, 0, 1] * A[i, 2, 0]
        res[i, 2, 2] = A[i, 0, 0] * A[i, 1, 1] - A[i, 0, 1] * A[i, 1, 0]
        res[i] /= det
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def inv3x3_bulk2(A):
    res = np.zeros_like(A)
    for i in prange(A.shape[0]):
        res[i] = inv3x3(A[i])
    return res


@njit(nogil=True, cache=__cache)
def normalize(A):
    return A/np.linalg.norm(A)


@njit(nogil=True, parallel=True, cache=__cache)
def normalize2d(A):
    res = np.zeros_like(A)
    for i in prange(A.shape[0]):
        res[i] = normalize(A[i])
    return res


@njit(nogil=True, cache=__cache)
def norm(A):
    return np.linalg.norm(A)


@njit(nogil=True, parallel=True, cache=__cache)
def norm2d(A):
    res = np.zeros(A.shape[0])
    for i in prange(A.shape[0]):
        res[i] = norm(A[i, :])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _to_range(vals: ndarray, source: ndarray, target: ndarray):
    res = np.zeros_like(vals)
    s0, s1 = source
    t0, t1 = target
    b = (t1 - t0) / (s1 - s0)
    a = (t0 + t1) / 2 - b * (s0 + s1) / 2
    for i in prange(res.shape[0]):
        res[i] = a + b * vals[i]
    return res


def to_range(vals: ndarray, *args, source: ndarray, target: ndarray = None,
             squeeze=False, **kwargs):
    if not isinstance(vals, ndarray):
        vals = np.array([vals, ])
    source = np.array([0., 1.]) if source is None else np.array(source)
    target = np.array([-1., 1.]) if target is None else np.array(target)
    if squeeze:
        return np.squeeze(_to_range(vals, source, target))
    else:
        return _to_range(vals, source, target)


@njit(nogil=True, parallel=True, cache=__cache)
def _linspace(p0: ndarray, p1: ndarray, N):
    s = p1 - p0
    L = np.linalg.norm(s)
    n = s / L
    djac = L/(N-1)
    step = n * djac
    res = np.zeros((N, p0.shape[0]))
    res[0] = p0
    for i in prange(1, N-1):
        res[i] = p0 + i*step
    res[-1] = p1
    return res


def linspace(start, stop, N):
    if isinstance(start, ndarray):
        return _linspace(start, stop, N)
    else:
        return np.linspace(start, stop, N)


@njit(nogil=True, parallel=True, cache=__cache)
def linspace1d(start, stop, N):
    res = np.zeros(N)
    di = (stop - start) / (N - 1)
    for i in prange(N):
        res[i] = start + i * di
    return res