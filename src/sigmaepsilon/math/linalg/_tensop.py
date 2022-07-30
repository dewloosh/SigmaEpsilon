# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, config, prange


def tr_3333(array: np.ndarray, dcm: np.ndarray):
    Q = dcm.T
    res = np.zeros((3, 3, 3, 3), dtype=array.dtype)
    for p in range(3):
        for q in range(3):
            for r in range(3):
                for s in range(3):
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                for m in range(3):
                                    res[p, q, r, s] += \
                                        Q[p, i] * Q[q, j] * Q[r, k] * \
                                        Q[s, m] * array[i, j, k, m]
    return res


@njit(nogil=True, parallel=True, cache=True)
def tr_3333_jit(array: np.ndarray, dcm: np.ndarray):
    Q = dcm.T
    res = np.zeros((3, 3, 3, 3), dtype=array.dtype)
    for p in prange(3):
        for q in prange(3):
            for r in prange(3):
                for s in prange(3):
                    for i in prange(3):
                        for j in prange(3):
                            for k in prange(3):
                                for m in prange(3):
                                    res[p, q, r, s] += \
                                        Q[p, i] * Q[q, j] * Q[r, k] * \
                                        Q[s, m] * array[i, j, k, m]
    return res


def tr_3333_jit_compiled(array: np.ndarray, dcm: np.ndarray):
    ...


_code = compile("""
@njit(nogil = True, parallel = True)
def tr_3333_jit_compiled(array : np.ndarray, dcm : np.ndarray):
    Q = dcm.T
    res = np.zeros((3, 3, 3, 3), dtype = array.dtype)
    for p in prange(3):
        for q in prange(3):
            for r in prange(3):
                for s in prange(3):
                    for i in prange(3):
                        for j in prange(3):
                            for k in prange(3):
                                for m in prange(3):
                                    res[p, q, r, s] += \
                                        Q[p, i] * Q[q, j] * Q[r, k] * \
                                        Q[s, m] * array[i, j, k, m]
    return res
""", 'tr_3333_jit_compiled', 'exec')
exec(_code)


def tr_3333_np(array: np.ndarray, dcm: np.ndarray,
               optimize='greedy') -> np.ndarray:
    Q = dcm.T
    return np.einsum('pi, qj, rk, sl, ijkl', Q, Q, Q, Q, array,
                     optimize=optimize)


if __name__ == '__main__':
    from dewloosh.math.linalg.frame import ReferenceFrame
    from dewloosh.math.linalg.tensor import Tensor, Tensor3333
    import time

    A = ReferenceFrame()
    B = A.orient_new('Body', [0, 0, 90*np.pi/180],  'XYZ')

    tA = Tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]], frame=A)
    tA.transform_to_frame(B)

    C = Tensor3333(np.zeros((3, 3, 3, 3)), frame=A)
    dcm = B.dcm()
    array = C._array
    Q = np.transpose(dcm)
    path = np.einsum_path('pi,qj,rk,sl,ijkl', Q, Q, Q, Q, array,
                          optimize='greedy')[0]

    # PERFORMENCE MEASURE
    times = []
    # measuring solution 1
    ts = time.time()
    for i in range(100):
        tr_3333(array, dcm)
    te = time.time()
    times.append((te-ts)*1000)
    
    # measuring solution 2
    ts = time.time()
    for i in range(100):
        tr_3333_jit(array, dcm)
    te = time.time()
    times.append((te-ts)*1000)
    
    # measuring solution 3
    ts = time.time()
    for i in range(100):
        tr_3333_jit_compiled(array, dcm)
    te = time.time()
    times.append((te-ts)*1000)
    
    # measuring solution 4
    ts = time.time()
    for i in range(100):
        tr_3333_np(array, dcm)
    te = time.time()
    times.append((te-ts)*1000)
    
    # measuring solution 5
    ts = time.time()
    for i in range(100):
        tr_3333_np(array, dcm, optimize=path)
    te = time.time()
    times.append((te-ts)*1000)
    print(times)
