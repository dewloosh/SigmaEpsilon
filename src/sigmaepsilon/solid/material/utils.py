# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit

__cache = True


@njit(nogil=True, cache=__cache)
def HMH_M(strs: ndarray):
    s11, s22, s12 = strs
    return np.sqrt(s11**2 - s11*s22 + s22**2 + 3*s12**2)


@njit(nogil=True, cache=__cache)
def HMH_S(strs: ndarray):
    s11, s22, s12, s13, s23 = strs
    return np.sqrt(s11**2 - s11*s22 + s22**2 +
                   3*s12**2 + 3*s13**2 + 3*s23**2)


@njit(nogil=True, cache=__cache)
def HMH_3d(strs: ndarray):
    s11, s22, s33, s12, s13, s23 = strs
    return np.sqrt(0.5*((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2) +
                   3*(s12**2 + s13**2 + s23**2))
    