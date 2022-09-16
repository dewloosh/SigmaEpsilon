# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit, vectorize

__cache = True


@njit(nogil=True, cache=__cache)
def HMH_M(strs: ndarray):
    """
    Evaluates the Huber-Mises-Hencky formula for membranes.

    Parameters
    ----------
    strs : numpy.ndarray
        The stresses s11, s22, s12

    """
    s11, s22, s12 = strs
    return np.sqrt(s11**2 - s11*s22 + s22**2 + 3*s12**2)


@njit(nogil=True, cache=__cache)
def HMH_S(strs: ndarray):
    """
    Evaluates the Huber-Mises-Hencky formula for shells.

    Parameters
    ----------
    strs : numpy.ndarray
        The stresses s11, s22, s12, s13, s23

    """
    s11, s22, s12, s13, s23 = strs
    return np.sqrt(s11**2 - s11*s22 + s22**2 +
                   3*s12**2 + 3*s13**2 + 3*s23**2)


@njit(nogil=True, cache=__cache)
def HMH_3d(strs: ndarray):
    """
    Evaluates the Huber-Mises-Hencky formula for 3d solids.

    Parameters
    ----------
    strs : numpy.ndarray
        The stresses s11, s22, s33, s12, s13, s23

    """
    s11, s22, s33, s12, s13, s23 = strs
    return np.sqrt(0.5*((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2) +
                   3*(s12**2 + s13**2 + s23**2))


@vectorize("f8(f8, f8, f8, f8, f8, f8)", target='parallel', cache=__cache)
def HMH_3d_v(s11, s22, s33, s12, s13, s23):
    """
    Vectorized evaluation of the HMH failure criterion.

    The input values s11, s22, s33, s12, s13, s23 can be
    arbitrary dimensional arrays. 

    """
    return np.sqrt(0.5*((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2) +
                   3*(s12**2 + s13**2 + s23**2))
