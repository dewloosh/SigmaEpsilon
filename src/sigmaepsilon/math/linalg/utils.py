# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange, prange

__cache = True


@njit(nogil=True, cache=__cache)
def show_vector(dcm: np.ndarray, arr: np.ndarray):
    """
    Returns the coordinates of a single vector in a frame specified
    by a DCM matrix.
    
    Parameters
    ----------
    dcm : ndarray
        The dcm matrix of the transformation as a 2d float array.
    
    arr : ndarray
        1d float array of coordinates of a single vector.
    
    Returns
    -------      
    numpy.ndarray
        The new coordinates of the vector with the same shape as `arr`.
        
    """
    return dcm @ arr


@njit(nogil=True, parallel=True, cache=__cache)
def show_vectors(dcm: np.ndarray, arr: np.ndarray):
    """
    Returns the coordinates of multiple vectors in a frame specified
    by a DCM matrix.
    
    Parameters
    ----------
    dcm : ndarray
        The dcm matrix of the transformation as a 2d float array.
    
    arr : ndarray
        2d float array of coordinates of multiple vectors.
    
    Returns
    -------      
    numpy.ndarray
        The new coordinates of the vectors with the same shape as `arr`.
        
    """
    res = np.zeros_like(arr)
    for i in prange(arr.shape[0]):
        res[i] = dcm @ arr[i, :]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def show_vectors_multi(dcm: np.ndarray, arr: np.ndarray):
    """
    Returns the coordinates of multiple vectors and multiple DCM matrices.
    
    Parameters
    ----------
    dcm : ndarray
        The dcm matrix of the transformation as a 3d float array.
    
    arr : ndarray
        2d float array of coordinates of multiple vectors.
    
    Returns
    -------      
    numpy.ndarray
        The new coordinates of the vectors with the same shape as `arr`.
        
    """
    res = np.zeros_like(arr)
    for i in prange(arr.shape[0]):
        res[i] = dcm[i] @ arr[i, :]
    return res
