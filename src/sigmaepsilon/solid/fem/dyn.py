# -*- coding: utf-8 -*-
from functools import partial
from typing import Union
from numpy import ndarray
from scipy.sparse import spmatrix


ArrayLike = Union[ndarray, spmatrix]


def effective_modal_mass(M: ArrayLike, action: ndarray, mode: ndarray):
    """
    Returns the effective modal mass for a specific mode.
    
    Assumes that the modal shapes are normalized to the mass matrix.
    
    Parameters
    ----------
    M : 2d array
        Mass matrix as a NumPy or SciPy 2d float array.
        
    action : Iterable
        1d iterable, with a length matching the dof layout of the structure. 
    
    mode : numpy array
        1d array representing a modal shape.
        
    Returns
    -------
    float  
        the effective modal mass
    """
    return (mode @ M @ action)**2


def effective_modal_masses(M: ArrayLike, action: ndarray, modes: ndarray):
    """
    Returns the effective modal mass for several modes.
    
    Assumes that the modal shapes are normalized to the mass matrix.
    
    Parameters
    ----------
    M : 2d array
        Mass matrix as a NumPy or SciPy 2d float array.
        
    action : Iterable
        1d iterable, with a length matching the dof layout of the structure. 
    
    modes : numpy array
        A matrix, whose columns are eigenmodes of interest.
    
    Returns
    -------
    numpy array  
        1d float array of effective modal masses
        
    """
    foo = partial(effective_modal_mass, M, action)
    v = map(lambda i: modes[:, i], range(modes.shape[-1]))
    return list(map(foo, v))
