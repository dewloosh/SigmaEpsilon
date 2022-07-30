# -*- coding: utf-8 -*-
import numpy as np
from collections import namedtuple


__all__ = ['Quadrature', 'GaussPoints', 'GaussPoints1D', 
           'GaussPoints2D', 'GaussPoints3D']


Quadrature = namedtuple('QuadratureRule', ['inds', 'pos', 'weight'])


def GaussPoints(*args):
    """
    Returns data for numerical integration using the Gauss-Legendre rule.
    
    Parameters
    ----------
    Self evident from the examples.
    
    Returns
    -------
    Self evident from the examples.
    
    
    Examples
    --------
    
    
    """
    nD = len(args)
    if nD == 1:
        return GaussPoints1D(args[0])
    elif nD == 2:
        return GaussPoints2D(args)
    elif nD == 3:
        return GaussPoints3D(args)


def GaussPoints1D(NumPoints):
    x, w = np.polynomial.legendre.leggauss(NumPoints)
    v = np.zeros([2, NumPoints])
    v[0, :] = x
    v[1, :] = w
    return v


def GaussPoints2D(NumPoints):
    nGaus = NumPoints[0] * NumPoints[1]
    QuadraturePos = np.zeros((nGaus, 2))
    QuadratureWeight = np.zeros((nGaus))
    quad1 = GaussPoints1D(NumPoints[0])
    quad2 = GaussPoints1D(NumPoints[1])
    g = 0
    for gi in np.nditer(quad1, flags=['external_loop'], order='F'):
        for gj in np.nditer(quad2, flags=['external_loop'], order='F'):
            QuadraturePos[g] = np.array([gi[0], gj[0]])
            QuadratureWeight[g] = gi[1] * gj[1]
            g += 1
    return QuadraturePos, QuadratureWeight


def GaussPoints3D(NumPoints):
    nGaus = NumPoints[0] * NumPoints[1] * NumPoints[2]
    QuadraturePos = np.zeros((nGaus, 3))
    QuadratureWeight = np.zeros((nGaus))
    quad1 = GaussPoints1D(NumPoints[0])
    quad2 = GaussPoints1D(NumPoints[1])
    quad3 = GaussPoints1D(NumPoints[2])
    g = 0
    for gi in np.nditer(quad1, flags=['external_loop'], order='F'):
        for gj in np.nditer(quad2, flags=['external_loop'], order='F'):
            for gk in np.nditer(quad3, flags=['external_loop'], order='F'):
                QuadraturePos[g] = np.array([gi[0], gj[0], gk[0]])
                QuadratureWeight[g] = gi[1] * gj[1] * gk[1]
                g += 1
    return QuadraturePos, QuadratureWeight
