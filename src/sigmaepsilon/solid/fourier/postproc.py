# -*- coding: utf-8 -*-
import numpy as np
#from dewloosh.solid.model.mindlin.utils import \
#    stiffness_data_Mindlin, pproc_Mindlin_3D
from numpy import sin, cos, ndarray as nparray, pi as PI
from numba import njit, prange

from neumann.array import atleast2d, atleast3d, atleast4d, itype_of_ftype


__all__ = ['postproc']


UZ, ROTX, ROTY, CX, CY, CXY, EXZ, EYZ, MX, MY, MXY, QX, QY = list(range(13))


def postproc(ABDS: nparray, points: nparray, *args,
             size: tuple = None, shape: tuple = None, squeeze=True,
             angles=None, bounds=None, shear_factors=None,
             C_126=None, C_45=None, separate=True, res2d=None,
             solution=None, model: str = 'mindlin', loads: nparray, **kwargs):
    nD = points.shape[1]
    ABDS = atleast3d(ABDS)
    dtype = ABDS.dtype
    itype = itype_of_ftype(ABDS.dtype)

    # 2d postproc
    if res2d is None and solution is not None:
        assert size is not None
        assert shape is not None
        assert points is not None
        if model.lower() in ['mindlin', 'm']:
            res2d = pproc_Mindlin_2d(np.array(size).astype(dtype),
                                     np.array(shape).astype(itype),
                                     atleast2d(points)[:, :2].astype(dtype),
                                     atleast4d(solution).astype(dtype),
                                     ABDS[:, :3, :3], ABDS[:, 3:, 3:])
        elif model.lower() in ['kirchhoff', 'k']:
            res2d = pproc_Kirchhoff_2d(np.array(size).astype(dtype),
                                       np.array(shape).astype(itype),
                                       atleast2d(points)[:, :2].astype(dtype),
                                       atleast3d(solution).astype(dtype),
                                       ABDS[:, :3, :3], ABDS[:, 3:, 3:], loads)
        res2d = np.sum(res2d, axis=0)

    # 3d postproc
    assert res2d is not None
    assert points is not None
    assert points.shape[0] == res2d.shape[2]
    if nD == 2 and C_126 is not None:
        # lamination scheme may br provided
        assert bounds is not None
        raise NotImplementedError
    elif nD == 3 and C_126 is not None:
        """
        res3d = pproc_Mindlin_Navier_3D(ABDS, C_126, C_45, bounds, points,
                                        res2d, shear_factors=shear_factors,
                                        squeeze=squeeze, angles=angles,
                                        separate=separate)
        return res3d if not squeeze else np.squeeze(res3d)
        """
        raise NotImplementedError
    else:
        return res2d if not squeeze else np.squeeze(res2d)


@njit(nogil=True, parallel=True, cache=True)
def pproc_Mindlin_2d(size, shape: nparray, points: nparray,
                     solution: nparray, D: nparray, S: nparray):
    """
    JIT-compiled function that calculates post-processing quantities 
    at selected ponts for multiple left- and right-hand sides.       

    Parameters
    ----------
    size : tuple 
        (Lx, Ly) : size of domain

    shape : tuple 
        (M, N) : number of harmonic terms involved in x and y
                 directions

    points : numpy.ndarray[nP, 2] 
        2d array of point coordinates

    solution : numpy.ndarray[nRHS, M * N, 3] 
        results of a Navier solution as a 3d array

    D : numpy.ndarray[nLHS, 3, 3] 
        3d array of bending stiffness terms

    S : numpy.ndarray[nLHS, 2, 2]
        3d array of shear stiffness terms

    Returns
    -------
    numpy.ndarray[M * N, nRHS, nLHS, nP, ...] 
        numpy array of post-processing items. The indices along 
        the last axpis denote the following quantities:

            0 : displacement z
            1 : rotation x
            2 : rotation y
            3 : curvature x
            4 : curvature y
            5 : curvature xy
            6 : shear strain xz
            7 : shear strain yz
            8 : moment y
            9 : moment y
            10 : moment xy
            11 : shear force x
            12 : shear force y
    """
    Lx, Ly = size
    M, N = shape
    nP = points.shape[0]
    nRHS, nLHS = solution.shape[:2]
    res2d = np.zeros((M * N, nRHS, nLHS, nP, 13), dtype=D.dtype)
    for iRHS in prange(nRHS):
        for iLHS in prange(nLHS):
            D11, D12, D22, D66 = D[iLHS, 0, 0], D[iLHS, 0, 1], \
                D[iLHS, 1, 1], D[iLHS, 2, 2]
            S44, S55 = S[iLHS, 0, 0], S[iLHS, 1, 1]
            for iP in prange(nP):
                xp, yp = points[iP, :2]
                for m in prange(1, M + 1):
                    Sm = sin(PI*m*xp/Lx)
                    Cm = cos(PI*m*xp/Lx)
                    for n in prange(1, N + 1):
                        iMN = (m-1) * N + n - 1
                        Amn, Bmn, Cmn = solution[iRHS, iLHS, iMN]
                        Sn = sin(PI*n*yp/Ly)
                        Cn = cos(PI*n*yp/Ly)
                        res2d[iMN, iRHS, iLHS, iP, UZ] = Cmn * Sm*Sn
                        res2d[iMN, iRHS, iLHS, iP, ROTX] = Amn * Sm*Cn
                        res2d[iMN, iRHS, iLHS, iP, ROTY] = Bmn * Sn*Cm
                        res2d[iMN, iRHS, iLHS, iP, CX] = -PI*Bmn * m*Sm*Sn/Lx
                        res2d[iMN, iRHS, iLHS, iP, CY] = PI*Amn * n*Sm*Sn/Ly
                        res2d[iMN, iRHS, iLHS, iP, CXY] = -PI*Amn*m*Cm*Cn/Lx + \
                            PI*Bmn*n*Cm*Cn/Ly
                        res2d[iMN, iRHS, iLHS, iP, EXZ] = Bmn*Sn*Cm + \
                            PI*Cmn*m*Sn*Cm/Lx
                        res2d[iMN, iRHS, iLHS, iP, EYZ] = -Amn*Sm*Cn + \
                            PI*Cmn*n*Sm*Cn/Ly
                        res2d[iMN, iRHS, iLHS, iP, MX] = PI*Amn*D12*n*Sm*Sn/Ly - \
                            PI*Bmn*D11*m*Sm*Sn/Lx
                        res2d[iMN, iRHS, iLHS, iP, MY] = PI*Amn*D22*n*Sm*Sn/Ly - \
                            PI*Bmn*D12*m*Sm*Sn/Lx
                        res2d[iMN, iRHS, iLHS, iP, MXY] = -PI*Amn*D66*m*Cm*Cn/Lx + \
                            PI*Bmn*D66*n*Cm*Cn/Ly
                        res2d[iMN, iRHS, iLHS, iP, QX] = Bmn*S55*Sn*Cm + \
                            PI*Cmn*S55*m*Sn*Cm/Lx
                        res2d[iMN, iRHS, iLHS, iP, QY] = -Amn*S44*Sm*Cn + \
                            PI*Cmn*S44*n*Sm*Cn/Ly
    return res2d


@njit(nogil=True, parallel=True, cache=True)
def pproc_Kirchhoff_2d(size, shape: nparray, points: nparray,
                       solution: nparray, D: nparray, S: nparray,
                       loads: nparray):
    """
    JIT-compiled function that calculates post-processing quantities 
    at selected ponts for multiple left- and right-hand sides.       

    Parameters
    ----------
    size : tuple 
        (Lx, Ly) : size of domain

    shape : tuple 
        (M, N) : number of harmonic terms involved in x and y
                 directions

    points : numpy.ndarray[nP, 2] 
        2d array of point coordinates

    solution : numpy.ndarray[nRHS, M * N] 
        results of a Navier solution as a 2d array

    D : numpy.ndarray[nLHS, 3, 3] 
        3d array of bending stiffness terms

    S : numpy.ndarray[nLHS, 2, 2]
        3d array of shear stiffness terms

    Returns
    -------
    numpy.ndarray[M * N, nRHS, nLHS, nP, ...] 
        numpy array of post-processing items. The indices along 
        the last axpis denote the following quantities:

            0 : displacement z
            1 : rotation x
            2 : rotation y
            3 : curvature x
            4 : curvature y
            5 : curvature xy
            6 : shear strain xz
            7 : shear strain yz
            8 : moment y
            9 : moment y
            10 : moment xy
            11 : shear force x
            12 : shear force y
    """
    Lx, Ly = size
    M, N = shape
    nP = points.shape[0]
    nRHS, nLHS = solution.shape[:2]
    res2d = np.zeros((M * N, nRHS, nLHS, nP, 13), dtype=D.dtype)
    for iRHS in prange(nRHS):
        for iLHS in prange(nLHS):
            D11, D12, D22, D66 = D[iLHS, 0, 0], D[iLHS, 0, 1], \
                D[iLHS, 1, 1], D[iLHS, 2, 2]
            S44, S55 = S[iLHS, 0, 0], S[iLHS, 1, 1]
            for iP in prange(nP):
                x, y = points[iP, :2]
                for m in prange(1, M + 1):
                    Sm = sin(PI*m*x/Lx)
                    Cm = cos(PI*m*x/Lx)
                    for n in prange(1, N + 1):
                        Sn = sin(PI*n*y/Ly)
                        Cn = cos(PI*n*y/Ly)
                        iMN = (m-1) * N + n - 1
                        Cmn = solution[iRHS, iLHS, iMN]
                        qxx, qyy = loads[iRHS, iMN, :2]
                        res2d[iMN, iRHS, iLHS, iP, UZ] = Cmn * Sm*Sn
                        res2d[iMN, iRHS, iLHS, iP, ROTX] = PI * Cmn*n*Sm*Cn/Ly
                        res2d[iMN, iRHS, iLHS, iP, ROTY] = -PI * Cmn*m*Sn*Cm/Lx
                        res2d[iMN, iRHS, iLHS, iP, CX] = PI**2 * \
                            Cmn * m**2*Sm*Sn/Lx**2
                        res2d[iMN, iRHS, iLHS, iP, CY] = PI**2 * \
                            Cmn * n**2*Sm*Sn/Ly**2
                        res2d[iMN, iRHS, iLHS, iP, CXY] = - \
                            2*PI**2 * Cmn*m*n*Cm*Cn/(Lx*Ly)
                        res2d[iMN, iRHS, iLHS, iP, EXZ] = PI**3*Cmn*D11*m**3*Sn*Cm/(Lx**3*S44) + \
                            PI**3*Cmn*D12*m*n**2*Sn*Cm/(Lx*Ly**2*S44) + \
                            2*PI**3*Cmn*D66*m*n**2*Sn*Cm / \
                            (Lx*Ly**2*S44) + qxx*Sn*Cm/S44
                        res2d[iMN, iRHS, iLHS, iP, EYZ] = PI**3*Cmn*D12*m**2*n*Sm*Cn/(Lx**2*Ly*S55) + \
                            PI**3*Cmn*D22*n**3*Sm*Cn/(Ly**3*S55) + \
                            2*PI**3*Cmn*D66*m**2*n*Sm*Cn / \
                            (Lx**2*Ly*S55) + qyy*Sm*Cn/S55
                        res2d[iMN, iRHS, iLHS, iP, MX] = PI**2*Cmn*D11*m**2*Sm*Sn/Lx**2 + \
                            PI**2*Cmn*D12*n**2 * Sm*Sn/Ly**2
                        res2d[iMN, iRHS, iLHS, iP, MY] = PI**2*Cmn*D12*m**2*Sm*Sn/Lx**2 + \
                            PI**2*Cmn*D22*n**2 * Sm*Sn/Ly**2
                        res2d[iMN, iRHS, iLHS, iP, MXY] = - \
                            2*PI**2*Cmn * D66*m*n*Cm*Cn/(Lx*Ly)
                        res2d[iMN, iRHS, iLHS, iP, QX] = PI**3*Cmn*D11*m**3*Sn*Cm/Lx**3 + \
                            PI**3*Cmn*D12*m*n**2*Sn*Cm/(Lx*Ly**2) + \
                            2*PI**3*Cmn*D66*m*n**2*Sn*Cm/(Lx*Ly**2) + qxx*Sn*Cm
                        res2d[iMN, iRHS, iLHS, iP, QY] = PI**3*Cmn*D12*m**2*n*Sm*Cn/(Lx**2*Ly) + \
                            PI**3*Cmn*D22*n**3*Sm*Cn/Ly**3 + \
                            2*PI**3*Cmn*D66*m**2*n*Sm*Cn/(Lx**2*Ly) + qyy*Sm*Cn
    return res2d

"""
def pproc_Mindlin_Navier_3D(ABDS: nparray,
                            C_126: nparray, C_45: nparray,
                            bounds: nparray, points: nparray,
                            res2d: nparray, *args,
                            angles=None, separate=True,
                            squeeze=False, shear_factors: nparray = None,
                            **kwargs):
    nLHS, nRHS, nP, _ = res2d.shape
    strains2d = np.zeros((nLHS, nRHS, nP, 8), dtype=ABDS.dtype)
    strains2d[:, :, :, 3:8] = res2d[:, :, :, 3:8]
    ABDS = atleast3d(ABDS)
    res3d = pproc_Mindlin_3D(ABDS, C_126, C_45, bounds, points, strains2d,
                             separate=separate,
                             squeeze=squeeze, angles=angles,
                             shear_factors=shear_factors)
    return res3d


def shell_stiffness_data(shell):
    shell.stiffness_matrixp()
    layers = shell.layers()
    C_126, C_45 = [], []
    for layer in layers:
        Cm = layer.material.stiffness_matrixp()
        C_126.append(Cm[0:3, 0:3])
        C_45.append(Cm[3:, 3:])
    C_126 = np.stack(C_126)
    C_45 = np.stack(C_45)
    angles = np.stack([layer.angle for layer in layers])
    bounds = np.stack([[layer.tmin, layer.tmaxp]
                       for layer in layers])
    ABDS, shear_corrections, shear_factors = \
        stiffness_data_Mindlin(C_126, C_45, angles, bounds)
    ABDS[-2:, -2:] *= shear_corrections
    return C_126, C_45, bounds, angles, ABDS, shear_factors
"""

if __name__ == '__main__':
    pass
