from numba import njit, prange
import numpy as np
from numpy import ndarray
from typing import Union, Iterable

from .solid import Solid

__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def strain_displacement_matrix(gdshp: ndarray):
    """
    Returns the matrix expressing the relationship of generalized strains
    and generalized displacements of a Bernoulli beam element. The order
    of strain components is
        0 : strain along x
        1 : curvature around x
        2 : curvature around y
        3 : curvature around z

    Parameters
    ----------
    gdshp : numpy.ndarray
        First, second and third derivatives for every dof(6) of every node(nNE)
        as a 3d float array of shape (nNE, nDOF=6, 3).

    Returns
    -------
    numpy.ndarray
        Apprpximation coefficients for every generalized strain and every
        shape function as a 2d float array of shape (nSTRE=4, nNE * 6).
    """
    nNE = gdshp.shape[0]
    B = np.zeros((4, nNE * 6), dtype=gdshp.dtype)
    for i in prange(nNE):
        di = i * 6
        # \epsilon_x
        B[0, 0 + di] = gdshp[i, 0, 0]
        # \kappa_x
        B[1, 3 + di] = gdshp[i, 3, 0]
        # \kappa_y
        B[2, 2 + di] = -gdshp[i, 2, 1]
        B[2, 4 + di] = -gdshp[i, 4, 1]
        # \kappa_z
        B[3, 1 + di] = gdshp[i, 1, 1]
        B[3, 5 + di] = gdshp[i, 5, 1]
    return B


@njit(nogil=True, parallel=True, cache=__cache)
def strain_displacement_matrix_bulk(gdshp: ndarray) -> ndarray:
    """
    Calculates the strain-displacement matrix for several elements.

    Parameters
    ----------
    gdshp : numpy.ndarray
        5d float array of shape (nE, nP, nNE, nDOF=6, 3).

    Returns
    -------
    numpy.ndarray
        4d float array of shape (nE, nP, nSTRE=4, nNODE * nDOF=6).
    """
    nE, nP, nNE = gdshp.shape[:3]
    B = np.zeros((nE, nP, 4, nNE * 6), dtype=gdshp.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            B[iE, iP] = strain_displacement_matrix(gdshp[iE, iP])
    return B


@njit(nogil=True, parallel=True, cache=__cache)
def _postproc_bernoulli_internal_forces_L_(
    forces: ndarray, dshp_geom: ndarray, jac: ndarray
) -> ndarray:
    # forces (nE, nNE, nSTRS, nRHS)
    # dshp_geom (nNE, nNE)
    nE, nNE, _, nRHS = forces.shape
    nNE = forces.shape[1]
    for i in prange(nE):
        for j in prange(nNE):
            jac_inv = 1 / jac[i, j, 0, 0]
            for k in prange(nRHS):
                for m in range(nNE):
                    # Vy
                    forces[i, j, 1, k] -= forces[i, m, 5, k] * dshp_geom[j, m] * jac_inv
                    # Vz
                    forces[i, j, 2, k] += forces[i, m, 4, k] * dshp_geom[j, m] * jac_inv
    return forces


@njit(nogil=True, parallel=True, cache=__cache)
def _postproc_bernoulli_internal_forces_H_(
    dofsol: ndarray, forces: ndarray, D: ndarray, gdshp: ndarray
) -> ndarray:
    # dofsol (nE, nNE, nDOF=6, nRHS)
    # forces (nE, nP, 4, nRHS)
    # gdshp (nE, nP, nNE, nDOF=6, 3)
    # D (nE, nSTRE, nSTRE)
    nE, nP, _, nRHS = forces.shape
    nNE = dofsol.shape[1]
    res = np.zeros((nE, nP, 6, nRHS), dtype=forces.dtype)
    res[:, :, 0, :] = forces[:, :, 0, :]
    res[:, :, 3, :] = forces[:, :, 1, :]
    res[:, :, 4, :] = forces[:, :, 2, :]
    res[:, :, 5, :] = forces[:, :, 3, :]
    for i in prange(nE):
        for j in prange(nP):
            for k in prange(nRHS):
                for m in range(nNE):
                    # Vy
                    res[i, j, 1, k] -= D[i, 3, 3] * (
                        gdshp[i, j, m, 1, 2] * dofsol[i, m, 1, k]
                        + gdshp[i, j, m, 5, 2] * dofsol[i, m, 5, k]
                    )
                    # Vz
                    res[i, j, 2, k] -= D[i, 2, 2] * (
                        gdshp[i, j, m, 2, 2] * dofsol[i, m, 2, k]
                        + gdshp[i, j, m, 4, 2] * dofsol[i, m, 4, k]
                    )
    return res


class BernoulliBeam(Solid):
    dofs = ("UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ")

    NDOFN = 6
    NSTRE = 4

    def strain_displacement_matrix(
        self,
        pcoords: Union[float, Iterable[float]] = None,
        *_,
        rng: Iterable = None,
        jac: ndarray = None,
        dshp: ndarray = None,
        **__
    ) -> ndarray:
        """
        Calculates the strain displacement matrix.

        Parameters
        ----------
        pcoords : float or Iterable, Optional
            Locations of the evaluation points. Default is None.
        rng : Iterable, Optional
            The range in which the locations ought to be understood,
            typically [0, 1] or [-1, 1]. Only if 'pcoords' is provided.
            Default is [0, 1].
        jac : Iterable, Optional
            The jacobian matrix as a float array of shape (nE, nP, 1, 1), evaluated for
            each point in each cell. Default is None.
        dshp : Iterable, Optional
            The shape function derivatives of the master element as a float array of
            shape (nP, nNE, nDOF=6, 3), evaluated at a 'nP' number of points.
            Default is None.
        """
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        gdshp = self.shape_function_derivatives(pcoords, rng=rng, jac=jac, dshp=dshp)
        return strain_displacement_matrix_bulk(gdshp)

    def masses(self, *_, values=None, **__) -> ndarray:
        """
        Returns the masses of the cells in the block.
        """
        if isinstance(values, np.ndarray):
            dens = values
        else:
            dens = self.density
        lengths = self.lengths()
        return dens * lengths

    def mass(self, *args, **kwargs) -> float:
        """
        Returns the total mass of the block.
        """
        return np.sum(self.masses(*args, **kwargs))
