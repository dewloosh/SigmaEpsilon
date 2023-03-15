from numba import njit, prange
import numpy as np
from numpy import ndarray

from .solid import Solid
from ...utils.material.hmh import HMH_3d

__cache = True


__all__ = ["Solid3d"]


_NSTRE_ = 6
_NDOFN_ = 3


@njit(nogil=True, parallel=True, cache=__cache)
def strain_displacement_matrix(dshp: ndarray, jac: ndarray) -> ndarray:
    # sxx, syy, szz, syz, sxz, sxy
    nE = jac.shape[0]
    nP, nN = dshp.shape[:2]
    nTOTV = nN * _NDOFN_
    B = np.zeros((nE, nP, _NSTRE_, nTOTV), dtype=dshp.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            gdshp = dshp[iP] @ np.linalg.inv(jac[iE, iP]).T
            for i in prange(nN):
                B[iE, iP, 0, 0 + i * _NDOFN_] = gdshp[i, 0]
                B[iE, iP, 1, 1 + i * _NDOFN_] = gdshp[i, 1]
                B[iE, iP, 2, 2 + i * _NDOFN_] = gdshp[i, 2]
                B[iE, iP, 3, 1 + i * _NDOFN_] = gdshp[i, 2]
                B[iE, iP, 3, 2 + i * _NDOFN_] = gdshp[i, 1]
                B[iE, iP, 4, 0 + i * _NDOFN_] = gdshp[i, 2]
                B[iE, iP, 4, 2 + i * _NDOFN_] = gdshp[i, 0]
                B[iE, iP, 5, 0 + i * _NDOFN_] = gdshp[i, 1]
                B[iE, iP, 5, 1 + i * _NDOFN_] = gdshp[i, 0]
    return B


@njit(nogil=True, parallel=True, cache=__cache)
def HMH_3d_bulk_multi(estrs: np.ndarray) -> ndarray:
    nE, nP = estrs.shape[:2]
    res = np.zeros((nE, nP), dtype=estrs.dtype)
    for iE in prange(nE):
        for jNE in prange(nP):
            res[iE, jNE] = HMH_3d(estrs[iE, jNE])
    return res


class Solid3d(Solid):
    """
    A model for 3d solids.

    See Also
    --------
    :class:`sigmaepsilon.fem.model.solid.Solid`
    """

    dofs = ("UX", "UY", "UZ")

    NDOFN = 3
    NSTRE = 6

    def strain_displacement_matrix(
        self, pcoords: ndarray = None, *, dshp: ndarray = None, jac: ndarray = None, **_
    ) -> ndarray:
        """
        Calculates the strain displacement matrix for solids.

        Parameters
        ----------
        pcoords : numpy.ndarray, Optional
            Locations of evaluation points. Either 'pcoords' or
            both 'dshp' and 'jac' must be provided.
        dshp : numpy.ndarray, Optional
            Shape function derivatives evaluated at some points.
            Only required if 'pcoords' is not provided.
        jac : numpy.ndarray
            Jacobian matrices evaluated at some points.
            Only required if 'pcoords' is not provided.

        Returns
        -------
        numpy.ndarray
            An array of shape (nE, nP, nSTRE, nTOTV), where nE, nP
            nSTRE and nTOTV are the number of elements, evaulation
            points, stress components and total DOFs per element.
        """
        if dshp is None and pcoords is not None:
            dshp = self.shape_function_derivatives(pcoords)
        if jac is None:
            jac = self.jacobian_matrix(dshp=dshp)
        return strain_displacement_matrix(dshp, jac)

    @classmethod
    def HMH(cls, stresses: ndarray) -> ndarray:
        """
        Evaluates the Huber-Mises-Hencky stress at multiple points
        of multiple cells.

        s11, s22, s33, s12, s13, s23

        Parameters
        ----------
        stresses : numpy.ndarray, Optional
            Array of shape (nE, nP, nSTRE), where nE, nP and nSTRE
            are the number of elements, evaulation points and stress
            components. The stresses are expected in the order
            s11, s22, s33, s12, s13, s23.

        Returns
        -------
        numpy.ndarray
            An array of shape (nE, nP), where nE and nP are the
            number of elements and evaulation points.
        """
        return HMH_3d_bulk_multi(stresses)
