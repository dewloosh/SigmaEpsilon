# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from collections import Iterable
from typing import Union, Callable

from dewloosh.math import squeeze, is_none_or_false
from dewloosh.math.array import atleast1d, atleastnd, ascont
from dewloosh.math.utils import to_range

from dewloosh.solid.fem.cells.utils.bernoulli import (
    shape_function_matrix_bulk, body_load_vector_bulk,
    global_shape_function_derivatives_bulk as gdshpB,
    lumped_mass_matrices_direct as dlump
)

from ..model.beam import BernoulliBeam, calculate_shear_forces


__all__ = ['BernoulliBase']


ArrayOrFloat = Union[ndarray, float]


class BernoulliBase(BernoulliBeam):

    qrule: str = None
    quadrature: dict = None
    shpfnc: Callable = None
    dshpfnc: Callable = None

    def shape_function_values(self, pcoords: ArrayOrFloat, *args,
                              rng: Iterable = None, lengths=None,
                              **kwargs) -> ndarray:
        """
        (nE, nP, nNE=2, nDOF=6)
        """
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        pcoords = atleast1d(np.array(pcoords) if isinstance(
            pcoords, list) else pcoords)
        pcoords = to_range(pcoords, source=rng, target=[-1, 1])
        lengths = self.lengths() if lengths is None else lengths
        return self.__class__.shpfnc(pcoords, lengths).astype(float)

    def shape_function_derivatives(self, pcoords=None, *args, rng=None,
                                   jac: ndarray = None, dshp: ndarray = None,
                                   lengths=None, **kwargs) -> ndarray:
        """
        (nE, nP, nNE=2, nDOF=6, 3)
        """
        lengths = self.lengths() if lengths is None else lengths
        if pcoords is not None:
            # calculate derivatives wrt. the parametric coordinates in the range [-1, 1]
            pcoords = atleast1d(np.array(pcoords) if isinstance(
                pcoords, list) else pcoords)
            rng = np.array([-1, 1]) if rng is None else np.array(rng)
            pcoords = to_range(pcoords, source=rng, target=[-1, 1])
            dshp = self.__class__.dshpfnc(pcoords, lengths)
            # return derivatives wrt the local frame if jacobian is provided, otherwise
            # return derivatives wrt. the parametric coordinate in the range [-1, 1]
            return dshp.astype(float) if jac is None else gdshpB(dshp, jac).astype(float)
        elif dshp is not None and jac is not None:
            # return derivatives wrt the local frame
            return gdshpB(dshp, jac).astype(float)

    def shape_function_matrix(self, pcoords=None, *args, rng=None,
                              lengths=None, **kwargs) -> ndarray:
        """
        (nE, nP, nDOF, nDOF * nNODE)
        """
        pcoords = atleast1d(np.array(pcoords) if isinstance(
            pcoords, list) else pcoords)
        rng = np.array([-1., 1.]) if rng is None else np.array(rng)
        lengths = self.lengths() if lengths is None else lengths
        shp = self.shape_function_values(pcoords, rng=rng, lengths=lengths)
        gdshp = self.shape_function_derivatives(pcoords, *args, rng=rng,
                                                lengths=lengths, **kwargs)
        return shape_function_matrix_bulk(shp, gdshp).astype(float)

    def integrate_body_loads(self, values: ndarray) -> ndarray:
        """
        values (nE, nNE * nDOF, nRHS)
        """
        values = atleastnd(values, 3, back=True)
        # (nE, nNE * nDOF, nRHS) -> (nE, nRHS, nNE * nDOF)
        values = np.swapaxes(values, 1, 2)
        values = ascont(values)
        qpos, qweights = self.quadrature['full']
        rng = np.array([-1., 1.])
        shp = self.shape_function_values(
            qpos, rng=rng)  # (nE, nP, nNE=2, nDOF=6)
        dshp = self.shape_function_derivatives(
            qpos, rng=rng)  # (nE, nP, nNE=2, nDOF=6, 3)
        ecoords = self.local_coordinates()
        jac = self.jacobian_matrix(
            dshp=dshp, ecoords=ecoords)  # (nE, nP, 1, 1)
        djac = self.jacobian(jac=jac)  # (nE, nG)
        gdshp = self.shape_function_derivatives(
            qpos, rng=rng, jac=jac, dshp=dshp)  # (nE, nP, nNE=2, nDOF=6, 3)
        return body_load_vector_bulk(values, shp, gdshp, djac, qweights).astype(float)

    def _postproc_local_internal_forces(self, values: np.ndarray, *args,
                                        rng, points, cells, dofsol, **kwargs):
        """
        Documentation is at the base element at '...solid\\fem\\elem.py'
        values (nE, nP, 4, nRHS)
        dofsol (nE, nEVAB, nRHS)
        points (nP,)
        (nE, nP, 6, nRHS) out
        """
        ecoords = self.local_coordinates()[cells]
        dshp = self.shape_function_derivatives(points, rng=rng)[cells]
        jac = self.jacobian_matrix(
            dshp=dshp, ecoords=ecoords)  # (nE, nP, 1, 1)
        gdshp = self.shape_function_derivatives(
            points, rng=rng, jac=jac, dshp=dshp)
        nE, nEVAB, nRHS = dofsol.shape
        nNE, nDOF = self.__class__.NNODE, self.__class__.NDOFN
        dofsol = dofsol.reshape(nE, nNE, nDOF, nRHS)
        D = self.model_stiffness_matrix()[cells]
        values = calculate_shear_forces(dofsol, values, D, gdshp)
        return values  # (nE, nP, 6, nRHS)

    @squeeze(True)
    def mass_matrix(self, *args, lumping=None, alpha: float = 1/50,
                    frmt='full', **kwargs):
        """
        lumping : 'direct', None or False
        frmt : only if lumping is specified
        """
        dbkey = self.__class__._attr_map_['M']
        if is_none_or_false(lumping):
            return super().mass_matrix(*args, squeeze=False, **kwargs)
        if lumping == 'direct':
            dens = self.db.density
            try:
                areas = self.areas()
            except Exception:
                areas = np.ones_like(dens)
            lengths = self.lengths()
            topo = self.topology()
            ediags = dlump(dens, lengths, areas, topo, alpha)
            if frmt == 'full':
                N = ediags.shape[-1]
                M = np.zeros((ediags.shape[0], N, N))
                inds = np.arange(N)
                M[:, inds, inds] = ediags
                self.db[dbkey] = M
                return M
            elif frmt == 'diag':
                self.db[dbkey] = ediags
                return ediags
        raise RuntimeError("Lumping mode not recognized :(")
