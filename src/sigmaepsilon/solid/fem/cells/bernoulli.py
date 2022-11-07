# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from typing import Union, Callable, Iterable

from neumann import squeeze
from neumann.array import atleast1d, atleastnd, ascont
from neumann.utils import to_range_1d

from .utils.bernoulli import (
    shape_function_matrix_bulk, body_load_vector_bulk,
    global_shape_function_derivatives_bulk as gdshpB,
    lumped_mass_matrices_direct as dlump
)

from .meta import MetaFiniteElement
from ..model.beam import BernoulliBeam, calculate_shear_forces


__all__ = ['BernoulliBase']


class MetaBernoulli(MetaFiniteElement):
    """
    Python metaclass for safe inheritance. Throws a TypeError
    if a method tries to shadow a definition in any of the base
    classes.
    """

    def __init__(self, name, bases, namespace, *args, **kwargs):
        super().__init__(name, bases, namespace, *args, **kwargs)

    def __new__(metaclass, name, bases, namespace, *args, **kwargs):
        cls = super().__new__(metaclass, name, bases, namespace, *args, **kwargs)

        if hasattr(cls, 'shpfnc'):
            if cls.shpfnc is None and isinstance(cls.NNODE, int):
                # generate functions
                N = cls.NNODE

    @classmethod
    def generate_functions(cls, N):
        pass


class ABCBernoulli(metaclass=MetaBernoulli):
    """
    Helper class.
    """
    __slots__ = ()
    dofs = ()
    dofmap = ()


class BernoulliBase(BernoulliBeam):
    """
    Skeleton class for 1d finite elements, whose bending behaviour
    is governed by the Bernoulli theory. The implementations covered
    by this class are independent of the number of nodes of the cells.
    Specification of an actual finite element consists of specifying class
    level attributes only.

    See Also
    --------
    :class:`sigmaepsilon.solid.fem.cells.bernoulli2.Bernoulli2`
    :class:`sigmaepsilon.solid.fem.cells.bernoulli3.Bernoulli3`

    Note
    ----
    Among normal circumstances, you do not have to interact with this class 
    directly. Only use it if you know what you are doing and understand the 
    meaning of the inputs precisely.

    """

    qrule: str = None
    quadrature: dict = None
    shpfnc: Callable = None
    dshpfnc: Callable = None

    def shape_function_values(self, pcoords: Union[float, Iterable[float]], *args,
                              rng: Iterable = None, lengths: Iterable[float] = None, 
                              **kwargs) -> ndarray:
        """
        Evaluates the shape functions at the points specified by 'pcoords'.

        Parameters
        ----------
        pcoords : float or Iterable[float]
            Locations of the evaluation points.

        rng : Iterable, Optional
            The range in which the locations ought to be understood, 
            typically [0, 1] or [-1, 1]. Default is [0, 1].
            
        lengths : Iterable, Optional
            The lengths of the beams in the block. Default is None.

        Returns
        -------
        numpy.ndarray
            The returned array has a shape of (nP, nNE=2, nDOF=6), where nP, 
            nNE and nDOF stand for the number of evaluation points, nodes 
            per element and number of degrees of freedom, respectively.

        """
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        pcoords = atleast1d(np.array(pcoords))
        pcoords = to_range_1d(pcoords, source=rng, target=[-1, 1])
        lengths = self.lengths() if lengths is None else lengths
        return self.__class__.shpfnc(pcoords, lengths).astype(float)

    def shape_function_derivatives(self, pcoords:Union[float, Iterable[float]]=None, *args, 
                                   rng:Iterable=None, jac: ndarray = None, dshp: ndarray = None,
                                   lengths: Iterable[float] = None, **kwargs) -> ndarray:
        """
        Evaluates the shape function derivatives (up to third) at the points specified 
        by 'pcoords'. The function either returns the derivatives on the master or the actual 
        element, depending on the inputs.

        Valid combination of inputs are:

            * 'pcoords' and optionally 'jac' : this can be used to calculate the derivatives
              in both the local ('jac' is provided) and the parametric frame ('jac' is not provided).

            * 'dshp' and 'jac' : this combination can only be used to return the derivatives
              wrt. to any frame.

        Parameters
        ----------
        pcoords : float or Iterable, Optional
            Locations of the evaluation points. Default is None.

        rng : Iterable, Optional
            The range in which the locations ought to be understood, 
            typically [0, 1] or [-1, 1]. Only if 'pcoords' is provided.
            Default is [0, 1].
            
        lengths : Iterable, Optional
            The lengths of the beams in the block. Default is None.

        jac : Iterable, Optional
            The jacobian matrix as a float array of shape (nE, nP, 1, 1), evaluated for 
            each point in each cell. Default is None.

        dshp : Iterable, Optional
            The shape function derivatives of the master element as a float array of 
            shape (nP, nNE, nDOF=6, 3), evaluated at a 'nP' number of points. 
            Default is None.

        Returns
        -------
        numpy.ndarray
            The returned array has a shape of (nP, nNE=2, nDOF=6, 3), where nP, nNE and 
            nDOF stand for the number of evaluation points, nodes per element and number of 
            degrees of freedom, respectively. Number 3 refers to first, second and third 
            derivatives.

        Notes
        -----
        The returned array is always 4 dimensional, even if there is only one 
        evaluation point.

        """
        if pcoords is not None:
            lengths = self.lengths() if lengths is None else lengths
            # calculate derivatives wrt. the parametric coordinates in the range [-1, 1]
            pcoords = atleast1d(np.array(pcoords))
            rng = np.array([-1, 1]) if rng is None else np.array(rng)
            pcoords = to_range_1d(pcoords, source=rng, target=[-1, 1])
            dshp = self.__class__.dshpfnc(pcoords, lengths)
            if jac is None:
                # return derivatives wrt. the parametric frame
                return dshp.astype(float) 
            else:
                # return derivatives wrt. the local frame
                gdshpB(dshp, jac).astype(float)
        elif dshp is not None and jac is not None:
            # return derivatives wrt. the local frame
            return gdshpB(dshp, jac).astype(float)

    def shape_function_matrix(self, pcoords:Union[float, Iterable[float]]=None, *args, 
                              rng:Iterable=None, lengths: Iterable[float] = None, 
                              **kwargs) -> ndarray:
        """
        Evaluates the shape function matrix at the points specified by 'pcoords'.

        Parameters
        ----------
        pcoords : float or Iterable
            Locations of the evaluation points.

        rng : Iterable, Optional
            The range in which the locations ought to be understood, 
            typically [0, 1] or [-1, 1]. Default is [0, 1].

        Other Parameters
        ----------------
        These parameters are for advanced users and can be omitted.
        They are here only to avoid repeated evaulation of common quantities.

        lengths : Iterable, Optional
            The lengths of the beams in the block. Default is None.

        Returns
        -------
        numpy.ndarray
            The returned array has a shape of (nE, nP, nDOF, nDOF * nNE), where nE, nP, 
            nNE and nDOF stand for the number of elements, evaluation points, nodes 
            per element and number of degrees of freedom respectively.

        Notes
        -----
        The returned array is always 4 dimensional, even if there is only one 
        evaluation point.

        """
        pcoords = atleast1d(np.array(pcoords))
        rng = np.array([-1., 1.]) if rng is None else np.array(rng)
        pcoords = to_range_1d(pcoords, source=rng, target=[-1, 1])
        lengths = self.lengths() if lengths is None else lengths
        dshp = self.__class__.dshpfnc(pcoords, lengths)
        ecoords = self.local_coordinates()
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)
        shp = self.shape_function_values(pcoords, rng=rng)
        gdshp = self.shape_function_derivatives(jac=jac, dshp=dshp) 
        return shape_function_matrix_bulk(shp, gdshp).astype(float)

    def integrate_body_loads(self, values: ndarray) -> ndarray:
        """
        Returns nodal representation of body loads.

        Parameters
        ----------
        values : numpy.ndarray
            2d or 3d numpy float array of material densities of shape
            (nE, nNE * nDOF, nRHS) or (nE, nNE * nDOF), where nE, nNE, nDOF and nRHS 
            stand for the number of elements, nodes per element, number of degrees 
            of freedom and number of load cases respectively.

        Returns
        -------
        numpy.ndarray
            An array of shape (nE, nNE * 6, nRHS).

        Notes
        -----
        The returned array is always 3 dimensional, even if there is only one 
        load case.

        See Also
        --------
        :func:`~body_load_vector_bulk`

        """
        values = atleastnd(values, 3, back=True)
        # (nE, nNE * nDOF, nRHS) -> (nE, nRHS, nNE * nDOF)
        values = np.swapaxes(values, 1, 2)
        values = ascont(values)
        qpos, qweights = self.quadrature['full']
        rng = np.array([-1., 1.])
        shp = self.shape_function_values(qpos, rng=rng)  
        # (nP, nNE=2, nDOF=6)
        dshp = self.shape_function_derivatives(qpos, rng=rng)  
        # (nP, nNE=2, nDOF=6, 3)
        ecoords = self.local_coordinates()
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)  
        # (nE, nP, 1, 1)
        djac = self.jacobian(jac=jac)  # (nE, nG)
        gdshp = self.shape_function_derivatives(jac=jac, dshp=dshp)  
        # (nE, nP, nNE=2, nDOF=6, 3)
        return body_load_vector_bulk(values, shp, gdshp, djac, qweights).astype(float)

    def _postproc_local_internal_forces_(self, values: np.ndarray, *args,
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
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)  
        # (nE, nP, 1, 1)
        gdshp = self.shape_function_derivatives(jac=jac, dshp=dshp)
        nE, _, nRHS = dofsol.shape
        nNE, nDOF = self.__class__.NNODE, self.__class__.NDOFN
        dofsol = dofsol.reshape(nE, nNE, nDOF, nRHS)
        D = self.model_stiffness_matrix()[cells]
        values = calculate_shear_forces(dofsol, values, D, gdshp)
        return values  # (nE, nP, 6, nRHS)

    @squeeze(True)
    def lumped_mass_matrix(self, *args, lumping: str = 'direct', alpha: float = 1/50,
                           frmt: str = 'full', **kwargs):
        """
        Returns the lumped mass matrix of the block.

        Parameters
        ----------
        alpha : float, Optional
            A nonnegative parameter, typically between 0 and 1/50 (see notes).
            Default is 1/20.

        lumping : str, Optional
            Controls lumping. Currently only direct lumping is available.
            Default is 'direct'.

        frmt : str, Optional
            Possible values are 'full' or 'diag'. If 'diag', only the diagonal
            entries are returned, if 'full' a full matrix is returned.
            Default is 'full'.

        """
        dbkey = self.__class__._attr_map_['M']
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
