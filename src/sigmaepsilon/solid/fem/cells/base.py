# -*- coding: utf-8 -*-
from Bernoulli3DBeam.utils import cell_coords_bulk, \
    stiffness_matrix_bulk, mass_matrix_bulk, gdshp_bulk, \
    jacobian_matrix_bulk, jacobian_det_bulk, topo_to_gnum, \
    pcoords_to_coords, collect_nodal_data, nodal_distribution_factors, \
    lengths_of_lines, distribute_nodal_data, to_range
from Bernoulli3DBeam.preproc import fem_coeff_matrix_coo, assemble_load_vector
from Bernoulli3DBeam.transform import element_transformation_matrices, \
    nodal_transformation_matrices, \
    transform_element_matrices_out as tr_2d_out, \
    transform_element_vectors_in as tr_1d_in, \
    transform_element_vectors_out as tr_1d_out, \
    transform_element_vectors_out_multi as tr_1d_out_multi, \
    transform_numint_forces_out as tr_cell_gauss_out, \
    transform_numint_forces_in as tr_cell_gauss_in
from Bernoulli3DBeam.postproc import element_dof_solution_bulk, \
    approx_element_dof_solution_bulk, extrapolate_gauss_data
from dewloosh.math.array import atleast1d, atleast2d, atleast3d, repeat
from dewloosh.math.numint import Quadrature
from dewloosh.core import squeeze
from scipy.sparse import coo_matrix as coo
from typing import Union
import numpy as np
from numpy import ndarray, swapaxes as swap, ascontiguousarray as ascont


ArrayOrFloat = Union[float, ndarray, list]


class BernoulliBase:
    """
    Base class for Bernoulli Beam-Columns.
    """

    qrule = None
    quadrature: dict = None
    NDOFN = 6  # number of dofs per node
    NSTRE = 4  # number of strain components
    lcoords = None

    @classmethod
    @squeeze(True)
    def shape_function_values(cls, pcoords: ArrayOrFloat, *args,
                              rng=None, **kwargs) -> ndarray:
        """
        Evaulates shape functions at local points. The coordinates
        of the poins are expected in the range [0, 1].

        Parameters
        ----------
        pcoords : ArrayOrFloat
            float or 1d float array of local coordinates.

        squeeze : bool, Optional
            If True, returns the result without dummy axes.
            Default is True.

        rng : Iterable
            Bounds of the interval, where the evaluation points are understood.
            Default is [0, 1]

        Returns
        -------
        numpy.ndarray 
            2d array of size (nP, nSHP) if pcoords is iterable or squeeze=False,
            1d array of size (nSHP,) otherwise.
                nP : number of evaluation points
                nSHP : number of shape functions

        Examples
        --------
        Typical usage:

        >>> BernoulliBase.shape_function_values(0.5, squeeze=False).shape
        (1, nSHP)
        >>> BernoulliBase.shape_function_values(0.5).shape
        (nSHP,)
        >>> BernoulliBase.shape_function_values([0.0, 0.5, 1.0]).shape
        (3, nSHP)

        These calls are equivalent:

        >>> BernoulliBase.shape_function_values([0.0, 1.0])
        ...
        >>> BernoulliBase.shape_function_values([-1.0, 1.0], rng=[-1, 1])
        ...

        """
        rng = np.array([0., 1.]) if rng is None else np.array(rng)
        pcoords = atleast1d(np.array(pcoords) if isinstance(
            pcoords, list) else pcoords)
        return cls._shape_function_values(pcoords, rng=rng)

    @classmethod
    @squeeze(True)
    def shape_function_derivatives(cls, pcoords: ArrayOrFloat, *args,
                                   rng=None, **kwargs) -> ndarray:
        """
        Evaulates derivatives of shape functions at local points. 
        The points of evaluation are expected in the range [0, 1].

        Parameters
        ----------

        pcoords : ArrayOrFloat
            float or 1d float array of local coordinates.

        squeeze : bool, Optional
            If True, returns the result without dummy axes.
            Default is True.

        rng : Iterable
            Bounds of the interval, where the evaluation points are understood.
            Default is [0, 1]

        Returns
        -------
        numpy.ndarray
            3d array of size (nP, nSHP, 3) if pcoords is iterable or squeeze=False,
            2d array of size (nSHP, 3) otherwise.
                nP : number of evaluation points
                nSHP : number of shape functions
                3 : values for first, second and third derivatives

        """
        rng = np.array([0., 1.]) if rng is None else np.array(rng)
        pcoords = atleast1d(np.array(pcoords) if isinstance(
            pcoords, list) else pcoords)
        return cls._shape_function_derivatives(pcoords, *args, rng=rng, **kwargs)

    @classmethod
    @squeeze(True)
    def shape_function_matrix(cls, pcoords: ArrayOrFloat, *args,
                              coords: ndarray = None, topo: ndarray = None,
                              ecoords=None, rng=None, **kwargs) -> ndarray:
        """
        Returns the shape function matrix N evaluated in a number of 
        evaluation points nP. The points of evaluation are expected 
        in the range [0, 1].

        Parameters
        ----------

        pcoords : ArrayOrFloat
            float or 1d float array of local coordinates.

        coords : numpy.ndarray, Optional
            2d array of vertex coordinates of shape (nV, nD)
            If provided, it must contain entries for all indices in `topo`.
                nV : number of vertices in the model
                nD : number of dimensions of the model space

        topo : (nNE) or (nE, nNE) numpy.ndarray, Optional
            1d or 2d array of vertex indices. 
            The i-th row contains the vertex indices of the i-th element.
                nE : number of elements
                nNE : number of nodes per element

        ecoords : (nNE, nD) or (nE, nNE, nD) numpy.ndarray, Optional
            2d or 3d array of element coordinates. 
                nE : number of elements
                nNE : number of nodes per element
                nD : number of dimensions of the model space

        squeeze : bool, Optional
            If True, returns the result without dummy axes.
            Default is True.

        rng : Iterable
            Bounds of the interval, where the evaluation points are understood.
            Default is [0, 1]

        Returns
        -------
        numpy.ndarray
            4d array of size (nE, nP, nSHP, nDOF)
                nE : number of elements
                nP : number of evaluation points
                nSHP : number of shape functions
                nDOF : total number of degrees of freedom of an element

        Notes
        -----
        (1) If squeeze=True, the returned array can be 2d, 3d or 4d, depending
            on nE and nP.

        (2) Either `ecoords`, or both `coords` and `topo` must be provided.

        """
        rng = np.array([0., 1.]) if rng is None else np.array(rng)
        if ecoords is None:
            assert topo is not None and coords is not None
            topo = atleast2d(topo)
            ecoords = cell_coords_bulk(coords, topo)
        lengths = lengths_of_lines(ecoords)
        shp = cls.shape_function_values(pcoords, rng=rng, squeeze=False)
        gdshp = cls.shape_function_derivatives(pcoords, rng=rng,
                                               squeeze=False,
                                               lengths=lengths)
        return cls._shape_function_matrix(shp, gdshp)

    @classmethod
    @squeeze(True)
    def strain_displacement_matrix(cls, pcoords: ArrayOrFloat, *args,
                                   coords: ndarray = None, topo: ndarray = None,
                                   ecoords=None, rng=None, **kwargs) -> ndarray:
        """
        Return the strain displacement matrix B evaluated in a number of 
        evaluation points nP. The points of evaluation are expected in the 
        range [0, 1].

        Parameters
        ----------

        pcoords : ArrayOrFloat
            float or 1d float array of local coordinates.

        coords : numpy.ndarray, Optional
            2d array of vertex coordinates of shape (nV, nD)
            If provided, it must contain entries for all indices in `topo`.
                nV : number of points in the model
                nD : number of dimensions of the model space

        topo : (nNE) or (nE, nNE) numpy.ndarray, Optional
            1d or 2d array of vertex indices. 
            The i-th row contains the vertex indices of the i-th element.
                nE : number of elements
                nNE : number of nodes per element

        ecoords : (nNE, nD) or (nE, nNE, nD) numpy.ndarray, Optional
            2d or 3d array of element coordinates. 
                nE : number of elements
                nNE : number of nodes per element
                nD : number of dimensions of the model space

        squeeze : bool, Optional
            If True, returns the result without dummy axes.
            Default is True.

        rng : Iterable
            Bounds of the interval, where the evaluation points are understood.
            Default is [0, 1]

        Returns
        -------
        numpy.ndarray
            4d array of size (nE, nP, 4, nDOF)
                nE : number of elements
                nP : number of evaluation points
                4 : number of strain components
                nDOF : total number of degrees of freedom of an element

        Notes
        -----
        (1) If squeeze=True, the returned array can be 2d, 3d or 4d, depending
            on nE and nP.

        (2) Either `ecoords`, or both `coords` and `topo` must be provided.

        """
        rng = np.array([0., 1.]) if rng is None else np.array(rng)
        if ecoords is None:
            assert topo is not None and coords is not None
            topo = atleast2d(topo)
            ecoords = cell_coords_bulk(coords, topo)
        else:
            ecoords = atleast3d(ecoords)
        lengths = lengths_of_lines(ecoords)
        gdshp = cls.shape_function_derivatives(pcoords, rng=rng,
                                               squeeze=False,
                                               lengths=lengths)
        return cls._strain_displacement_matrix(gdshp)

    @classmethod
    @squeeze(True)
    def stiffness_matrix(cls, coords: ndarray, topo: ndarray, D: ndarray,
                         frames: ndarray = None, *args, **kwargs):
        """
        Returns the element stiffness matrices in dense format.

        Parameters
        ----------

        coords : (nP, nD) numpy.ndarray
            2d float array of all vertex coordinates. 
            It must contain entries for all indices in `topo`.
                nP : number of points in the model
                nD : number of dimensions of the model space

        topo : (nNE) or (nE, nNE) numpy.ndarray
            1d or 2d integer array of vertex indices. 
            The i-th row contains the vertex indices of the i-th element.
                nE : number of elements
                nNE : number of nodes per element

        D : (NSTRE, NSTRE) or (nE, NSTRE, NSTRE) numpy.ndarray
            2d or 3d float array of model stiffness matrices.
                nE : number of elements
                NSTRE : number of internal forces

        frames : (3, 3) or (nE, 3, 3) numpy.ndarray
            2d or  3d float array of unit basis vectors forming the 
            local coordinate systems of the elements.
                nE : number of elements

        squeeze : bool, Optional
            If True, returns the result without dummy axes.
            Default is True.

        Returns
        -------
        numpy.ndarray
            array of size (nE, nDOF, nDOF) or (nDOF, nDOF)
                nE : number of elements
                nDOF : total number of degrees of freedom of an element

        Notes
        -----
        (1) If squeeze=True, the returned array can be 2d or 3d depending
            on nE.

        """

        # prepare data for bulk evaluation
        coords = np.array(coords, dtype=float) if not isinstance(
            coords, ndarray) else coords
        topo = np.array(topo, dtype=int) if not isinstance(
            topo, ndarray) else topo
        topo = atleast2d(topo)
        D = atleast3d(D)
        if len(D) < len(topo):
            D = repeat(D[0], len(topo))
        if frames is not None:
            frames = atleast3d(frames)
            if len(frames) < len(topo):
                frames = repeat(frames[0], len(topo))

        # handle integration
        _q = kwargs.get('_q', None)
        if _q is None:
            _q = cls.quadrature[cls.qrule]
            if isinstance(_q, dict):
                N = cls.NNODE * cls.NDOFN
                nE = len(topo)
                res = np.zeros((nE, N, N))
                for qinds, qvalue in _q.items():
                    if isinstance(qvalue, str):
                        qpos, qweight = cls.quadrature[qvalue]
                    else:
                        qpos, qweight = qvalue
                    q = Quadrature(qinds, qpos, qweight)
                    res += cls.stiffness_matrix(coords,
                                                topo, D, squeeze=False, _q=q)
            else:
                qpos, qweight = cls.quadrature[cls.qrule]
                _q = Quadrature(None, qpos, qweight)
                res = cls.stiffness_matrix(
                    coords, topo, D, squeeze=False, _q=_q)
            if frames is not None:
                Q = element_transformation_matrices(frames, cls.NNODE)
                return tr_2d_out(res, Q)
            else:
                return res

        ecoords = cell_coords_bulk(coords, topo)
        dshp = cls.shape_function_derivatives(
            _q.pos, rng=cls.lcoords, squeeze=False)
        B = cls.strain_displacement_matrix(
            _q.pos, ecoords=ecoords, squeeze=False, rng=cls.lcoords)
        if _q.inds is not None:
            # zero out unused indices, only for selective integration
            inds = np.where(~np.in1d(np.arange(cls.NSTRE), _q.inds))[0]
            B[:, :, inds, :] = 0.
        jac = jacobian_matrix_bulk(dshp, ecoords)
        djac = jacobian_det_bulk(jac)
        return stiffness_matrix_bulk(D, B, djac, _q.weight)

    @classmethod
    @squeeze(True)
    def mass_matrix(cls, coords: ndarray, topo: ndarray, dens: ndarray,
                    areas: ndarray, *args, **kwargs):

        _q = kwargs.get('_q', None)
        if _q is None:
            _q = cls.quadrature[cls.qrule]
            if isinstance(_q, dict):
                N = cls.NNODE * cls.NDOFN
                nE = len(topo)
                res = np.zeros((nE, N, N))
                for qinds, qvalue in _q.items():
                    if isinstance(qvalue, str):
                        qpos, qweight = cls.quadrature[qvalue]
                    else:
                        qpos, qweight = qvalue
                    _q = Quadrature(qinds, qpos, qweight)
                    res += cls.mass_matrix(coords, topo, dens, areas, _q=_q)
            else:
                qpos, qweight = cls.quadrature[cls.qrule]
                _q = Quadrature(None, qpos, qweight)
                res = cls.mass_matrix(coords, topo, dens, areas, _q=_q)
            return res
        ecoords = cell_coords_bulk(coords, topo)
        dshp = cls._shape_function_derivatives(_q.pos)
        jac = jacobian_matrix_bulk(dshp, ecoords)
        shp = cls._shape_function_values(_q.pos)
        gdshp = gdshp_bulk(dshp, jac)
        N = cls._shape_function_matrix(shp, gdshp)
        if _q.inds is not None:
            # zero out unused indices, only for selective integration
            inds = np.where(~np.in1d(np.arange(cls.NSTRE), _q.inds))[0]
            N[:, :, inds, :] = 0.
        djac = jacobian_det_bulk(jac)
        return mass_matrix_bulk(N, dens, areas, djac, _q.weight)

    # !FIXME : check
    @classmethod
    def mass_matrix_coo(cls, coords: ndarray, topo: ndarray, dens: ndarray,
                        areas: ndarray, frames: ndarray,
                        conn: ndarray = None) -> coo:
        """
        Returns the global mass matrix in sparse coo format.

        Parameters
        ----------

        coords : (nP, nD) numpy.ndarray
            2d float array of all vertex coordinates.
                nP : number of points in the model
                nD : number of dimensions of the model space

        topo : (nNE) or (nE, nNE) numpy.ndarray
            1d or 2d integer array of vertex indices. The i-th row contains 
            the vertex indices of the i-th element.
                nE : number of elements
                nNE : number of nodes per element

        dens : (nE) numpy.ndarray
            1d float array of element densities.
                nE : number of elements

        areas : (nE) numpy.ndarray
            1d float array of element densities.
                nE : number of elements

        Returns
        -------
        scipy.sparse.coo_matrix

        """
        N = coords.shape[0] * cls.NDOFN
        Q = element_transformation_matrices(frames, cls.NNODE)
        M_bulk = tr_2d_out(cls.mass_matrix(coords, topo, dens, areas), Q)
        gnum = topo_to_gnum(topo, cls.NDOFN)
        return fem_coeff_matrix_coo(M_bulk, inds=gnum, N=N)
    
    @classmethod
    def stiffness_matrix_coo(cls, coords: ndarray, topo: ndarray,
                             D: ndarray, frames: ndarray, *args,
                             conn: ndarray = None, N: int = None, **kwargs) -> coo:
        """
        Returns the global stiffness matrix in sparse scipy-coo format.

        Parameters
        ----------

        coords : (nP, nD) numpy.ndarray
            2d float array of vertex coordinates. 
            It must contain entries for all indices in `topo`.
                nP : number of points in the model
                nD : number of dimensions of the model space

        topo : (nNE) or (nE, nNE) numpy.ndarray
            1d or 2d integer array of vertex indices. 
            The i-th row contains the vertex indices of the i-th element.
                nE : number of elements
                nNE : number of nodes per element

        D : (NSTRE, NSTRE) or (nE, NSTRE, NSTRE) numpy.ndarray
            2d or 3d float array of model stiffness matrices.
                nE : number of elements
                NSTRE : number of internal forces

        frames : (3, 3) or (nE, 3, 3) numpy.ndarray
            2d or  3d float array of unit basis vectors forming the 
            local coordinate systems of the elements.
                nE : number of elements

        N : int, Optional
            Number of total degrees of freedom of the model. Only
            required if the array `coords` is not of full length,
            otherwise it is inferred from the size of `coords`.

        Returns
        -------
        scipy.sparse.coo_matrix

        """
        # prepare data for bulk evaluation
        coords = np.array(coords, dtype=float) if not isinstance(
            coords, ndarray) else coords
        topo = np.array(topo, dtype=int) if not isinstance(
            topo, ndarray) else topo
        topo = atleast2d(topo)
        D = atleast3d(D)
        frames = atleast3d(frames)
        if len(D) < len(topo):
            D = repeat(D[0], len(topo))
        if len(frames) < len(topo):
            frames = repeat(frames[0], len(topo))
        N = coords.shape[0] * cls.NDOFN if N is None else N
        Q = element_transformation_matrices(frames, cls.NNODE)
        K_bulk = tr_2d_out(cls.stiffness_matrix(coords, topo, D, squeeze=False), Q)
        gnum = topo_to_gnum(topo, cls.NDOFN)
        return fem_coeff_matrix_coo(K_bulk, inds=gnum, N=N)

    @classmethod
    @squeeze(True)
    def body_load_vector(cls, ecoords: ndarray, values: ndarray,
                         frames: ndarray, *args,
                         topo: ndarray = None, N: int = None, **kwargs):
        """
        Turns body loads defined on elements to nodal loads. 

        If `frames` are provided, returns the nodal load values in the global system.

        If `topo` and optionally `N` is provided, returns a load vector
        with entries for all nodes in the model.

        Parameters
        ----------

        ecoords : (nNE, nD) or (nE, nNE, nD) numpy.ndarray
            2d or 3d array of element coordinates. 
                nE : number of elements
                nNE : number of nodes per element
                nD : number of dimensions of the model space

        values : (nNE, nDOF) or (nE, nNE, nDOF) numpy.ndarray
            2d or 3d array of body loads defined in the local system 
            for one or multiple elements. 
                nE : number of elements
                nNE : number of nodes per element
                nDOF : number of degrees of freedom per node

        frames : (3, 3) or (nE, 3, 3) numpy.ndarray, Optional
            2d or  3d float array of unit basis vectors forming the 
            local coordinate systems of one or more elements.
                nE : number of elements

        topo : (nNE) or (nE, nNE) numpy.ndarray, Optional
            1d or 2d integer array of vertex indices for one or more elements. 
            The i-th row contains the vertex indices of the i-th element.
                nE : number of elements
                nNE : number of nodes per element

        N : int, Optional
            Number of total degrees of freedom of the model. Only
            required if `topo` is not provided or if the provided 
            array does not contain all indices of the model,
            otherwise it is inferred from `topo`.

        squeeze : bool, Optional
            If True, returns the result without dummy axes.
            Default is True.

        Returns
        -------
        numpy.ndarray

        """
        ecoords = atleast3d(ecoords)
        lengths = kwargs.get('lengths', lengths_of_lines(ecoords))
        if topo is not None:
            topo = np.array(topo, dtype=int) if not isinstance(
                topo, ndarray) else topo
            topo = atleast2d(topo)
        frames = atleast3d(frames)
        if len(frames) < len(ecoords):
            frames = repeat(frames[0], len(ecoords))
        values = atleast3d(values)
        if len(values) < len(ecoords):
            values = repeat(values[0], len(ecoords))

        _q = kwargs.get('_q', None)
        if _q is None:
            _q = cls.quadrature[cls.qrule]
            if isinstance(_q, dict):
                N = cls.NNODE * cls.NDOFN
                nE = len(ecoords)
                res = np.zeros((nE, N))
                for qinds, qvalue in _q.items():
                    if isinstance(qvalue, str):
                        qpos, qweight = cls.quadrature[qvalue]
                    else:
                        qpos, qweight = qvalue
                    q = Quadrature(qinds, qpos, qweight)
                    res += cls.body_load_vector(ecoords, values, frames, _q=q,
                                                topo=topo, lengths=lengths,
                                                squeeze=False)
            else:
                qpos, qweight = cls.quadrature[cls.qrule]
                _q = Quadrature(None, qpos, qweight)
                res = cls.body_load_vector(ecoords, values, frames, _q=_q,
                                           topo=topo, lengths=lengths,
                                           squeeze=False)
            if frames is not None:
                Q = element_transformation_matrices(frames, cls.NNODE)
                res = tr_1d_out(res, Q)
            if topo is not None:
                gnum = topo_to_gnum(topo, cls.NDOFN)
                N = gnum.max() + 1 if N is None else N
                return assemble_load_vector(res, gnum, N)
            else:
                return res

        rng = cls.lcoords
        dshp = cls.shape_function_derivatives(_q.pos, rng=rng,
                                              squeeze=False)
        jac = jacobian_matrix_bulk(dshp, ecoords)
        shp = cls.shape_function_values(_q.pos, rng=rng, squeeze=False)
        gdshp = cls.shape_function_derivatives(_q.pos, rng=rng,
                                               squeeze=False,
                                               jac=jac)
        djac = jacobian_det_bulk(jac)
        res = cls._body_load_vector(values, shp, gdshp, djac, _q.weight)
        if _q.inds is not None:
            # zero out unused indices, only for selective integration
            inds = np.where(~np.in1d(np.arange(cls.NDOFN), _q.inds))[0]
            res[:, inds] = 0.
        return res

    @classmethod
    @squeeze(True)
    def dofsol_at(cls, pcoords: ArrayOrFloat, dofsol: ndarray,
                  ecoords: ndarray, *args, frames: ndarray = None,
                  topo: ndarray = None, return_coords=False,
                  rng=None, **kwargs) -> ndarray:
        """
        Approximates primary variables (generalized displacements)
        at one or more evaluation points of one or more elements.

        If `frames` are provided, assumes that `dofsol` refers to 
        the global system.

        Parameters
        ----------

        pcoords : ArrayOrFloat
            float or 1d float array of local coordinates.

        dofsol : (nDOF), (nTOTV) or (nE, nDOF) numpy.ndarray
            2d or 3d array of dof solutions. 
                nE : number of elements
                     used for multiple element evaluation
                nDOF : number of degrees of freedom per element
                       this is used for single element evaluation
                       (12 for a 2-node element) 
                nTOTV : number of degrees of freedom of the entire model
                        used for multiple element evaluation                   
            If `dofsol` contains entries for all degrees of freedom of the model,
            (len(topo) == nTOTV) `topo` must be provided.

        ecoords : (nNE, nD) or (nE, nNE, nD) numpy.ndarray
            2d or 3d array of element coordinates. 
                nE : number of elements
                nNE : number of nodes per element
                nD : number of dimensions of the model space

        frames : (3, 3) or (nE, 3, 3) numpy.ndarray, Optional
            2d or  3d float array of unit basis vectors forming the 
            local coordinate systems of one or more elements.
                nE : number of elements

        topo : (nNE) or (nE, nNE) numpy.ndarray, Optional
            1d or 2d integer array of vertex indices for one or more elements. 
            The i-th row contains the vertex indices of the i-th element.
            Only required if `dofsol` contains entries for all dofs in the model.
                nE : number of elements
                nNE : number of nodes per element

        N : int, Optional
            Number of total degrees of freedom of the model. Only
            required if `topo` is not provided or if the provided 
            array does not contain all indices of the model,
            otherwise it is inferred from `topo`.

        return_coords : bool, Optional
            If `True`, returns the global coordinates of the evaluation points.
            Default is False.

        rng : Iterable
            Bounds of the interval, where the evaluation points are understood.
            Default is [0, 1]

        squeeze : bool, Optional
            If True, returns the result without dummy axes.
            Default is True.

        Returns
        -------
        numpy.ndarray

        """
        rng = np.array([0., 1.]) if rng is None else np.array(rng)
        ecoords = atleast3d(ecoords)
        if topo is not None:
            topo = atleast2d(topo)
            gnum = topo_to_gnum(topo, cls.NDOFN)
            dofsol = element_dof_solution_bulk(dofsol.flatten(), gnum)
            dofsol = atleast2d(dofsol)
        else:
            raise NotImplementedError
        if frames is not None:
            frames = atleast3d(frames)
            if len(frames) < len(ecoords):
                frames = repeat(frames[0], len(ecoords))
            Q = element_transformation_matrices(frames, cls.NNODE)
            dofsol = tr_1d_in(dofsol, Q)
        N = cls.shape_function_matrix(
            pcoords, ecoords=ecoords, squeeze=False, rng=rng)
        data = approx_element_dof_solution_bulk(dofsol, N)
        if frames is not None:
            Q = nodal_transformation_matrices(frames)
            data = tr_1d_out_multi(data, Q)
        if return_coords:
            pcoords = np.array(pcoords) if isinstance(
                pcoords, list) else pcoords
            pcorods = to_range(pcoords, source=rng, target=[0., 1.])
            datacoords = pcoords_to_coords(pcorods, ecoords)
            return data, datacoords
        return data

    @classmethod
    def dofsol_at_center(cls, *args, **kwargs) -> ndarray:
        return cls.dofsol_at(0.5, *args, **kwargs)

    @classmethod
    @squeeze(True)
    def calc_internal_forces_at(cls, pcoords: ndarray, dofsol: ndarray, ecoords: ndarray,
                                D: ndarray, body_loads: ndarray = None, *args, frames: ndarray = None,
                                return_coords=False, extrapolate=True, rng=None,
                                smoothen=False, topo: ndarray = None, **kwargs) -> ndarray:
        """
        Calculates secondary variables (generalized internal forces)
        at one or more evaluation points of one or more elements.

        If `frames` are provided, assumes that `dofsol` is given wrt. to 
        the global frame.

        Parameters
        ----------

        pcoords : ArrayOrFloat
            float or 1d float array of local coordinates.

        dofsol : (nDOF), (nTOTV) or (nE, nDOF) numpy.ndarray
            2d or 3d array of degree-of-freedom solution of one loadcase. 
                nE : number of elements
                     used for multiple element evaluation
                nDOF : number of degrees of freedom per element
                       this is used for single element evaluation 
                nTOTV : number of degrees of freedom of the entire model
                        used for multiple element evaluation                   
            If `dofsol` contains entries for all degrees of freedom of the model,
            (len(topo) == nTOTV) `topo` must be provided.

        ecoords : (nNE, nD) or (nE, nNE, nD) numpy.ndarray
            2d or 3d array of element coordinates. 
                nE : number of elements
                nNE : number of nodes per element
                nD : number of dimensions of the model space

        D : (NSTRE, NSTRE) or (nE, NSTRE, NSTRE) numpy.ndarray
            2d or 3d float array of model stiffness matrices.
                nE : number of elements
                NSTRE : number of internal forces

        body_loads : (nNE, nDOF) or (nE, nNE, nDOF) numpy.ndarray, Optional
            2d or 3d array of body loads defined in the local system 
            for one or multiple elements. It must follow the shape of
            `ecoords`.
                nE : number of elements
                nNE : number of nodes per element
                nDOF : number of degrees of freedom per node

        frames : (3, 3) or (nE, 3, 3) numpy.ndarray, Optional
            2d or  3d float array of unit basis vectors forming the 
            local coordinate systems of one or more elements.
                nE : number of elements

        topo : (nNE) or (nE, nNE) numpy.ndarray, Optional
            1d or 2d integer array of vertex indices for one or more elements. 
            The i-th row contains the vertex indices of the i-th element.
            Only required if `dofsol` contains entries for all dofs in the model.
                nE : number of elements
                nNE : number of nodes per element

        return_coords : bool, Optional
            If `True`, returns the global coordinates of the evaluation points.
            Default is False.

        rng : Iterable, Optional
            Bounds of the interval, where the evaluation points are understood.
            Default is [0, 1]

        extrapolate : bool, Optional
            If `True`, the internal forces are extrapolated from the Gauss 
            integration points, otherwise they are calculated directly from
            the displacement solution.
            Default is True.

        smoothen : bool, Optional
            If `True`, smoothens the extrapolated nodal values.
            It has no effect if `extrapolate` is `False`.
            Default is False.

        squeeze : bool, Optional
            If True, returns the result without dummy axes.
            Default is True.

        Returns
        -------
        numpy.ndarray (nDOF, nE, nNE)
            nDOF : number of degrees of freedom per node
            nE : number of elements
            nNE : number of nodes per element
            
        """
        # prepare data for bulk evaluation
        pcoords = atleast1d(np.array(pcoords) if isinstance(
            pcoords, list) else pcoords)
        ecoords = atleast3d(ecoords)
        frames = atleast3d(frames)
        if len(frames) < len(ecoords):
            frames = repeat(frames[0], len(ecoords))
        D = atleast3d(D)
        if len(D) < len(ecoords):
            D = repeat(D[0], len(ecoords))
        lengths = kwargs.get('lengths', lengths_of_lines(ecoords))

        # handle extrapolation and smoothing
        gauss = kwargs.get('gauss', None)
        if extrapolate:
            gauss = cls.quadrature[cls.qrule] if gauss is None else gauss
            if isinstance(gauss, dict):
                nE = len(ecoords)
                nP = len(pcoords)
                cell_ndata = np.zeros((6, nE, nP))
                for qinds, qvalue in gauss.items():
                    if isinstance(qvalue, str):
                        qpos, qweight = cls.quadrature[qvalue]
                    else:
                        qpos, qweight = qvalue
                    gauss = Quadrature(qinds, qpos, qweight)
                    gdata = cls.calc_internal_forces_at(gauss.pos, dofsol, ecoords, D, body_loads,
                                                        frames=frames, return_coords=False, smoothen=False,
                                                        gauss=gauss, extrapolate=False, rng=[-1, 1],
                                                        squeeze=False, topo=topo, lengths=lengths)
                    approx = extrapolate_gauss_data(gauss.pos, gdata)
                    cell_ndata += approx(cls.lcoords)
            else:
                qpos, qweight = cls.quadrature[cls.qrule]
                gauss = Quadrature(None, qpos, qweight)

                # get cell data at the gauss points
                gdata = cls.calc_internal_forces_at(gauss.pos, dofsol, ecoords, D, body_loads,
                                                    frames=frames, return_coords=False, smoothen=False,
                                                    gauss=gauss, extrapolate=False, rng=[-1, 1],
                                                    squeeze=False, topo=topo, lengths=lengths)
                # interpolate gauss data to the nodes of the cells
                approx = extrapolate_gauss_data(gauss.pos, gdata)
                cell_ndata = approx(cls.lcoords)

            if smoothen:
                assert frames is not None, "Frames must be provided for smoothing!"
                assert topo is not None, "Topology of the entire model must be \
                    provided for smoothing."
                N = (topo.max() + 1) * cls.NDOFN
                # inperpolate with fitting polynomial of smoothed nodal values
                lengths = lengths_of_lines(ecoords)
                lengths[:] = 1.0
                ndf = nodal_distribution_factors(topo, lengths)
                # we need to transform all values to the global frame, so we can add them
                cell_ndata = swap(swap(cell_ndata, 0, 1), 1, 2)  # (nDOFN, nE, nP) --> (nE, nP, nDOFN)
                cell_ndata[:, :, :3] = tr_cell_gauss_out(ascont(cell_ndata[:, :, :3]), frames)
                cell_ndata[:, :, 3:] = tr_cell_gauss_out(ascont(cell_ndata[:, :, 3:]), frames)
                # the following line calculates nodal values as the weighted average of cell-nodal data,
                # the weights are the nodal distribution factors
                cell_ndata = collect_nodal_data(cell_ndata, topo, ndf=ndf, N=N)
                # the following line redistributes nodal data to the cells (same nodal value for neighbouring cells)
                cell_ndata = distribute_nodal_data(cell_ndata, topo)
                # transform all values back to the local frames of the elements
                cell_ndata[:, :, :3] = tr_cell_gauss_in(ascont(cell_ndata[:, :, :3]), frames)
                cell_ndata[:, :, 3:] = tr_cell_gauss_in(ascont(cell_ndata[:, :, 3:]), frames)
                cell_ndata = swap(swap(cell_ndata, 0, 2), 1, 2)  # (nE, nP, nDOFN) --> (nDOFN, nE, nP) 
                # extrapolate
                approx = extrapolate_gauss_data(cls.lcoords, cell_ndata)
                pcoords = to_range(pcoords, source=rng, target=[-1., 1.])
                rng = [-1., 1.]
                cell_ndata = approx(pcoords)
            else:
                # extrapolate directly from gauss data
                approx = extrapolate_gauss_data(cls.lcoords, cell_ndata)
                pcoords = to_range(pcoords, source=rng, target=[-1., 1.])
                rng = [-1., 1.]
                cell_ndata = approx(pcoords)

            # return global coordinates of evaluation points
            if not return_coords:
                return cell_ndata
            else:
                pcoords = to_range(pcoords, source=rng, target=[0., 1.])
                return cell_ndata, pcoords_to_coords(pcoords, ecoords)

        # the remaining part is only called from inside the function, or with
        # `extrapolate=False`
        rng = np.array([0., 1.]) if rng is None else np.array(rng)

        # calculate global dofsol for each evaluation point and element
        gdshp = cls.shape_function_derivatives(pcoords, rng=rng,
                                               squeeze=False,
                                               lengths=lengths)
        B = cls._strain_displacement_matrix(gdshp)
        
        # turn dofsol into (nE, nTOTV) shape (called bulk shape later on)
        if topo is not None:
            # dofsol is provided as an 1d array and we turn it into
            # a 2d array using topological information
            topo = atleast2d(topo)
            gnum = topo_to_gnum(topo, cls.NDOFN)
            dofsol = element_dof_solution_bulk(dofsol, gnum)
        else:
            # dofsol is already provided as a 2d array
            dofsol = atleast2d(dofsol)

        # transform bulk dofsol to local frames
        if frames is not None:
            frames = atleast3d(frames)
            if len(frames) < len(ecoords):
                frames = repeat(frames[0], len(ecoords))
            Q = element_transformation_matrices(frames, cls.NNODE)
            dofsol = tr_1d_in(dofsol, Q)

        # calculate local internal forces from bulk dof solution at 
        # the points of evaluation
        if body_loads is not None:
            body_loads = atleast2d(body_loads)
        else:
            body_loads = repeat(np.zeros((2, 6)), len(ecoords))
        shp = cls.shape_function_values(pcoords, rng=rng, squeeze=False)
        data = cls._calc_element_forces_bulk(
            dofsol, B, D, shp, gdshp, body_loads)

        # handle mixed quadrature rules
        inds = None if gauss is None else gauss.inds
        if inds is not None:
            # zero out unused indices, only for selective integration
            inds = np.where(~np.in1d(np.arange(cls.NSTRE), inds))[0]
            data[inds, :, :] = 0.

        # optionally, return global coordinates of the evaluation points
        if not return_coords:
            return data
        pcoords = np.array(pcoords) if isinstance(pcoords, list) else pcoords
        pcoords = to_range(pcoords, source=rng, target=[0., 1.])
        datacoords = pcoords_to_coords(pcoords, ecoords)
        return data, datacoords

    @classmethod
    def calc_internal_forces_at_center(cls, coords: ndarray, topo: ndarray,
                                       dofsol1d: ndarray, D: ndarray, frames: ndarray,
                                       body_forces: ndarray, *args, **kwargs) -> ndarray:
        return cls.calc_internal_forces_at(0.5, coords, topo, dofsol1d, D, frames, 
                                           body_forces, *args, **kwargs)
    
    @classmethod
    @squeeze(True)
    def interpolate_internal_data(cls, edata: ndarray, pcoords: ndarray, 
                                  rng=None, return_coords=False, ecoords=None, **kwargs):
        rng = np.array([0., 1.]) if rng is None else np.array(rng)
        pcoords = atleast1d(np.array(pcoords) if isinstance(
            pcoords, list) else pcoords)
        ecoords = atleast3d(ecoords) if ecoords is not None else ecoords
        edata = atleast3d(edata)
        data = cls._interp_element_data_bulk(edata, pcoords, rng)
        if return_coords:
            assert ecoords is not None
            pcorods = to_range(pcoords, source=rng, target=[0., 1.])
            datacoords = pcoords_to_coords(pcorods, ecoords)
            return data, datacoords
        return data