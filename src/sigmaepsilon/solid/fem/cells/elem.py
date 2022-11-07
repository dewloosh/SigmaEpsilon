# -*- coding: utf-8 -*-
from collections import namedtuple, Iterable
from typing import Iterable, Union
from functools import partial

from scipy.sparse import coo_matrix
import numpy as np
from numpy import ndarray

from neumann import squeeze, config
from neumann.linalg import ReferenceFrame
from neumann.array import atleast1d, atleastnd, ascont
from neumann.utils import to_range_1d
from neumann.linalg.sparse.jaggedarray import JaggedArray
from polymesh.topo import TopologyArray

from ..preproc import fem_coeff_matrix_coo
from ..postproc import (approx_element_solution_bulk, calculate_external_forces_bulk,
                        calculate_internal_forces_bulk, explode_kinetic_strains)
from ..utils import (topo_to_gnum, assemble_load_vector, element_dof_solution_bulk, 
                     constrain_local_stiffness_bulk, assert_min_stiffness_bulk, 
                     expand_stiffness_bulk, element_dofmap_bulk, topo_to_gnum_jagged)
from ..tr import (nodal_dcm, nodal_dcm_bulk, element_dcm, element_dcm_bulk,
                  tr_element_vectors_bulk_multi as tr1d, tr_element_matrices_bulk as tr2d)
from .utils import (stiffness_matrix_bulk2, strain_displacement_matrix_bulk2,
                    unit_strain_load_vector_bulk, strain_load_vector_bulk, mass_matrix_bulk)
from .meta import FemMixin
from .celldata import CellData

Quadrature = namedtuple('QuadratureRule', ['inds', 'pos', 'weight'])


UX, UY, UZ, ROTX, ROTY, ROTZ = FX, FY, FZ, MX, MY, MZ = list(range(6))
ulabels = {UX: 'UX', UY: 'UY', UZ: 'UZ',
           ROTX: 'ROTX', ROTY: 'ROTY', ROTZ: 'ROTZ'}
flabels = {FX: 'FX', FY: 'FY', FZ: 'FZ',
           MX: 'MX', MY: 'MY', MZ: 'MZ'}
ulabel_to_int = {v: k for k, v in ulabels.items()}
flabel_to_int = {v: k for k, v in flabels.items()}


class FiniteElement(CellData, FemMixin):
    """
    Base mixin class for all element types. Functions of this class
    can be called on any class of SigmEpsilon.

    """

    def direction_cosine_matrix(self, *args, source: Union[ndarray, str] = None,
                                target: Union[ndarray, str] = None, N: int = None,
                                **kwargs) -> ndarray:
        """
        Returns the DCM matrix from local to global for all elements 
        in the block.

        Parameters
        ----------
        source : str or ReferenceFrame, Optional
            A source frame. Default is None.

        target : str or ReferenceFrame, Optional
            A target frame. Default is None.

        N : int, Optional
            Number of points. If not specified, the number of nodes is inferred from 
            the class of the instance the function is called upon. Default is None.

        Returns
        -------
        numpy.ndarray
            The dcm matrix for linear transformations from source to target.

        """
        nNE = self.__class__.NNODE if N is None else N
        nDOF = self.container.root().NDOFN
        c, r = divmod(nDOF, 3)
        assert r == 0, "The default mechanism assumes that the number of " + \
            "deegrees of freedom per node is a multiple of 3."
        ndcm = nodal_dcm_bulk(self.frames, c)
        dcm = element_dcm_bulk(ndcm, nNE, nDOF)  # (nE, nEVAB, nEVAB)
        if source is None and target is None:
            target = 'global'
        if source is not None:
            if isinstance(source, str):
                if source == 'global':
                    S = ReferenceFrame(dim=dcm.shape[-1])
                    return ReferenceFrame(dcm).dcm(source=S)
            elif isinstance(source, ReferenceFrame):
                if len(source) == 3:
                    c, r = divmod(nDOF, 3)
                    assert r == 0, "The default mechanism assumes that the number of " + \
                        "deegrees of freedom per node is a multiple of 3."
                    s_ndcm = nodal_dcm(source.dcm(), c)
                    s_dcm = element_dcm(s_ndcm, nNE, nDOF)  # (nEVAB, nEVAB)
                    source = ReferenceFrame(s_dcm)
                return ReferenceFrame(dcm).dcm(source=source)
            else:
                raise NotImplementedError
        elif target is not None:
            if isinstance(target, str):
                if target == 'global':
                    T = ReferenceFrame(dim=dcm.shape[-1])
                    return ReferenceFrame(dcm).dcm(target=T)
            elif isinstance(target, ReferenceFrame):
                if len(target) == 3:
                    c, r = divmod(nDOF, 3)
                    assert r == 0, "The default mechanism assumes that the number of " + \
                        "deegrees of freedom per node is a multiple of 3."
                    t_ndcm = nodal_dcm(target.dcm(), c)
                    t_dcm = element_dcm(t_ndcm, nNE, nDOF)  # (nEVAB, nEVAB)
                    target = ReferenceFrame(t_dcm)
                return ReferenceFrame(dcm).dcm(target=target)
            else:
                raise NotImplementedError
        raise NotImplementedError

    def transform_coeff_matrix(self, A: ndarray, *args, invert: bool = False, 
                               **kwargs) -> ndarray:
        """
        Transforms element coefficient matrices (eg. the stiffness or the 
        mass matrix) from local to global.

        Parameters
        ----------
        *args
            Forwarded to :func:`direction_cosine_matrix`

        **kwargs
            Forwarded to :func:`direction_cosine_matrix`

        A : numpy.ndarray
            The coefficient matrix in the source frame.

        invert : bool, Optional
            If True, the DCM matrices are transposed before transformation.
            This makes this function usable in both directions.
            Default is False.

        Returns
        -------
        numpy.ndarray
            The coefficient matrix in the target frame.

        """
        # DCM from local to global
        dcm = self.direction_cosine_matrix(*args, **kwargs)
        return A if dcm is None else tr2d(A, dcm, invert)

    @squeeze(True)
    def dof_solution(self, *args, target: Union[str, ReferenceFrame] = 'local',
                     cells: Union[int, Iterable[int]] = None, points: Union[float, Iterable] = None,
                     rng: Iterable = None, flatten: bool = True, **kwargs) -> ndarray:
        """
        Returns nodal displacements for the cells, wrt. their local frames.

        Parameters
        ----------
        points : float or Iterable, Optional
                Points of evaluation. If provided, it is assumed that the given values
                are wrt. the range [0, 1], unless specified otherwise with the 'rng' parameter. 
                If not provided, results are returned for the nodes of the selected elements.
                Default is None.

        rng : Iterable, Optional
            Range where the points of evauation are understood. Default is [0, 1].

        cells : int or Iterable[int], Optional
            Indices of cells. If not provided, results are returned for all cells.
            If cells are provided, the function returns a dictionary, with the cell indices
            being the keys. Default is None.

        target : str or ReferenceFrame, Optional
            Reference frame for the output. A value of None or 'local' refers to the local system
            of the cells. Default is 'local'.

        Returns
        -------
        numpy.ndarray
            An array of shape (nE, nEVAB, nRHS) if 'flatten' is True else (nE, nNE, nDOF, nRHS).

        """
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id)
            cells = atleast1d(cells[conds])
            assert len(cells) > 0, "Length of cells is zero!"
        else:
            cells = np.s_[:]

        dofsol = self.pointdata.dofsol
        dofsol = atleastnd(dofsol, 3, back=True)
        nP, nDOF, nRHS = dofsol.shape
        dofsol = dofsol.reshape(nP * nDOF, nRHS)
        gnum = self.global_dof_numbering()[cells]

        # transform values to cell-local frames
        dcm = self.direction_cosine_matrix(source='global')[cells]
        values = element_dof_solution_bulk(dofsol, gnum)  # (nE, nEVAB, nRHS)
        values = ascont(np.swapaxes(values, 1, 2))  # (nE, nRHS, nEVAB)
        values = tr1d(values, dcm)
        values = ascont(np.swapaxes(values, 1, 2))  # (nE, nEVAB, nRHS)

        if points is None:
            points = np.array(self.lcoords()).flatten()
            rng = [-1, 1]
        else:
            rng = np.array([0, 1]) if rng is None else np.array(rng)

        # approximate at points
        # values -> (nE, nEVAB, nRHS)
        points, rng = to_range_1d(
            points, source=rng, target=[-1, 1]).flatten(), [-1, 1]

        N = self.shape_function_matrix(points, rng=rng)
        # N -> (nP, nDOF, nDOF * nNODE)
        values = ascont(np.swapaxes(values, 1, 2))  # (nE, nRHS, nEVAB)
        values = approx_element_solution_bulk(values, N)
        # values -> (nE, nRHS, nP, nDOF)

        if target is not None:
            # transform values to a destination frame, otherwise return
            # the results in the local frames of the cells
            if isinstance(target, str) and target == 'local':
                pass
            else:
                nE, nRHS, nP, nDOF = values.shape
                values = values.reshape(nE, nRHS, nP * nDOF)
                dcm = self.direction_cosine_matrix(N=nP, target=target)[cells]
                values = tr1d(values, dcm)
                values = values.reshape(nE, nRHS, nP, nDOF)

        values = np.moveaxis(values, 1, -1)  # (nE, nP, nDOF, nRHS)

        if flatten:
            nE, nP, nDOF, nRHS = values.shape
            values = values.reshape(nE, nP * nDOF, nRHS)

        # values -> (nE, nP, nDOF, nRHS) or (nE, nP * nDOF, nRHS)
        return values

    @squeeze(True)
    def strains(self, *args, cells: Union[int, Iterable[int]] = None, rng: Iterable = None,
                points: Union[float, Iterable] = None, **kwargs) -> ndarray:
        """
        Returns strains for one or more cells.

        Parameters
        ----------
        points : float or Iterable[float], Optional
                Points of evaluation. If provided, it is assumed that the given values
                are wrt. the range [0, 1], unless specified otherwise with the 'rng' parameter. 
                If not provided, results are returned for the nodes of the selected elements.
                Default is None.

        rng : Iterable, Optional
            Range where the points of evauation are understood. Only for 1d cells.
            Default is [0, 1].

        cells : int or Iterable[int], Optional
            Indices of cells. If not provided, results are returned for all cells.
            Default is None.

        Returns
        -------
        numpy.ndarray
            An array of shape (nE, nP, nSTRE, nRHS). 

        """
        dofsol = self.dof_solution(squeeze=False, flatten=True, cells=cells)
        # (nE, nEVAB, nRHS)
        
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id)
            cells = atleast1d(cells[conds])
            assert len(cells) > 0, "Length of cells is zero!"
        else:
            cells = np.s_[:]

        if points is None:
            points = np.array(self.lcoords()).flatten()
            rng = [-1, 1]
        else:
            rng = np.array([0, 1]) if rng is None else np.array(rng)
        points, rng = to_range_1d(
            points, source=rng, target=[-1, 1]).flatten(), [-1, 1]
        
        dshp = self.shape_function_derivatives(points, rng=rng)[cells]
        ecoords = self.local_coordinates()[cells]
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)
        # (nE, nP, nD, nD)
        B = self.strain_displacement_matrix(dshp=dshp, jac=jac)
        # (nE, nP, nSTRE, nEVAB)
        dofsol = ascont(np.swapaxes(dofsol, 1, 2))  # (nE, nRHS, nEVAB)
        strains = approx_element_solution_bulk(dofsol, B)  # (nE, nRHS, nP, nSTRE)
        strains = ascont(np.moveaxis(strains, 1, -1))   # (nE, nP, nSTRE, nRHS)
        return strains

    @squeeze(True)
    def kinetic_strains(self, *args, cells: Union[int, Iterable[int]] = None, 
                        points: Union[float, Iterable] = None, **kwargs) -> ndarray:
        """
        Returns kinetic strains for one or more cells.

        Parameters
        ----------
        points : float or Iterable, Optional
                 Points of evaluation. If provided, it is assumed that the given values
                 are wrt. the range [0, 1], unless specified otherwise with the 'rng' 
                 parameter. If not provided, results are returned for the nodes of the 
                 selected elements. Default is None.

        rng : Iterable, Optional
            Range where the points of evauation are understood. Default is [0, 1].

        cells : int or Iterable[int], Optional
            Indices of cells. If not provided, results are returned for all cells.
            Default is None.

        Returns
        -------
        numpy.ndarray 
            An array of shape (nE, nSTRE, nRHS), where nE is the number of cells,
            nSTRE is the number of strain components and nRHS is the number of 
            load cases.

        """
        key = self.__class__._attr_map_['strain-loads']

        try:
            sloads = self.db[key].to_numpy()
        except Exception as e:
            if key not in self.db.fields:
                nE = len(self)
                nodal_loads = self.pointdata.loads
                if len(nodal_loads.shape) == 2:
                    nRHS = 1
                else:
                    nRHS = self.pointdata.loads.shape[-1]
                nSTRE = self.__class__.NSTRE
                sloads = np.zeros((nE, nSTRE, nRHS))
            else:
                raise e
        finally:
            sloads = atleastnd(sloads, 3, back=True)  # (nE, nSTRE=4, nRHS)

        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id)
            cells = atleast1d(cells[conds])
            if len(cells) == 0:
                return {}
        else:
            cells = np.s_[:]

        if isinstance(points, Iterable):
            nP = len(points)
            return explode_kinetic_strains(sloads[cells], nP)
        else:
            return sloads[cells]

    @squeeze(True)
    def external_forces(self, *args, cells: Union[int, Iterable[int]] = None, flatten: bool = True,
                        target: Union[str, ReferenceFrame] = 'local', **kwargs) -> ndarray:
        """
        Evaluates :math:`\mathbf{f}_e = \mathbf{K}_e @ \mathbf{u}_e` for one or several 
        cells and load cases.

        Parameters
        ----------
        cells : int or Iterable[int], Optional
            Indices of cells. If not provided, results are returned for all cells.
            Default is None.

        target : Union[str, ReferenceFrame], Optional
            The target frame. Default is 'local', which means that the returned forces 
            should be understood as coordinates of generalized vectors in the local 
            frames of the cells.

        flatten: bool, Optional
            Determines the shape of the resulting array. Default is True.

        Returns
        -------
        numpy.ndarray
            An array of shape (nE, nP * nDOF, nRHS) if 'flatten' is True else (nE, nP, nDOF, nRHS).

        """
        dofsol = self.dof_solution(squeeze=False, flatten=True, cells=cells)
        # (nE, nNE * nDOF, nRHS)
        
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id)
            cells = atleast1d(cells[conds])
            assert len(cells) > 0, "Length of cells is zero!"
        else:
            cells = np.s_[:]

        # approximation matrix
        # values -> (nE, nEVAB, nRHS)
        points = np.array(self.lcoords()).flatten()
        N = self.shape_function_matrix(points, rng=[-1, 1])[cells]
        # N -> (nE, nNE, nDOF, nDOF * nNODE)

        # calculate external forces
        dbkey = self._dbkey_stiffness_matrix_
        K = self.db[dbkey].to_numpy()
        dofsol = ascont(np.swapaxes(dofsol, 1, 2))  # (nE, nRHS, nEVAB)
        forces = calculate_external_forces_bulk(K, dofsol)
        # forces -> (nE, nRHS, nEVAB)
        forces = approx_element_solution_bulk(forces, N)
        # forces -> (nE, nRHS, nNE, nDOF)
        dofsol = ascont(np.swapaxes(dofsol, 1, 2))  # (nE, nEVAB, nRHS)
        forces = ascont(np.moveaxis(forces, 1, -1))
        # forces ->  (nE, nNE, nDOF, nRHS)

        if target is not None:
            if isinstance(target, str) and target == 'local':
                values = forces
            else:
                # transform values to a destination frame, otherwise return
                # the forces are in the local frames of the cells
                values = np.moveaxis(forces, -1, 1)
                nE, nRHS, nP, nDOF = values.shape
                values = values.reshape(nE, nRHS, nP * nDOF)
                dcm = self.direction_cosine_matrix(N=nP, target=target)[cells]
                values = tr1d(values, dcm)
                values = values.reshape(nE, nRHS, nP, nDOF)
                values = np.moveaxis(forces, 1, -1)
        else:
            values = forces

        if flatten:
            nE, nP, nX, nRHS = values.shape
            values = values.reshape(nE, nP * nX, nRHS)

        # forces : (nE, nP, nDOF, nRHS)
        return values

    @squeeze(True)
    def internal_forces(self, *args, cells: Union[int, Iterable[int]] = None, rng: Iterable = None, 
                        points: Union[float, Iterable] = None, flatten: bool = True, 
                        target: Union[str, ReferenceFrame] = 'local', **kwargs) -> ndarray:
        """
        Returns internal forces for many cells and evaluation points.

        Parameters
        ----------
        points : float or Iterable[float], Optional
                Points of evaluation. If provided, it is assumed that the given values
                are wrt. the range [0, 1], unless specified otherwise with the 'rng' parameter. 
                If not provided, results are returned for the nodes of the selected elements.
                Default is None.

        rng : Iterable[float], Optional
            Range where the points of evauation are understood. 
            Only for 1d cells. Default is [0, 1].

        cells : int or Iterable[int], Optional
            Indices of cells. If not provided, results are returned for all cells.
            Default is None.

        target : Union[str, ReferenceFrame], Optional
            The target frame. Default is 'local', which means that the returned forces should
            be understood as coordinates of generalized vectors in the local frames of the cells.

        flatten: bool, Optional
            Determines the shape of the resulting array. Default is True.

        Returns
        -------
        numpy.ndarray
            An array of shape (nE, nP * nDOF, nRHS) if 'flatten' is True else (nE, nP, nDOF, nRHS).

        """
        dofsol = self.dof_solution(squeeze=False, flatten=True, cells=cells)
        # (nE, nNE * nDOF, nRHS)
        
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id)
            cells = atleast1d(cells[conds])
            assert len(cells) > 0, "Length of cells is zero!"
        else:
            cells = np.s_[:]

        if points is None:
            points = np.array(self.lcoords()).flatten()
            rng = [-1, 1]
        else:
            if isinstance(points, Iterable):
                points = np.array(points)
            rng = np.array([0, 1]) if rng is None else np.array(rng)

        # approximate at points
        # values : (nE, nEVAB, nRHS)
        points, rng = to_range_1d(
            points, source=rng, target=[-1, 1]).flatten(), [-1, 1]
                
        dshp = self.shape_function_derivatives(points, rng=rng)[cells]
        ecoords = self.local_coordinates()[cells]
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)
        # jac -> (nE, nP, 1, 1)
        B = self.strain_displacement_matrix(dshp=dshp, jac=jac)
        # B -> (nE, nP, nSTRE, nNODE * 6)

        dofsol = ascont(np.swapaxes(dofsol, 1, 2))  # (nE, nRHS, nEVAB)
        strains = approx_element_solution_bulk(dofsol, B)
        # -> (nE, nRHS, nP, nSTRE)
        strains -= self.kinetic_strains(points=points,
                                        squeeze=False)[cells]
        D = self.model_stiffness_matrix()[cells]
        forces = calculate_internal_forces_bulk(strains, D)
        # forces -> (nE, nRHS, nP, nSTRE)
        # (nE, nP, nSTRE, nRHS)
        forces = ascont(np.moveaxis(forces, 1, -1))
        dofsol = ascont(np.swapaxes(dofsol, 1, 2))  # (nE, nEVAB, nRHS)
        # forces -> (nE, nP, nSTRE, nRHS)
        forces = self._postproc_local_internal_forces_(forces, points=points, rng=rng,
                                                       cells=cells, dofsol=dofsol)
        # forces -> (nE, nP, nDOF, nRHS)

        if target is not None:
            if isinstance(target, str) and target == 'local':
                values = forces
            else:
                # transform values to a destination frame, otherwise return
                # the forces are in the local frames of the cells
                values = np.moveaxis(forces, -1, 1)
                nE, nRHS, nP, nDOF = values.shape
                values = values.reshape(nE, nRHS, nP * nDOF)
                dcm = self.direction_cosine_matrix(N=nP, target=target)[cells]
                values = tr1d(values, dcm)
                values = values.reshape(nE, nRHS, nP, nDOF)
                values = np.moveaxis(values, 1, -1)
        else:
            values = forces

        if flatten:
            nE, nP, nSTRE, nRHS = values.shape
            values = values.reshape(nE, nP * nSTRE, nRHS)

        return values

    def global_dof_numbering(self, *args, **kwargs) -> Union[TopologyArray, ndarray]:
        """
        Returns global numbering of the degrees of freedom of the cells.        
        """
        topo = kwargs.get("topo", None)
        topo = self.topology() if topo is None else topo
        nDOFN = self.container.NDOFN
        try:
            if topo.is_jagged():
                cuts = topo.widths() * nDOFN
                data1d = np.zeros(np.sum(cuts), dtype=int)
                gnum = JaggedArray(data1d, cuts=cuts)
                topo_to_gnum_jagged(topo, gnum, nDOFN)
                return TopologyArray(gnum, to_numpy=False)
            else:
                return topo_to_gnum(topo.to_numpy(), nDOFN)
        except Exception:
            return topo_to_gnum(topo, nDOFN)

    @squeeze(True)
    def elastic_stiffness_matrix(self, *args, transform:bool=True,
                                 minval: float = 1e-12, **kwargs) -> ndarray:
        """
        Returns the elastic stiffness matrix of the cells.

        Parameters
        ----------
        transform : bool, Optional
            If True, local matrices are transformed to the global frame.
            Default is True.

        minval : float, Optional
            A minimal value for the entries in the main diagonal. Set it to a negative
            value to diable its effect. Default is 1e-12.

        Returns
        -------
        numpy.ndarray
            The stiffness matrices as a 3d numpy array, where the elements run along
            the first axis.

        """
        # build
        K = self._elastic_stiffness_matrix_(*args, transform=False, **kwargs)
        K = self._constrain_elastic_stiffness_matrix_(K)
        assert_min_stiffness_bulk(K, minval)

        # store
        dbkey = self._dbkey_stiffness_matrix_
        self.db[dbkey] = K

        # expand, transform, return
        nDOFN = self.container.NDOFN
        dofmap = self.__class__.dofmap
        if len(dofmap) < nDOFN:
            nE = K.shape[0]
            nX = nDOFN * self.NNODE
            K_ = np.zeros((nE, nX, nX), dtype=float)
            dofmap = element_dofmap_bulk(dofmap, nDOFN, self.NNODE)
            K = expand_stiffness_bulk(K, K_, dofmap)
            assert_min_stiffness_bulk(K)

        return self.transform_coeff_matrix(K) if transform else K

    @config(store_strains=False)
    def _elastic_stiffness_matrix_(self, *args, transform: bool = True, **kwargs):

        ec = kwargs.get('_ec', None)
        if ec is None:
            nSTRE = self.__class__.NSTRE
            nDOF = self.__class__.NDOFN
            nNE = self.__class__.NNODE
            nE = len(self)

            _topo = kwargs.get('_topo', self.topology().to_numpy())
            _frames = kwargs.get('_frames', self.frames)
            if _frames is not None:
                ec = self.local_coordinates(_topo=_topo, frames=_frames)
            else:
                ec = self.points_of_cells(topo=_topo)

            if kwargs.get('store_strains', False):
                dbkey = self._dbkey_strain_displacement_matrix_
                self.db[dbkey] = np.zeros((nE, nSTRE, nDOF * nNE))

        q = kwargs.get('_q', None)
        if q is None:
            # main loop
            D = self.model_stiffness_matrix()
            func = partial(self._elastic_stiffness_matrix_, _D=D, **kwargs)
            q = self.quadrature[self.qrule]
            N = nDOF * nNE
            K = np.zeros((len(self), N, N))
            if isinstance(q, dict):
                # many side loops
                for qinds, qvalue in q.items():
                    if isinstance(qvalue, str):
                        qpos, qweight = self.quadrature[qvalue]
                    else:
                        qpos, qweight = qvalue
                    q = Quadrature(qinds, qpos, qweight)
                    K += func(_q=q, _ec=ec)
            else:
                # one side loop
                qpos, qweight = self.quadrature[self.qrule]
                q = Quadrature(None, qpos, qweight)
                K = func(_q=q, _ec=ec)
            # end of main cycle
            dbkey = self._dbkey_stiffness_matrix_
            self.db[dbkey] = K
            return self.transform_coeff_matrix(K) if transform else K

        # in side loop
        dshp = self.shape_function_derivatives(q.pos)
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ec)
        B = self.strain_displacement_matrix(dshp=dshp, jac=jac)
        if q.inds is not None:
            # zero out unused indices, only for selective integration
            inds = np.where(~np.in1d(np.arange(self.NSTRE), q.inds))[0]
            B[:, :, inds, :] = 0.
        djac = self.jacobian(jac=jac)
        if kwargs.get('store_strains', False):
            # B (nE, nG, nSTRE=4, nNODE * nDOF=6)
            dbkey = self._dbkey_strain_displacement_matrix_
            _B = self.db[dbkey].to_numpy()
            _B += strain_displacement_matrix_bulk2(B, djac, q.weight)
            self.db[dbkey] = _B
        D = kwargs.get('_D', self.model_stiffness_matrix())
        return stiffness_matrix_bulk2(D, B, djac, q.weight)

    def _constrain_elastic_stiffness_matrix_(self, K: ndarray):
        """
        References
        ----------
        .. [1] Duan Jin, Li-Yun-gui "About the Finite Element 
           Analysis for Beam-Hinged Frame," Advances in Engineering 
           Research, vol. 143, pp. 231-235, 2017.

        """
        # FIXME only for line meshes at the moment
        nNE = self.__class__.NNODE
        if self.has_connectivity:
            conn = self.connectivity
            if isinstance(conn, ndarray):
                if len(conn.shape) == 3 and conn.shape[1] == 2:  # nE, 2, nDOF
                    nE, _, nDOF = conn.shape
                    factors = np.ones((nE, K.shape[-1]))
                    factors[:, :nDOF] = conn[:, 0, :]
                    factors[:, -nDOF:] = conn[:, -1, :]
                    nEVAB2 = nNE * nDOF - 0.5
                    cond = np.sum(factors, axis=1) < nEVAB2
                    i = np.where(cond)[0]
                    K[i] = constrain_local_stiffness_bulk(K[i], factors[i])
                else:
                    msg = "Unknown shape of <{}> for 'connectivity'.".format(
                        conn.shape)
                    raise NotImplementedError(msg)
        return K

    def elastic_stiffness_matrix_coo(self, *args, **kwargs) -> coo_matrix:
        """
        Returns the elastic stiffness matrix of the cells in sparse COO format.

        Returns
        -------
        scipy.sparse.coo_matrix

        """
        nP = len(self.pointdata)
        N = nP * self.NDOFN
        topo = self.topology().to_numpy()
        K_bulk = self.elastic_stiffness_matrix(*args, _topo=topo, **kwargs)
        gnum = self.global_dof_numbering(topo=topo)
        return fem_coeff_matrix_coo(K_bulk, *args, inds=gnum, N=N, **kwargs)

    @squeeze(True)
    def consistent_mass_matrix(self, *args, **kwargs) -> ndarray:
        """
        Returns the mass matrix of the cells.

        Returns
        -------
        numpy.ndarray
            A 3d numpy array, where the elements run along the first axis.

        """
        return self._consistent_mass_matrix_(*args, **kwargs)

    def _consistent_mass_matrix_(self, *args, values=None, **kwargs):
        _topo = kwargs.get('_topo', self.topology().to_numpy())
        if 'frames' not in kwargs:
            key = self.__class__._attr_map_['frames']
            if key in self.db.fields:
                frames = self.frames
            else:
                frames = None
        _ecoords = kwargs.get('_ecoords', None)
        if _ecoords is None:
            if frames is not None:
                _ecoords = self.local_coordinates(_topo=_topo, frames=frames)
            else:
                _ecoords = self.points_of_cells(topo=_topo)
        if isinstance(values, np.ndarray):
            _dens = values
        else:
            _dens = kwargs.get('_dens', self.density)

        try:
            _areas = self.areas()
        except Exception:
            _areas = np.ones_like(_dens)

        q = kwargs.get('_q', None)
        if q is None:
            q = self.quadrature['mass']
            if isinstance(q, dict):
                N = self.NNODE * self.NDOFN
                nE = len(self)
                res = np.zeros((nE, N, N))
                for qinds, qvalue in q.items():
                    if isinstance(qvalue, str):
                        qpos, qweight = self.quadrature[qvalue]
                    else:
                        qpos, qweight = qvalue
                    q = Quadrature(qinds, qpos, qweight)
                    res += self._consistent_mass_matrix_(*args, _topo=_topo,
                                                         frames=frames, _q=q,
                                                         values=_dens,
                                                         _ecoords=_ecoords,
                                                         **kwargs)
            else:
                qpos, qweight = self.quadrature['mass']
                q = Quadrature(None, qpos, qweight)
                res = self._consistent_mass_matrix_(*args, _topo=_topo,
                                                    frames=frames, _q=q,
                                                    values=_dens,
                                                    _ecoords=_ecoords,
                                                    **kwargs)

            key = self.__class__._attr_map_['M']
            self.db[key] = res
            return self.transform_coeff_matrix(res)

        rng = np.array([-1., 1.])
        dshp = self.shape_function_derivatives(q.pos)
        jac = self.jacobian_matrix(dshp=dshp, ecoords=_ecoords)
        djac = self.jacobian(jac=jac)
        N = self.shape_function_matrix(q.pos, rng=rng)
        return mass_matrix_bulk(N, _dens, _areas, djac, q.weight)

    def consistent_mass_matrix_coo(self, *args, **kwargs) -> coo_matrix:
        """
        Returns the consistent mass matrix of the cells in COO format.

        Returns
        -------
        scipy.sparse.coo_matrix

        """
        nP = len(self.pointdata)
        N = nP * self.NDOFN
        topo = self.topology().to_numpy()
        M_bulk = self.consistent_mass_matrix(*args, **kwargs)
        gnum = self.global_dof_numbering(topo=topo)
        return fem_coeff_matrix_coo(M_bulk, *args, inds=gnum, N=N, **kwargs)

    @squeeze(True)
    def strain_load_vector(self, values:ndarray=None, *args, squeeze: bool,
                           return_zeroes:bool=False, **kwargs) -> ndarray:
        """
        Generates a load vector from strain loads specified for all cells.

        Parameters
        ----------
        values : numpy.ndarray, Optional
            Strain loads as a 3d numpy array of shape (nE, nS, nRHS). 
            The array must contain values for all cells (nE), strain 
            components (nS) and load cases (nL).

        return_zeroes : bool, Optional
            Controls what happends if there are no strain loads provided.
            If True, a zero array is retured with correct shape, otherwise None. 
            Default is False.

        Returns
        -------
        numpy.ndarray
            The equivalent load vector.

        """
        nRHS = self.pointdata.loads.shape[-1]
        nSTRE = self.__class__.NSTRE
        dbkey = self.__class__._attr_map_['strain-loads']
        try:
            if values is None:
                values = self.db[dbkey].to_numpy()
        except Exception as e:
            if dbkey not in self.db.fields:
                if not return_zeroes:
                    return None
                nE = len(self)
                values = np.zeros((nE, nSTRE, nRHS))
            else:
                raise e
        finally:
            values = atleastnd(values, 3, back=True)  # (nE, nSTRE=4, nRHS)

        dbkey = self.__class__._attr_map_['B']
        if dbkey not in self.db.fields:
            self.elastic_stiffness_matrix(*args, squeeze=False,
                                          store_strains=True, **kwargs)

        B = self.db[dbkey].to_numpy()  # (nE, nSTRE=4, nNODE * nDOF=6)
        D = self.model_stiffness_matrix()  # (nE, nSTRE=4, nSTRE=4)
        BTD = unit_strain_load_vector_bulk(D, B)
        values = np.swapaxes(values, 1, 2)  # (nE, nRHS, nSTRE=4)
        nodal_loads = strain_load_vector_bulk(BTD, ascont(values))
        return self._transform_local_nodal_loads_(nodal_loads)

    @squeeze(True)
    def body_load_vector(self, values: ndarray = None, *, constant: bool = False,
                         source: Union[str, ReferenceFrame] = 'local',
                         return_zeroes: bool = False, **kwargs) -> ndarray:
        """
        Builds the equivalent discrete representation of body loads
        and returns it in either the global frame or cell-local frames.

        Parameters
        ----------
        values : numpy.ndarray, Optional
            Body load values for all cells. Default is None.

        source : numpy.ndarray, Optional
            The frame in which the provided values are to be understood.
            Default is None.

        constant : bool, Optional
            Set this True if the input represents a constant load.
            Default is False.

        assemble : bool, Optional
            If True, the returned values are in the global frame, otherwise
            they are returned in cell-local frames. Default is True.

        return_zeroes : bool, Optional
            Controls what happends if there are no strain loads provided.
            If True, a zero array is retured with correct shape, otherwise None. 
            Default is False.

        Returns
        -------
        numpy.ndarray
            The nodal load vector for all load cases as a 2d numpy array
            of shape (nX, nRHS), where nX and nRHS are the total number of unknowns of
            the structure and the number of load cases.

        """
        nRHS = self.pointdata.loads.shape[-1]
        nNE = self.__class__.NNODE
        dbkey = self.__class__._attr_map_['loads']
        try:
            if values is None:
                values = self.db[dbkey].to_numpy()
        except Exception as e:
            if dbkey not in self.db.fields:
                if not return_zeroes:
                    return None
                nE = len(self)
                values = np.zeros((nE, nNE, nDOF, nRHS))
            else:
                raise e
        finally:
            values = atleastnd(values, 4, back=True)  # (nE, nNE, nDOF, nRHS)

        # prepare data to shape (nE, nNE * nDOF, nRHS)
        if constant:
            values = atleastnd(values, 3, back=True)  # (nE, nDOF, nRHS)
            #np.insert(values, 1, values, axis=1)
            nE, nDOF, nRHS = values.shape
            values_ = np.zeros((nE, nNE, nDOF, nRHS), dtype=values.dtype)
            for i in range(nNE):
                values_[:, i, :, :] = values
            values = values_
        values = atleastnd(values, 4, back=True)  # (nE, nNE, nDOF, nRHS)
        nE, _, nDOF, nRHS = values.shape
        # (nE, nNE, nDOF, nRHS) -> (nE, nNE * nDOF, nRHS)
        values = values.reshape(nE, nNE * nDOF, nRHS)
        values = ascont(values)

        if source is not None:
            if isinstance(source, str) and source == 'local':
                pass
            else:
                raise NotImplementedError
                nE, nNE, nDOF, nRHS = values.shape
                dcm = self.direction_cosine_matrix(source=source)
                # (nE, nNE * nDOF, nRHS) -> (nE, nRHS, nNE * nDOF)
                values = np.swapaxes(values, 1, 2)
                values = ascont(values)
                values = tr_cells_1d_in_multi(values, dcm)
                # (nE, nRHS, nNE * nDOF) -> (nE, nNE * nDOF, nRHS)
                values = np.swapaxes(values, 1, 2)
                values = ascont(values)

        nodal_loads = self.integrate_body_loads(values)
        # (nE, nNE * nDOF, nRHS)
        return self._transform_local_nodal_loads_(nodal_loads)

    def _transform_local_nodal_loads_(self, nodal_loads: ndarray) -> ndarray:
        """
        Transforms discrete nodal loads to the global frame and 
        assembles the nodal load vector for multiple load cases.

        Parameters
        ----------
        nodal_loads : numpy.ndarray
            A 3d array of shape (nE, nEVAB, nRHS).

        Returns
        -------
        numpy.ndarray
            A numpy array of shape (nX, nRHS), where nX is the total
            number of unknowns in the total structure.

        """
        dcm = self.direction_cosine_matrix(target='global')
        # (nE, nNE * nDOF, nRHS) -> (nE, nRHS, nNE * nDOF)
        nodal_loads = np.swapaxes(nodal_loads, 1, 2)
        nodal_loads = ascont(nodal_loads)
        nodal_loads = tr1d(nodal_loads, dcm)
        # (nE, nRHS, nNE * nDOF) -> (nE, nNE * nDOF, nRHS)
        nodal_loads = np.swapaxes(nodal_loads, 1, 2)
        # assemble
        topo = self.topology().to_numpy()
        gnum = self.global_dof_numbering(topo=topo)
        nX = len(self.pointdata) * self.container.NDOFN
        return assemble_load_vector(nodal_loads, gnum, nX)  # (nX, nRHS)
