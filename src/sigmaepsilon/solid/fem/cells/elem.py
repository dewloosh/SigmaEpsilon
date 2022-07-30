# -*- coding: utf-8 -*-
from scipy.sparse import coo_matrix
import numpy as np
from collections import namedtuple, Iterable
from typing import Iterable
from functools import partial

from ....math import squeeze, config
from ....math.linalg import ReferenceFrame
from ....math.array import atleast1d, atleastnd, ascont
from ....math.utils import to_range
from ....mesh.utils import distribute_nodal_data, collect_nodal_data

from ..preproc import fem_coeff_matrix_coo
from ..postproc import approx_element_solution_bulk, calculate_internal_forces_bulk, \
    explode_kinetic_strains
from ..utils import topo_to_gnum, approximation_matrix, nodal_approximation_matrix, \
    nodal_compatibility_factors, compatibility_factors_to_coo, \
    compatibility_factors, penalty_factor_matrix, assemble_load_vector, \
    tr_cells_1d_in_multi, tr_cells_1d_out_multi, element_dof_solution_bulk, \
    transform_stiffness, constrain_local_stiffness_bulk, assert_min_stiffness_bulk, \
    expand_stiffness_bulk, element_dofmap_bulk
from ..tr import nodal_dcm, element_dcm

from .meta import FemMixin
from .celldata import CellData
from .utils import stiffness_matrix_bulk2, strain_displacement_matrix_bulk2, \
    unit_strain_load_vector_bulk, strain_load_vector_bulk, mass_matrix_bulk

Quadrature = namedtuple('QuadratureRule', ['inds', 'pos', 'weight'])


UX, UY, UZ, ROTX, ROTY, ROTZ = FX, FY, FZ, MX, MY, MZ = list(range(6))
ulabels = {UX: 'UX', UY: 'UY', UZ: 'UZ',
           ROTX: 'ROTX', ROTY: 'ROTY', ROTZ: 'ROTZ'}
flabels = {FX: 'FX', FY: 'FY', FZ: 'FZ',
           MX: 'MX', MY: 'MY', MZ: 'MZ'}
ulabel_to_int = {v: k for k, v in ulabels.items()}
flabel_to_int = {v: k for k, v in flabels.items()}


class FemCellData(CellData):

    _attr_map_ = {
        'K': 'K',
        'M': 'M',
        'B': 'B',
    }

    @property
    def _dbkey_stiffness_matrix_(self) -> str:
        return self.__class__._attr_map_['K']

    @property
    def _dbkey_strain_displacement_matrix_(self) -> str:
        return self.__class__._attr_map_['B']

    @property
    def _dbkey_mass_matrix_(self) -> str:
        return self.__class__._attr_map_['M']


class FiniteElement(FemCellData, FemMixin):

    def direction_cosine_matrix(self, *args, source=None, target=None,
                                N=None, **kwargs):
        """
        Returns the DCM matrix from local to global for all elements 
        in the block.

        """
        N = self.__class__.NNODE if N is None else N
        frames = self.frames  # dcm_G_L
        dcm = element_dcm(nodal_dcm(frames), N)
        if source is not None:
            if isinstance(source, str):
                if source == 'global':
                    S = ReferenceFrame(dim=dcm.shape[-1])
                    return ReferenceFrame(dcm).dcm(source=S)
            elif isinstance(source, ReferenceFrame):
                assert source.dim == 2
                return ReferenceFrame(dcm).dcm(source=source)
            else:
                raise NotImplementedError
        elif target is not None:
            if isinstance(target, str):
                if target == 'global':
                    return ReferenceFrame(dcm)
            elif isinstance(target, ReferenceFrame):
                assert target.dim == 2
                return ReferenceFrame(dcm).dcm(target=target)
            else:
                raise NotImplementedError
        return dcm

    def transform_coeff_matrix(self, A, *args, **kwargs):
        """
        Transforms element coefficient matrices (eg. the stiffness or the 
        mass matrix) from local to global.

        """
        dcm = self.direction_cosine_matrix()
        return A if dcm is None else transform_stiffness(A, dcm)

    @squeeze(True)
    def dof_solution(self, *args, target='local', cells=None, points=None,
                     rng=None, flatten=True, **kwargs):
        """
        Returns nodal displacements for the cells, wrt. their local frames.

        Parameters
        ----------
        points : scalar or array, Optional
                Points of evaluation. If provided, it is assumed that the given values
                are wrt. the range [0, 1], unless specified otherwise with the 'rng' parameter. 
                If not provided, results are returned for the nodes of the selected elements.
                Default is None.

        rng : array, Optional
            Range where the points of evauation are understood. Default is [0, 1].

        cells : integer or array, Optional.
            Indices of cells. If not provided, results are returned for all cells.
            Default is None.

        target : frame, Optional.
            Reference frame for the output. A value of None or 'local' refers to the local system
            of the cells. Default is 'local'.

        ---
        Returns
        -------
        numpy array or dictionary
            (nE, nEVAB, nRHS) if flatten else (nE, nNE, nDOF, nRHS) 

        """
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id)
            cells = atleast1d(cells[conds])
            if len(cells) == 0:
                return {}
        else:
            cells = np.s_[:]

        dofsol = self.pointdata.dofsol
        dofsol = atleastnd(dofsol, 3, back=True)
        nP, nDOF, nRHS = dofsol.shape
        dofsol = dofsol.reshape(nP * nDOF, nRHS)
        gnum = self.global_dof_numbering()[cells]
        dcm = self.direction_cosine_matrix()[cells]

        # transform values to cell-local frames
        values = element_dof_solution_bulk(dofsol, gnum)  # (nE, nEVAB, nRHS)
        values = ascont(np.swapaxes(values, 1, 2))  # (nE, nRHS, nEVAB)
        values = tr_cells_1d_in_multi(values, dcm)
        values = ascont(np.swapaxes(values, 1, 2))  # (nE, nEVAB, nRHS)

        if points is None:
            points = np.array(self.lcoords()).flatten()
            rng = [-1, 1]
        else:
            rng = np.array([0, 1]) if rng is None else np.array(rng)

        # approximate at points
        # values -> (nE, nEVAB, nRHS)
        points, rng = to_range(
            points, source=rng, target=[-1, 1]).flatten(), [-1, 1]
        N = self.shape_function_matrix(points, rng=rng)[cells]
        # N -> (nE, nP, nDOF, nDOF * nNODE)
        values = ascont(np.swapaxes(values, 1, 2))  # (nE, nRHS, nEVAB)
        values = approx_element_solution_bulk(values, N)
        # values -> (nE, nRHS, nP, nDOF)

        if target is not None:
            # transform values to a destination frame, otherwise return
            # the results in the local frames of the cells
            if isinstance(target, str):
                if target == 'local':
                    pass
                elif target == 'global':
                    nDOF = values.shape[2]
                    target = ReferenceFrame(dim=nDOF)
            if isinstance(target, ReferenceFrame):
                nE, nRHS, nP, nDOF = values.shape
                values = values.reshape(nE, nRHS, nP * nDOF)
                dcm = self.direction_cosine_matrix(N=nP)[cells]
                values = tr_cells_1d_out_multi(values, dcm)
                values = values.reshape(nE, nRHS, nP, nDOF)

        values = np.moveaxis(values, 1, -1)  # (nE, nP, nDOF, nRHS)

        if flatten:
            nE, nP, nDOF, nRHS = values.shape
            values = values.reshape(nE, nP * nDOF, nRHS)

        # values -> (nE, nP, nDOF, nRHS)
        if isinstance(cells, slice):
            # results are requested on all elements
            data = values
        elif isinstance(cells, Iterable):
            data = {c: values[i] for i, c in enumerate(cells)}
        else:
            raise TypeError(
                "Invalid data type <> for cells.".format(type(cells)))

        return data

    @squeeze(True)
    def strains(self, *args, cells=None, points=None, rng=None, separate=False,
                source='total', **kwargs):
        """
        Returns strains for the cells.

        Parameters
        ----------
        points : scalar or array, Optional
                Points of evaluation. If provided, it is assumed that the given values
                are wrt. the range [0, 1], unless specified otherwise with the 'rng' parameter. 
                If not provided, results are returned for the nodes of the selected elements.
                Default is None.

        rng : array, Optional
            Range where the points of evauation are understood. Default is [0, 1].

        cells : integer or array, Optional.
            Indices of cells. If not provided, results are returned for all cells.
            Default is None.

        ---
        Returns
        -------
        numpy array or dictionary
        (nE, nP * nSTRE, nRHS) if flatten else (nE, nP, nSTRE, nRHS) 

        """
        assert separate is False  # separate heat and force sources

        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id)
            cells = atleast1d(cells[conds])
            if len(cells) == 0:
                return {}
        else:
            cells = np.s_[:]

        dofsol = self.pointdata.dofsol
        dofsol = atleastnd(dofsol, 3, back=True)
        nP, nDOF, nRHS = dofsol. shape
        dofsol = dofsol.reshape(nP * nDOF, nRHS)
        gnum = self.global_dof_numbering()[cells]
        dcm = self.direction_cosine_matrix()[cells]
        ecoords = self.local_coordinates()[cells]

        # transform values to cell-local frames
        values = element_dof_solution_bulk(dofsol, gnum)  # (nE, nEVAB, nRHS)
        values = ascont(np.swapaxes(values, 1, 2))
        values = tr_cells_1d_in_multi(values, dcm)
        values = ascont(np.swapaxes(values, 1, 2))

        if points is None:
            points = np.array(self.lcoords()).flatten()
            rng = [-1, 1]
        else:
            rng = np.array([0, 1]) if rng is None else np.array(rng)

        # approximate at points
        # values : (nE, nEVAB, nRHS)
        points, rng = to_range(
            points, source=rng, target=[-1, 1]).flatten(), [-1, 1]

        dshp = self.shape_function_derivatives(points, rng=rng)[cells]
        jac = self.jacobian_matrix(
            dshp=dshp, ecoords=ecoords)  # (nE, nP, 1, 1)
        B = self.strain_displacement_matrix(
            dshp=dshp, jac=jac)  # (nE, nP, 4, nNODE * 6)

        values = ascont(np.swapaxes(values, 1, 2))  # (nE, nRHS, nEVAB)
        values = approx_element_solution_bulk(values, B)  # (nE, nRHS, nP, 4)
        values = ascont(np.moveaxis(values, 1, -1))   # (nE, nP, nDOF, nRHS)

        # values : (nE, nP, 4, nRHS)
        if isinstance(cells, slice):
            # results are requested on all elements
            data = values
        elif isinstance(cells, Iterable):
            data = {c: values[i] for i, c in enumerate(cells)}
        else:
            raise TypeError(
                "Invalid data type <> for cells.".format(type(cells)))

        return data

    @squeeze(True)
    def kinetic_strains(self, *args, cells=None, points=None, **kwargs):
        """
        Returns kinetic strains.

        Parameters
        ----------
        points : scalar or array, Optional
                 Points of evaluation. If provided, it is assumed that the given values
                 are wrt. the range [0, 1], unless specified otherwise with the 'rng' 
                 parameter. If not provided, results are returned for the nodes of the 
                 selected elements. Default is None.

        rng : array, Optional
            Range where the points of evauation are understood. Default is [0, 1].

        cells : integer or array, Optional.
            Indices of cells. If not provided, results are returned for all cells.
            Default is None.

        Returns
        -------
        numpy.ndarray 
            shape : (nE, nSTRE, nRHS)

        """
        key = self.__class__._attr_map_['strain-loads']
        try:
            sloads = self.db[key].to_numpy()
        except Exception as e:
            if key not in self.db.fields:
                nE = len(self)
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
    def internal_forces(self, *args, cells=None, points=None, rng=None,
                        flatten=True, target='local', **kwargs):
        """
        Returns internal forces for many cells and evaluation points.

        Parameters
        ----------
        points : scalar or array, Optional
                Points of evaluation. If provided, it is assumed that the given values
                are wrt. the range [0, 1], unless specified otherwise with the 'rng' parameter. 
                If not provided, results are returned for the nodes of the selected elements.
                Default is None.

        rng : array, Optional
            Range where the points of evauation are understood. Default is [0, 1].

        cells : integer or array, Optional.
            Indices of cells. If not provided, results are returned for all cells.
            Default is None.

        Returns
        -------
        numpy.ndarray or dictionary
            (nE, nP * nSTRE, nRHS) if flatten else (nE, nP, nSTRE, nRHS) 

        """
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id)
            cells = atleast1d(cells[conds])
            if len(cells) == 0:
                return {}
        else:
            cells = np.s_[:]

        dofsol = self.pointdata.dofsol
        dofsol = atleastnd(dofsol, 3, back=True)
        nP, nDOF, nRHS = dofsol. shape
        dofsol = dofsol.reshape(nP * nDOF, nRHS)
        gnum = self.global_dof_numbering()[cells]
        dcm = self.direction_cosine_matrix()[cells]
        ecoords = self.local_coordinates()[cells]

        # transform dofsol to cell-local frames
        dofsol = element_dof_solution_bulk(dofsol, gnum)  # (nE, nEVAB, nRHS)
        dofsol = ascont(np.swapaxes(dofsol, 1, 2))
        dofsol = tr_cells_1d_in_multi(dofsol, dcm)
        dofsol = ascont(np.swapaxes(dofsol, 1, 2))  # (nE, nEVAB, nRHS)

        if points is None:
            points = np.array(self.lcoords()).flatten()
            rng = [-1, 1]
        else:
            if isinstance(points, Iterable):
                points = np.array(points)
            rng = np.array([0, 1]) if rng is None else np.array(rng)

        # approximate at points
        # values : (nE, nEVAB, nRHS)
        points, rng = to_range(
            points, source=rng, target=[-1, 1]).flatten(), [-1, 1]

        dshp = self.shape_function_derivatives(points, rng=rng)[cells]
        jac = self.jacobian_matrix(
            dshp=dshp, ecoords=ecoords)  # (nE, nP, 1, 1)
        B = self.strain_displacement_matrix(
            dshp=dshp, jac=jac)  # (nE, nP, 4, nNODE * 6)

        dofsol = ascont(np.swapaxes(dofsol, 1, 2))  # (nE, nRHS, nEVAB)
        strains = approx_element_solution_bulk(dofsol, B)  # (nE, nRHS, nP, 4)
        strains -= self.kinetic_strains(points=points, squeeze=False)[cells]
        D = self.model_stiffness_matrix()[cells]
        forces = calculate_internal_forces_bulk(
            strains, D)  # (nE, nRHS, nP, 4)
        forces = ascont(np.moveaxis(forces, 1, -1))   # (nE, nP, 4, nRHS)
        dofsol = ascont(np.swapaxes(dofsol, 1, 2))  # (nE, nEVAB, nRHS)
        forces = self._postproc_local_internal_forces(forces, points=points, rng=rng,
                                                      cells=cells, dofsol=dofsol)

        if target is not None:
            # transform values to a destination frame, otherwise return
            # the results in the local frames of the cells
            assert target == 'local'

        if flatten:
            nE, nP, nSTRE, nRHS = forces.shape
            forces = forces.reshape(nE, nP * nSTRE, nRHS)

        # values : (nE, nP, 4, nRHS)
        if isinstance(cells, slice):
            # results are requested on all elements
            data = forces
        elif isinstance(cells, Iterable):
            data = {c: forces[i] for i, c in enumerate(cells)}
        else:
            raise TypeError(
                "Invalid data type <> for cells.".format(type(cells)))

        return data

    def approximation_matrix(self, *args, **kwargs):
        key = self.__class__._attr_map_['ndf']
        return approximation_matrix(self.db[key].to_numpy(), self.NDOFN)

    def approximation_matrix_coo(self, *args, **kwargs):
        N = len(self.pointdata) * self.NDOFN
        topo = self.topology().to_numpy()
        appr = self.approximation_matrix()
        gnum = self.global_dof_numbering(topo=topo)
        return fem_coeff_matrix_coo(appr, *args, inds=gnum, N=N, **kwargs)

    def nodal_approximation_matrix(self, *args, **kwargs):
        key = self.__class__._attr_map_['ndf']
        return nodal_approximation_matrix(self.db[key].to_numpy())

    def nodal_approximation_matrix_coo(self, *args, **kwargs):
        N = len(self.pointdata)
        topo = self.topology().to_numpy()
        appr = self.nodal_approximation_matrix()
        return fem_coeff_matrix_coo(appr, *args, inds=topo, N=N, **kwargs)

    def distribute_nodal_data(self, data, key):
        key = self.__class__._attr_map_['ndf']
        topo = self.topology().to_numpy()
        ndf = self.db[key].to_numpy()
        self.db[key] = distribute_nodal_data(data, topo, ndf)

    def collect_nodal_data(self, key, *args, N=None, **kwargs):
        topo = self.topology().to_numpy()
        N = len(self.pointdata) if N is None else N
        cellloads = self.db[key].to_numpy()
        return collect_nodal_data(cellloads, topo, N)

    def compatibility_penalty_matrix_coo(self, *args, nam_csr_tot, p=1e12, **kwargs):
        """
        Parameters
        ----------
        nam_csr_tot : nodal_approximation matrix for the whole structure
                      in csr format.  
        """
        topo = self.topology().to_numpy()
        nam = self.nodal_approximation_matrix()
        factors, nreg = nodal_compatibility_factors(nam_csr_tot, nam, topo)
        factors, nreg = compatibility_factors(factors, nreg, self.NDOFN)
        data, rows, cols = compatibility_factors_to_coo(factors, nreg)
        return coo_matrix((data*p, (rows, cols)))

    def penalty_factor_matrix(self, *args, **kwargs):
        cellfixity = self.fixity
        shp = self.shape_function_values(self.lcoords())
        return penalty_factor_matrix(cellfixity, shp)

    def penalty_matrix(self, *args, p=1e12, **kwargs):
        return self.penalty_factor_matrix() * p

    def penalty_matrix_coo(self, *args, p=1e12, **kwargs):
        nP = len(self.pointdata)
        N = nP * self.NDOFN
        topo = self.topology().to_numpy()
        K_bulk = self.penalty_matrix(*args, topo=topo, p=p, **kwargs)
        gnum = self.global_dof_numbering(topo=topo)
        return fem_coeff_matrix_coo(K_bulk, *args, inds=gnum, N=N, **kwargs)

    def global_dof_numbering(self, *args, topo=None, **kwargs):
        topo = self.topology().to_numpy() if topo is None else topo
        return topo_to_gnum(topo, self.container.NDOFN)

    @squeeze(True)
    def stiffness_matrix(self, *args, transform=True, **kwargs):
        # build
        K = self._stiffness_matrix_(*args, transform=False, **kwargs)

        # handle connectivity
        conn = self.connectivity
        if conn is not None:
            # only for line meshes
            if len(conn.shape) == 3 and conn.shape[1] == 2:  # nE, 2, nDOF
                nE, _, nDOF = conn.shape
                factors = np.ones((nE, K.shape[-1]))
                factors[:, :nDOF] = conn[:, 0, :]
                factors[:, -nDOF:] = conn[:, -1, :]
                nEVAB2 = 2*nDOF - 0.5
                cond = np.sum(factors, axis=1) < nEVAB2
                i = np.where(cond)[0]
                K[i] = constrain_local_stiffness_bulk(K[i], factors[i])
                assert_min_stiffness_bulk(K)
            else:
                raise NotImplementedError(
                    "Unknown shape of <{}> for 'connectivity'.".format(conn.shape))

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
    def _stiffness_matrix_(self, *args, transform: bool = True, **kwargs):

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
            func = partial(self._stiffness_matrix_, _D=D, **kwargs)
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

    def stiffness_matrix_coo(self, *args, **kwargs):
        nP = len(self.pointdata)
        N = nP * self.NDOFN
        topo = self.topology().to_numpy()
        K_bulk = self.stiffness_matrix(*args, _topo=topo, **kwargs)
        gnum = self.global_dof_numbering(topo=topo)
        return fem_coeff_matrix_coo(K_bulk, *args, inds=gnum, N=N, **kwargs)

    @squeeze(True)
    def mass_matrix(self, *args, **kwargs):
        return self._mass_matrix_(*args, **kwargs)

    def _mass_matrix_(self, *args, values=None, **kwargs):
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
                    res += self._mass_matrix_(*args, _topo=_topo,
                                              frames=frames, _q=q,
                                              values=_dens,
                                              _ecoords=_ecoords,
                                              **kwargs)
            else:
                qpos, qweight = self.quadrature['mass']
                q = Quadrature(None, qpos, qweight)
                res = self._mass_matrix_(*args, _topo=_topo,
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

    def mass_matrix_coo(self, *args, **kwargs):
        nP = len(self.pointdata)
        N = nP * self.NDOFN
        topo = self.topology().to_numpy()
        M_bulk = self.mass_matrix(*args, **kwargs)
        gnum = self.global_dof_numbering(topo=topo)
        return fem_coeff_matrix_coo(M_bulk, *args, inds=gnum, N=N, **kwargs)

    @squeeze(True)
    def strain_load_vector(self, values=None, *args, squeeze, **kwargs):
        nRHS = self.pointdata.loads.shape[-1]
        nSTRE = self.__class__.NSTRE
        dbkey = self.__class__._attr_map_['strain-loads']
        try:
            if values is None:
                values = self.db[dbkey].to_numpy()
        except Exception as e:
            if dbkey not in self.db.fields:
                nE = len(self)
                values = np.zeros((nE, nSTRE, nRHS))
            else:
                raise e
        finally:
            values = atleastnd(values, 3, back=True)  # (nE, nSTRE=4, nRHS)

        dbkey = self.__class__._attr_map_['B']
        if dbkey not in self.db.fields:
            self.stiffness_matrix(*args, squeeze=False,
                                  store_strains=True, **kwargs)

        B = self.db[dbkey].to_numpy()  # (nE, nSTRE=4, nNODE * nDOF=6)
        D = self.model_stiffness_matrix()  # (nE, nSTRE=4, nSTRE=4)
        BTD = unit_strain_load_vector_bulk(D, B)
        values = np.swapaxes(values, 1, 2)  # (nE, nRHS, nSTRE=4)
        nodal_loads = strain_load_vector_bulk(BTD, ascont(values))
        return self.transform_local_nodal_loads(nodal_loads)

    @squeeze(True)
    def body_load_vector(self, values=None, *args, source=None,
                         constant=False, target=None, **kwargs):
        """
        Builds the equivalent nodal node representation of body loads
        and returns it in the global frame, or an arbitrary target frame 
        if specified.

        """
        nNE = self.__class__.NNODE
        # prepare data to shape (nE, nNE * nDOF, nRHS)
        if values is None:
            values = self.loads
            values = atleastnd(values, 4, back=True)  # (nE, nNE, nDOF, nRHS)
        else:
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
        return self.transform_local_nodal_loads(nodal_loads)

    def transform_local_nodal_loads(self: 'FiniteElement', nodal_loads):
        """
        nodal_loads (nE, nNE * nDOF, nRHS) == (nE, nX, nRHS)
        (nX, nRHS)
        """
        dcm = self.direction_cosine_matrix()
        # (nE, nNE * nDOF, nRHS) -> (nE, nRHS, nNE * nDOF)
        nodal_loads = np.swapaxes(nodal_loads, 1, 2)
        nodal_loads = ascont(nodal_loads)
        nodal_loads = tr_cells_1d_out_multi(nodal_loads, dcm)
        # (nE, nRHS, nNE * nDOF) -> (nE, nNE * nDOF, nRHS)
        nodal_loads = np.swapaxes(nodal_loads, 1, 2)
        # assemble
        topo = self.topology().to_numpy()
        gnum = self.global_dof_numbering(topo=topo)
        nX = len(self.pointdata) * self.container.NDOFN
        return assemble_load_vector(nodal_loads, gnum, nX)  # (nX, nRHS)
