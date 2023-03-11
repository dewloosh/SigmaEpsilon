from collections import namedtuple
from typing import Iterable, Union
from functools import partial

from scipy.sparse import coo_matrix
import numpy as np
from numpy import ndarray

from neumann.linalg import ReferenceFrame
from neumann import atleast1d, atleastnd, ascont
from neumann.utils import to_range_1d
from neumann.linalg.sparse.jaggedarray import JaggedArray

from ...utils.fem.preproc import (
    fem_coeff_matrix_coo,
    assert_min_diagonals_bulk,
    assemble_load_vector,
    condensate_Kf_bulk,
    condensate_M_bulk,
)
from ...utils.fem.postproc import (
    approx_element_solution_bulk,
    calculate_external_forces_bulk,
    calculate_internal_forces_bulk,
    explode_kinetic_strains,
    element_dof_solution_bulk,
)
from ...utils.fem.tr import (
    nodal_dcm,
    nodal_dcm_bulk,
    element_dcm,
    element_dcm_bulk,
    tr_element_vectors_bulk_multi as tr_vectors,
    tr_element_matrices_bulk as tr2d,
)
from ...utils.fem.fem import (
    topo_to_gnum,
    expand_coeff_matrix_bulk,
    element_dofmap_bulk,
    topo_to_gnum_jagged,
    expand_load_vector_bulk,
)
from ...utils.fem.cells import (
    stiffness_matrix_bulk2,
    strain_displacement_matrix_bulk2,
    unit_strain_load_vector_bulk,
    strain_load_vector_bulk,
    mass_matrix_bulk,
    body_load_vector_bulk,
)
from .meta import FemMixin
from .celldata import CellData


Quadrature = namedtuple("QuadratureRule", ["inds", "pos", "weight"])


UX, UY, UZ, ROTX, ROTY, ROTZ = FX, FY, FZ, MX, MY, MZ = list(range(6))
ulabels = {UX: "UX", UY: "UY", UZ: "UZ", ROTX: "ROTX", ROTY: "ROTY", ROTZ: "ROTZ"}
flabels = {FX: "FX", FY: "FY", FZ: "FZ", MX: "MX", MY: "MY", MZ: "MZ"}
ulabel_to_int = {v: k for k, v in ulabels.items()}
flabel_to_int = {v: k for k, v in flabels.items()}


class FiniteElement(CellData, FemMixin):
    """
    Base mixin class for all element types. Functions of this class
    can be called on any class of SigmaEpsilon.
    """

    def direction_cosine_matrix(
        self,
        *_,
        source: Union[ndarray, str] = None,
        target: Union[ndarray, str] = None,
        N: int = None,
        **__,
    ) -> ndarray:
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
        assert r == 0, (
            "The default mechanism assumes that the number of "
            + "deegrees of freedom per node is a multiple of 3."
        )
        ndcm = nodal_dcm_bulk(self.frames, c)
        dcm = element_dcm_bulk(ndcm, nNE, nDOF)  # (nE, nEVAB, nEVAB)
        if source is None and target is None:
            target = "global"
        if source is not None:
            if isinstance(source, str):
                if source == "global":
                    S = ReferenceFrame(dim=dcm.shape[-1])
                    return ReferenceFrame(dcm).dcm(source=S)
            elif isinstance(source, ReferenceFrame):
                if len(source) == 3:
                    c, r = divmod(nDOF, 3)
                    assert r == 0, (
                        "The default mechanism assumes that the number of "
                        + "deegrees of freedom per node is a multiple of 3."
                    )
                    s_ndcm = nodal_dcm(source.dcm(), c)
                    s_dcm = element_dcm(s_ndcm, nNE, nDOF)  # (nEVAB, nEVAB)
                    source = ReferenceFrame(s_dcm)
                return ReferenceFrame(dcm).dcm(source=source)
            else:
                raise NotImplementedError
        elif target is not None:
            if isinstance(target, str):
                if target == "global":
                    T = ReferenceFrame(dim=dcm.shape[-1])
                    return ReferenceFrame(dcm).dcm(target=T)
            elif isinstance(target, ReferenceFrame):
                if len(target) == 3:
                    c, r = divmod(nDOF, 3)
                    assert r == 0, (
                        "The default mechanism assumes that the number of "
                        + "deegrees of freedom per node is a multiple of 3."
                    )
                    t_ndcm = nodal_dcm(target.dcm(), c)
                    t_dcm = element_dcm(t_ndcm, nNE, nDOF)  # (nEVAB, nEVAB)
                    target = ReferenceFrame(t_dcm)
                return ReferenceFrame(dcm).dcm(target=target)
            else:
                raise NotImplementedError
        raise NotImplementedError

    def dof_solution(
        self,
        *_,
        target: Union[str, ReferenceFrame] = "local",
        cells: Union[int, Iterable[int]] = None,
        points: Union[float, Iterable] = None,
        rng: Iterable = None,
        flatten: bool = True,
        **__,
    ) -> ndarray:
        """
        Returns nodal displacements for the cells, wrt. their local frames.

        Parameters
        ----------
        points : float or Iterable, Optional
                Points of evaluation. If provided, it is assumed that the given values
                are wrt. the range [0, 1], unless specified otherwise with the 'rng'
                parameter. If not provided, results are returned for the nodes of the
                selected elements. Default is None.
        rng : Iterable, Optional
            Range where the points of evauation are understood. Only for 1d cells.
            Default is [0, 1].
        cells : int or Iterable[int], Optional
            Indices of cells. If not provided, results are returned for all cells.
            If cells are provided, the function returns a dictionary, with the cell
            indices being the keys. Default is None.
        target : str or ReferenceFrame, Optional
            Reference frame for the output. A value of None or 'local' refers to the
            local system of the cells. Default is 'local'.

        Returns
        -------
        numpy.ndarray
            An array of shape (nE, nEVAB, nRHS) if 'flatten' is True
            else (nE, nNE, nDOF, nRHS).
        """
        if isinstance(cells, Iterable):
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id)
            cells = atleast1d(cells[conds])
            assert len(cells) > 0, "Length of cells is zero!"
        else:
            cells = np.s_[:]
        nDIM = self.NDIM

        dofsol = self.pointdata.dofsol
        dofsol = atleastnd(dofsol, 3, back=True)
        nP, nDOF, nRHS = dofsol.shape
        dofsol = dofsol.reshape(nP * nDOF, nRHS)
        gnum = self.global_dof_numbering().to_numpy()[cells]

        # transform values to cell-local frames
        dcm = self.direction_cosine_matrix(source="global")[cells]
        values = element_dof_solution_bulk(dofsol, gnum)  # (nE, nEVAB, nRHS)
        values = ascont(np.swapaxes(values, 1, 2))  # (nE, nRHS, nEVAB)
        values = tr_vectors(values, dcm)
        values = ascont(np.swapaxes(values, 1, 2))  # (nE, nEVAB, nRHS)

        if nDIM == 1:
            if points is None:
                points = np.array(self.lcoords()).flatten()
                rng = [-1, 1]
            else:
                rng = np.array([0, 1]) if rng is None else np.array(rng)
        else:
            if points is None:
                points = np.array(self.lcoords())

        # approximate at points
        # values -> (nE, nEVAB, nRHS)
        if nDIM == 1:
            _rng = [-1, 1]
            points = to_range_1d(points, source=rng, target=_rng).flatten()
            rng = _rng
            N = self.shape_function_matrix(points, rng=rng)[cells]
        else:
            N = self.shape_function_matrix(points, expand=True)[cells]
        # N -> (nP, nDOF, nDOF * nNODE) for constant metric
        # N -> (nE, nP, nDOF, nDOF * nNODE) for variable metric
        values = ascont(np.swapaxes(values, 1, 2))  # (nE, nRHS, nEVAB)
        values = approx_element_solution_bulk(values, N)
        # values -> (nE, nRHS, nP, nDOF)

        if target is not None:
            # transform values to a destination frame, otherwise return
            # the results in the local frames of the cells
            if isinstance(target, str) and target == "local":
                pass
            else:
                nE, nRHS, nP, nDOF = values.shape
                values = values.reshape(nE, nRHS, nP * nDOF)
                dcm = self.direction_cosine_matrix(N=nP, target=target)[cells]
                values = tr_vectors(values, dcm)
                values = values.reshape(nE, nRHS, nP, nDOF)

        values = np.moveaxis(values, 1, -1)  # (nE, nP, nDOF, nRHS)

        if flatten:
            nE, nP, nDOF, nRHS = values.shape
            values = values.reshape(nE, nP * nDOF, nRHS)

        # values -> (nE, nP, nDOF, nRHS) or (nE, nP * nDOF, nRHS)
        return values

    def strains(
        self,
        *_,
        cells: Union[int, Iterable[int]] = None,
        rng: Iterable = None,
        points: Union[float, Iterable] = None,
        **__,
    ) -> ndarray:
        """
        Returns strains for one or more cells.

        Parameters
        ----------
        points : float or Iterable[float], Optional
                Points of evaluation. If provided, it is assumed that the given values
                are wrt. the range [0, 1], unless specified otherwise with the 'rng'
                parameter. If not provided, results are returned for the nodes of the
                selected elements. Default is None.
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
        dofsol = self.dof_solution(flatten=True, cells=cells)
        # (nE, nEVAB, nRHS)

        if isinstance(cells, Iterable):
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id)
            cells = atleast1d(cells[conds])
            assert len(cells) > 0, "Length of cells is zero!"
        else:
            cells = np.s_[:]
        nDIM = self.NDIM

        if nDIM == 1:
            if points is None:
                points = np.array(self.lcoords()).flatten()
                rng = [-1, 1]
            else:
                rng = np.array([0, 1]) if rng is None else np.array(rng)
            points, rng = to_range_1d(points, source=rng, target=[-1, 1]).flatten(), [
                -1,
                1,
            ]
        else:
            if points is None:
                points = np.array(self.lcoords())

        if nDIM == 1:
            shp = self.shape_function_values(points, rng=rng)[cells]
            dshp = self.shape_function_derivatives(points, rng=rng)[cells]
        else:
            shp = self.shape_function_values(points)[cells]
            dshp = self.shape_function_derivatives(points)[cells]

        ecoords = self.local_coordinates()[cells]
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)
        # (nE, nP, nD, nD)
        B = self.strain_displacement_matrix(shp=shp, dshp=dshp, jac=jac)
        # (nE, nP, nSTRE, nEVAB)
        dofsol = ascont(np.swapaxes(dofsol, 1, 2))  # (nE, nRHS, nEVAB)
        strains = approx_element_solution_bulk(dofsol, B)  # (nE, nRHS, nP, nSTRE)
        strains = ascont(np.moveaxis(strains, 1, -1))  # (nE, nP, nSTRE, nRHS)
        return strains

    def kinetic_strains(
        self,
        *_,
        cells: Union[int, Iterable[int]] = None,
        points: Union[float, Iterable] = None,
        **__,
    ) -> ndarray:
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
        key = self._dbkey_strain_loads_
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

        if isinstance(cells, Iterable):
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

    def external_forces(
        self,
        *_,
        cells: Union[int, Iterable[int]] = None,
        target: Union[str, ReferenceFrame] = "local",
        flatten: bool = True,
        **__,
    ) -> ndarray:
        """
        Evaluates :math:`\mathbf{f}_e = \mathbf{K}_e @ \mathbf{u}_e` for one
        or several cells and load cases.

        Parameters
        ----------
        cells : int or Iterable[int], Optional
            Indices of cells. If not provided, results are returned for all cells.
            Default is None.
        target : Union[str, ReferenceFrame], Optional
            The target frame. Default is 'local', which means that the returned
            forces should be understood as coordinates of generalized vectors in
            the local frames of the cells.
        flatten: bool, Optional
            Determines the shape of the resulting array. Default is True.

        Returns
        -------
        numpy.ndarray
            An array of shape (nE, nP * nDOF, nRHS) if 'flatten' is True else
            (nE, nP, nDOF, nRHS).
        """
        dofsol = self.dof_solution(flatten=True, cells=cells)
        # (nE, nNE * nDOF, nRHS)

        if isinstance(cells, Iterable):
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
            if isinstance(target, str) and target == "local":
                values = forces
            else:
                # transform values to a destination frame, otherwise return
                # the forces are in the local frames of the cells
                values = np.moveaxis(forces, -1, 1)
                nE, nRHS, nP, nDOF = values.shape
                values = values.reshape(nE, nRHS, nP * nDOF)
                dcm = self.direction_cosine_matrix(N=nP, target=target)[cells]
                values = tr_vectors(values, dcm)
                values = values.reshape(nE, nRHS, nP, nDOF)
                values = np.moveaxis(forces, 1, -1)
        else:
            values = forces

        if flatten:
            nE, nP, nX, nRHS = values.shape
            values = values.reshape(nE, nP * nX, nRHS)

        # forces : (nE, nP, nDOF, nRHS)
        return values

    def _internal_forces_(
        self,
        *_,
        cells: Union[int, Iterable[int]] = None,
        points: Union[float, Iterable] = None,
    ) -> ndarray:
        dofsol = self.dof_solution(flatten=True, cells=cells)
        # dofsol -> (nE, nNE * nDOF, nRHS)

        shp = self.shape_function_values(points)[cells]
        dshp = self.shape_function_derivatives(points)[cells]
        ecoords = self.local_coordinates()[cells]
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)
        # jac -> (nE, nP, 1, 1)
        B = self.strain_displacement_matrix(shp=shp, dshp=dshp, jac=jac)
        # B -> (nE, nP, nSTRE, nNODE * nDOFN)

        dofsol = ascont(np.swapaxes(dofsol, 1, 2))  # (nE, nRHS, nEVAB)
        strains = approx_element_solution_bulk(dofsol, B)
        # strains -> (nE, nRHS, nP, nSTRE)
        strains -= self.kinetic_strains(points=points)[cells]
        D = self.elastic_material_stiffness_matrix()[cells]

        forces = np.zeros_like(strains)
        inds = np.arange(forces.shape[-1])
        calculate_internal_forces_bulk(strains, D, forces, inds)
        # forces -> (nE, nRHS, nP, nSTRE)
        forces = ascont(np.moveaxis(forces, 1, -1))
        # forces -> (nE, nP, nSTRE, nRHS)
        return forces

    def internal_forces(
        self,
        *_,
        cells: Union[int, Iterable[int]] = None,
        rng: Iterable = None,
        points: Union[float, Iterable] = None,
        flatten: bool = True,
        target: Union[str, ReferenceFrame] = "local",
        **__,
    ) -> ndarray:
        """
        Returns internal forces for many cells and evaluation points.

        Parameters
        ----------
        points : float or Iterable[float], Optional
                Points of evaluation. If provided, it is assumed that the given values
                are wrt. the range [0, 1], unless specified otherwise with the 'rng'
                parameter. If not provided, results are returned for the nodes of the
                selected elements. Default is None.
        rng : Iterable[float], Optional
            Range where the points of evauation are understood. Only for 1d cells.
            Default is [0, 1].
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
            An array of shape (nE, nP * nDOF, nRHS) if 'flatten' is True else
            (nE, nP, nDOF, nRHS).
        """
        if isinstance(cells, Iterable):
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id)
            cells = atleast1d(cells[conds])
            assert len(cells) > 0, "Length of cells is zero!"
        else:
            cells = np.s_[:]
        nDIM = self.NDIM

        if nDIM == 1:
            if points is None:
                points = np.array(self.lcoords()).flatten()
                rng = [-1, 1]
            else:
                if isinstance(points, Iterable):
                    points = np.array(points)
                rng = np.array([0, 1]) if rng is None else np.array(rng)
            points = to_range_1d(points, source=rng, target=[-1, 1]).flatten()
            rng = [-1, 1]
        else:
            if points is None:
                points = np.array(self.lcoords())

        forces = self._internal_forces_(cells=cells, points=points)
        # forces -> (nE, nP, nSTRE, nRHS)

        if target is not None:
            if isinstance(target, str) and target == "local":
                values = forces
            else:
                # transform values to a destination frame, or return
                # the forces in the local frames of the cells
                values = np.moveaxis(forces, -1, 1)
                nE, nRHS, nP, nSTRE = values.shape
                values = values.reshape(nE, nRHS, nP * nSTRE)
                dcm = self.direction_cosine_matrix(N=nP, target=target)[cells]
                values = tr_vectors(values, dcm)
                values = values.reshape(nE, nRHS, nP, nSTRE)
                values = np.moveaxis(values, 1, -1)
        else:
            values = forces

        if flatten:
            nE, nP, nSTRE, nRHS = values.shape
            values = values.reshape(nE, nP * nSTRE, nRHS)

        return values

    def global_dof_numbering(self, *_, **kwargs) -> Union[JaggedArray, ndarray]:
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
                return gnum
            else:
                data = topo_to_gnum(topo.to_numpy(), nDOFN)
                return JaggedArray(data)
        except Exception:
            data = topo_to_gnum(topo, nDOFN)
            return JaggedArray(data)

    def elastic_stiffness_matrix(
        self,
        *,
        transform: bool = True,
        minval: float = 1e-12,
        sparse: bool = False,
        **kwargs,
    ) -> Union[ndarray, coo_matrix]:
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
        sparse : bool, Optional
            If True, the returned object is a sparse COO matrix. Default is False.

        Returns
        -------
        numpy.ndarray or scipy.sparse.coo_matrix
            A sparse SciPy matrix if 'sparse' is True, or a 3d numpy array, where the
            elements run along the first axis.
        """
        dbkey = self._dbkey_stiffness_matrix_
        if dbkey not in self.db.fields:
            K = self._elastic_stiffness_matrix_(transform=False, **kwargs)
            assert_min_diagonals_bulk(K, minval)
            self.db[dbkey] = K
        else:
            K = self.db[dbkey].to_numpy()
        # if the model has more dofs than the element
        nDOFN = self.container.NDOFN
        dofmap = self.__class__.dofmap
        if len(dofmap) < nDOFN:
            nE = K.shape[0]
            nX = nDOFN * self.NNODE
            K_ = np.zeros((nE, nX, nX), dtype=float)
            dofmap = element_dofmap_bulk(dofmap, nDOFN, self.NNODE)
            K = expand_coeff_matrix_bulk(K, K_, dofmap)
            assert_min_diagonals_bulk(K, minval)
        if transform:
            K = self._transform_coeff_matrix_(K)
        if sparse:
            assert transform, "Must be transformed for a sparse result."
            nP = len(self.pointdata)
            N = nP * self.NDOFN
            topo = self.topology().to_numpy()
            gnum = self.global_dof_numbering(topo=topo).to_numpy()
            K = fem_coeff_matrix_coo(K, inds=gnum, N=N, **kwargs)
        return K

    def _elastic_stiffness_matrix_(
        self, *, transform: bool = True, **kwargs
    ) -> ndarray:
        ec = kwargs.get("_ec", None)
        if ec is None:
            nSTRE = self.__class__.NSTRE
            nDOF = self.__class__.NDOFN
            nNE = self.__class__.NNODE
            nE = len(self)
            _topo = kwargs.get("_topo", self.topology().to_numpy())
            _frames = kwargs.get("_frames", self.frames)
            if _frames is not None:
                ec = self.local_coordinates(target=_frames)
            else:
                ec = self.points_of_cells(topo=_topo)
            dbkey = self._dbkey_strain_displacement_matrix_
            self.db[dbkey] = np.zeros((nE, nSTRE, nDOF * nNE))

        q = kwargs.get("_q", None)
        if q is None:
            D = self.elastic_material_stiffness_matrix()
            func = partial(self._elastic_stiffness_matrix_, _D=D, **kwargs)
            q = self.quadrature[self.qrule]
            N = nDOF * nNE
            K = np.zeros((len(self), N, N))
            if isinstance(q, dict):
                for qinds, qvalue in q.items():
                    if isinstance(qvalue, str):
                        qpos, qweight = self.quadrature[qvalue]
                    else:
                        qpos, qweight = qvalue
                    q = Quadrature(qinds, qpos, qweight)
                    K += func(_q=q, _ec=ec)
            else:
                qpos, qweight = self.quadrature[self.qrule]
                q = Quadrature(None, qpos, qweight)
                K = func(_q=q, _ec=ec)
            dbkey = self._dbkey_stiffness_matrix_
            self.db[dbkey] = K
            return self._transform_coeff_matrix_(K) if transform else K

        # in side loop
        shp = self.shape_function_values(q.pos)
        dshp = self.shape_function_derivatives(q.pos)
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ec)
        B = self.strain_displacement_matrix(shp=shp, dshp=dshp, jac=jac)
        if q.inds is not None:
            # zero out unused indices, only for selective integration
            inds = np.where(~np.in1d(np.arange(self.NSTRE), q.inds))[0]
            B[:, :, inds, :] = 0.0
        djac = self.jacobian(jac=jac)
        # B (nE, nG, nSTRE=4, nNODE * nDOF=6)
        dbkey = self._dbkey_strain_displacement_matrix_
        _B = self.db[dbkey].to_numpy()
        _B += strain_displacement_matrix_bulk2(B, djac, q.weight)
        self.db[dbkey] = _B
        D = kwargs.get("_D", self.elastic_material_stiffness_matrix())
        return stiffness_matrix_bulk2(D, B, djac, q.weight)

    def consistent_mass_matrix(
        self,
        *args,
        sparse: bool = False,
        transform: bool = True,
        minval: float = 1e-12,
        **kwargs,
    ) -> Union[ndarray, coo_matrix]:
        """
        Returns the stiffness-consistent mass matrix of the cells.

        Parameters
        ----------
        transform : bool, Optional
            If True, local matrices are transformed to the global frame.
            Default is True.
        minval : float, Optional
            A minimal value for the entries in the main diagonal. Set it to a
            negative value to diable its effect. Default is 1e-12.
        sparse : bool, Optional
            If True, the returned object is a sparse COO matrix.
            Default is False.

        Returns
        -------
        numpy.ndarray or scipy.sparse.coo_matrix
            A sparse SciPy matrix if 'sparse' is True, or a 3d numpy array,
            where the elements run along the first axis.
        """
        dbkey = self._dbkey_mass_matrix_
        if dbkey not in self.db.fields:
            M = self._consistent_mass_matrix_(transform=False, **kwargs)
            assert_min_diagonals_bulk(M, minval)
            self.db[dbkey] = M
        else:
            M = self.db[dbkey].to_numpy()
        # if the model has more dofs than the element
        nDOFN = self.container.NDOFN
        dofmap = self.__class__.dofmap
        if len(dofmap) < nDOFN:
            nE = M.shape[0]
            nX = nDOFN * self.NNODE
            M_ = np.zeros((nE, nX, nX), dtype=float)
            dofmap = element_dofmap_bulk(dofmap, nDOFN, self.NNODE)
            M = expand_coeff_matrix_bulk(M, M_, dofmap)
            assert_min_diagonals_bulk(M, minval)
        if transform:
            M = self._transform_coeff_matrix_(M)
        if sparse:
            nP = len(self.pointdata)
            N = nP * self.NDOFN
            topo = self.topology().to_numpy()
            gnum = self.global_dof_numbering(topo=topo).to_numpy()
            M = fem_coeff_matrix_coo(M, *args, inds=gnum, N=N, **kwargs)
        return M

    def _consistent_mass_matrix_(
        self, *args, values=None, transform: bool = True, **kwargs
    ):
        _topo = kwargs.get("_topo", self.topology().to_numpy())
        if "frames" not in kwargs:
            key = self._dbkey_frames_
            if key in self.db.fields:
                frames = self.frames
            else:
                frames = None
        _ecoords = kwargs.get("_ecoords", None)
        if _ecoords is None:
            if frames is not None:
                _ecoords = self.local_coordinates(target=frames)
            else:
                _ecoords = self.points_of_cells()
        if isinstance(values, np.ndarray):
            _dens = values
        else:
            _dens = kwargs.get("_dens", self.density)

        try:
            _areas = self.areas()
        except Exception:
            _areas = np.ones_like(_dens)

        q = kwargs.get("_q", None)
        if q is None:
            q = self.quadrature["mass"]
            func = partial(self._consistent_mass_matrix_, **kwargs)
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
                    res += func(
                        *args,
                        _topo=_topo,
                        frames=frames,
                        _q=q,
                        values=_dens,
                        _ecoords=_ecoords,
                    )
            else:
                qpos, qweight = self.quadrature["mass"]
                q = Quadrature(None, qpos, qweight)
                res = func(
                    *args,
                    _topo=_topo,
                    frames=frames,
                    _q=q,
                    values=_dens,
                    _ecoords=_ecoords,
                )
            key = self._dbkey_mass_matrix_
            self.db[key] = res
            return self._transform_coeff_matrix_(res) if transform else res

        rng = np.array([-1.0, 1.0])
        dshp = self.shape_function_derivatives(q.pos)
        jac = self.jacobian_matrix(dshp=dshp, ecoords=_ecoords)
        djac = self.jacobian(jac=jac)
        N = self.shape_function_matrix(q.pos, rng=rng)
        return mass_matrix_bulk(N, _dens, _areas, djac, q.weight)

    def load_vector(
        self, transform: bool = True, assemble: bool = False, **__
    ) -> ndarray:
        """
        Builds the equivalent nodal load vector from all sources
        and returns it in either the global frame or cell-local frames.

        Parameters
        ----------
        assemble : bool, Optional
            If True, the values are returned with a matching shape to the total
            system. Default is False.
        transform : bool, Optional
            If True, local matrices are transformed to the global frame.
            Default is True.

        See Also
        --------
        :func:`body_load_vector`
        :func:`strain_load_vector`

        Returns
        -------
        numpy.ndarray
            The nodal load vector for all load cases as a 2d numpy array
            of shape (nX, nRHS), where nX and nRHS are the total number of
            unknowns of the structure and the number of load cases.
        """
        dbkey = self._dbkey_nodal_load_vector_
        if dbkey not in self.db.fields:
            options = dict(transform=False, assemble=False, return_zeroes=True)
            f = self.body_load_vector(**options)
            f += self.strain_load_vector(**options)
            self.db[dbkey] = f
        else:
            f = self.db[dbkey].to_numpy()
        if transform:
            f = self._transform_nodal_loads_(f)
            # (nE, nRHS, nNE * nDOF) -> (nE, nNE * nDOF, nRHS)
        if assemble:
            assert transform, "Must transform before assembly."
            f = self._assemble_nodal_loads_(f)
            # (nX, nRHS)
        return f

    def strain_load_vector(
        self,
        values: ndarray = None,
        *,
        return_zeroes: bool = False,
        transform: bool = True,
        assemble: bool = False,
        **__,
    ) -> ndarray:
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
        transform : bool, Optional
            If True, local matrices are transformed to the global frame.
            Default is True.
        assemble : bool, Optional
            If True, the values are returned with a matching shape to the total
            system. Default is False.

        See Also
        --------
        :func:`load_vector`
        :func:`body_load_vector`

        Returns
        -------
        numpy.ndarray
            The equivalent load vector.
        """
        dbkey = self._dbkey_strain_loads_
        try:
            if values is None:
                values = self.db[dbkey].to_numpy()
        except Exception as e:
            if dbkey not in self.db.fields:
                if not return_zeroes:
                    return None
                if len(self.pointdata.loads.shape) == 2:
                    nRHS = 1
                else:
                    nRHS = self.pointdata.loads.shape[-1]
                nE = len(self)
                nSTRE = self.__class__.NSTRE
                values = np.zeros((nE, nSTRE, nRHS))
            else:
                raise e
        finally:
            values = atleastnd(values, 3, back=True)  # (nE, nSTRE, nRHS)

        nodal_loads = self._strain_load_vector_(values)
        # (nE, nTOTV, nRHS)

        # if the model has more dofs than the element
        nDOFN = self.container.NDOFN
        dofmap = self.__class__.dofmap
        if len(dofmap) < nDOFN:
            nE, _, nRHS = nodal_loads.shape
            nX = nDOFN * self.NNODE
            f_ = np.zeros((nE, nX, nRHS), dtype=float)
            dofmap = element_dofmap_bulk(dofmap, nDOFN, self.NNODE)
            nodal_loads = expand_load_vector_bulk(nodal_loads, f_, dofmap)

        if transform:
            nodal_loads = self._transform_nodal_loads_(nodal_loads)
            # (nE, nRHS, nNE * nDOF) -> (nE, nNE * nDOF, nRHS)

        if assemble:
            assert transform, "Must transform before assembly."
            nodal_loads = self._assemble_nodal_loads_(nodal_loads)
            # (nX, nRHS)

        return nodal_loads

    def _strain_load_vector_(self, values: ndarray) -> ndarray:
        dbkey = self._dbkey_strain_displacement_matrix_
        if dbkey not in self.db.fields:
            self.elastic_stiffness_matrix(transform=False)
        B = self.db[dbkey].to_numpy()  # (nE, nSTRE, nNODE * nDOF)
        D = self.elastic_material_stiffness_matrix()  # (nE, nSTRE, nSTRE)
        BTD = unit_strain_load_vector_bulk(D, B)  # (nE, nTOTV, nSTRE)
        # values : # (nE, nSTRE, nRHS)
        values = np.swapaxes(values, 1, 2)  # (nE, nRHS, nSTRE)
        nodal_loads = strain_load_vector_bulk(BTD, ascont(values))
        return nodal_loads  # (nE, nTOTV, nRHS)

    def body_load_vector(
        self,
        values: ndarray = None,
        *,
        constant: bool = False,
        return_zeroes: bool = False,
        transform: bool = True,
        assemble: bool = False,
        **__,
    ) -> ndarray:
        """
        Builds the equivalent discrete representation of body loads
        and returns it in either the global frame or cell-local frames.

        Parameters
        ----------
        values : numpy.ndarray, Optional
            Body load values for all cells. Default is None.
        constant : bool, Optional
            Set this True if the input represents a constant load.
            Default is False.
        assemble : bool, Optional
            If True, the values are returned with a matching shape to the total
            system. Default is False.
        return_zeroes : bool, Optional
            Controls what happends if there are no strain loads provided.
            If True, a zero array is retured with correct shape, otherwise None.
            Default is False.
        transform : bool, Optional
            If True, local matrices are transformed to the global frame.
            Default is True.

        See Also
        --------
        :func:`load_vector`
        :func:`strain_load_vector`

        Returns
        -------
        numpy.ndarray
            The nodal load vector for all load cases as a 2d numpy array
            of shape (nX, nRHS), where nX and nRHS are the total number of
            unknowns of the structure and the number of load cases.
        """
        nNE = self.__class__.NNODE
        dbkey = self._dbkey_body_loads_
        try:
            if values is None:
                values = self.db[dbkey].to_numpy()
        except Exception as e:
            if dbkey not in self.db.fields:
                if not return_zeroes:
                    return None
                if len(self.pointdata.loads.shape) == 2:
                    nRHS = 1
                else:
                    nRHS = self.pointdata.loads.shape[-1]
                nE = len(self)
                nDOF = self.__class__.NDOFN
                values = np.zeros((nE, nNE, nDOF, nRHS))
            else:
                raise e
        finally:
            values = atleastnd(values, 4, back=True)  # (nE, nNE, nDOF, nRHS)

        # prepare data to shape (nE, nNE * nDOF, nRHS)
        if constant:
            values = atleastnd(values, 3, back=True)  # (nE, nDOF, nRHS)
            # np.insert(values, 1, values, axis=1)
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
        nodal_loads = self.integrate_body_loads(values)
        # (nE, nNE * nDOF, nRHS)

        # if the model has more dofs than the element
        nDOFN = self.container.NDOFN
        dofmap = self.__class__.dofmap
        if len(dofmap) < nDOFN:
            nE, _, nRHS = nodal_loads.shape
            nX = nDOFN * self.NNODE
            f_ = np.zeros((nE, nX, nRHS), dtype=float)
            dofmap = element_dofmap_bulk(dofmap, nDOFN, self.NNODE)
            nodal_loads = expand_load_vector_bulk(nodal_loads, f_, dofmap)

        if transform:
            nodal_loads = self._transform_nodal_loads_(nodal_loads)
            # (nE, nRHS, nNE * nDOF) -> (nE, nNE * nDOF, nRHS)

        if assemble:
            assert transform, "Must transform before assembly."
            nodal_loads = self._assemble_nodal_loads_(nodal_loads)
            # (nX, nRHS)

        return nodal_loads

    def integrate_body_loads(self, values: ndarray) -> ndarray:
        """
        Returns nodal representation of body loads.

        Parameters
        ----------
        values : numpy.ndarray
            2d or 3d numpy float array of material densities of shape
            (nE, nNE * nDOF, nRHS) or (nE, nNE * nDOF), where nE, nNE,
            nDOF and nRHS stand for the number of elements, nodes per
            element, number of degrees of freedom and number of load cases
            respectively.

        Returns
        -------
        numpy.ndarray
            An array of shape (nE, nNE * 6, nRHS).

        Notes
        -----
        1) The returned array is always 3 dimensional, even if there is only one
        load case.
        2) Reimplemented for elements with Hermite basis functions.

        See Also
        --------
        :func:`~body_load_vector_bulk`
        """
        values = atleastnd(values, 3, back=True)
        # (nE, nNE * nDOF, nRHS) -> (nE, nRHS, nNE * nDOF)
        values = np.swapaxes(values, 1, 2)
        values = ascont(values)
        qpos, qweights = self.quadrature["full"]
        N = self.shape_function_matrix(qpos)
        # (nP, nDOF, nDOF * nNE)
        dshp = self.shape_function_derivatives(qpos)
        # (nP, nNE==nSHP, nD)
        ecoords = self.local_coordinates()
        # (nE, nNE, nD)
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)
        # (nE, nP, nD, nD)
        djac = self.jacobian(jac=jac)
        # (nE, nG)
        f = body_load_vector_bulk(values, N, djac, qweights)
        # (nE, nEVAB, nRHS)
        return f

    def condensate(self):
        """
        Applies static condensation to account for cell fixity.

        References
        ----------
        .. [1] Duan Jin, Li-Yun-gui "About the Finite Element
           Analysis for Beam-Hinged Frame," Advances in Engineering
           Research, vol. 143, pp. 231-235, 2017.
        """
        if not self.has_fixity:
            return
        else:
            fixity = self.fixity
            nE, nNE, nDOF = fixity.shape
            fixity = fixity.reshape(nE, nNE * nDOF)
        dbkey_K = self._dbkey_stiffness_matrix_
        dbkey_M = self._dbkey_mass_matrix_
        dbkey_f = self._dbkey_nodal_load_vector_
        K = self.db[dbkey_K].to_numpy()
        f = self.db[dbkey_f].to_numpy()
        nEVAB_full = nNE * nDOF - 0.001
        cond = np.sum(fixity, axis=1) < nEVAB_full
        i = np.where(cond)[0]
        K[i], f[i] = condensate_Kf_bulk(K[i], f[i], fixity[i])
        assert_min_diagonals_bulk(K, 1e-12)
        self.db[dbkey_K] = K
        self.db[dbkey_f] = f
        if dbkey_M in self.db.fields:
            M = self.db[dbkey_M].to_numpy()
            M[i] = condensate_M_bulk(M[i], fixity[i])
            assert_min_diagonals_bulk(M, 1e-12)
            self.db[dbkey_M] = M

    def _transform_coeff_matrix_(
        self, A: ndarray, *args, invert: bool = False, **kwargs
    ) -> ndarray:
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

    def _transform_nodal_loads_(self, nodal_loads: ndarray) -> ndarray:
        """
        Transforms discrete nodal loads to the global frame.

        Parameters
        ----------
        nodal_loads : numpy.ndarray
            A 3d array of shape (nE, nEVAB, nRHS).

        Returns
        -------
        numpy.ndarray
            A numpy array of shape (nE, nEVAB, nRHS).
        """
        dcm = self.direction_cosine_matrix(target="global")
        # (nE, nNE * nDOF, nRHS) -> (nE, nRHS, nNE * nDOF)
        nodal_loads = np.swapaxes(nodal_loads, 1, 2)
        nodal_loads = ascont(nodal_loads)
        nodal_loads = tr_vectors(nodal_loads, dcm)
        nodal_loads = np.swapaxes(nodal_loads, 1, 2)
        # (nE, nRHS, nNE * nDOF) -> (nE, nNE * nDOF, nRHS)
        return nodal_loads

    def _assemble_nodal_loads_(self, nodal_loads: ndarray) -> ndarray:
        """
        Assembles the nodal load vector for multiple load cases.

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
        topo = self.topology().to_numpy()
        gnum = self.global_dof_numbering(topo=topo).to_numpy()
        nX = len(self.pointdata) * self.container.NDOFN
        return assemble_load_vector(nodal_loads, gnum, nX)  # (nX, nRHS)
