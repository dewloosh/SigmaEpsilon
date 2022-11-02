# -*- coding: utf-8 -*-
from typing import Union, List

from scipy.sparse import coo_matrix
import numpy as np

from neumann import squeeze
from neumann.array import atleast3d

from polymesh import PolyData

from .pointdata import PointData
from .cells.celldata import CellData
from .preproc import fem_load_vector, fem_penalty_matrix_coo, fem_nodal_mass_matrix_coo
from .cells import TET4, TET10, H8, H27


class FemMesh(PolyData):
    """
    A descendant of :class:`polymesh.PolyData` to handle polygonal meshes for 
    the **Finite Element Method**.

    Parameters
    ----------
    pd : polymesh.PolyData or polymesh.CellData, Optional
        A PolyData or a CellData instance. Dafault is None.

    cd : polymesh.CellData, Optional
        A CellData instance, if the first argument is provided. Dafault is None.

    coords : numpy.ndarray, Optional
        2d numpy array of floats, describing a pointcloud. Default is None.

    topo : numpy.ndarray, Optional
        2d numpy array of integers, describing the topology of a polygonal mesh. 
        Default is None.

    celltype : int, Optional
        An integer spcifying a valid celltype.

    Note
    ----
    The suggested way of usage is to create the pointdata and celldata
    objects in advance, and provide them as the firts argument when building the model.
    See below for how it works!

    Examples
    --------
    >>> from sigmaepsilon.solid import FemMesh
    >>> from polymesh.grid import grid
    >>> size = Lx, Ly, Lz = 100, 100, 100
    >>> shape = nx, ny, nz = 10, 10, 10
    >>> coords, topo = grid(size=size, shape=shape, eshape='H27')
    >>> pd = FemMesh(coords=coords)
    >>> pd['A']['Part1'] = FemMesh(topo=topo[:10])
    >>> pd['B']['Part2'] = FemMesh(topo=topo[10:-10])
    >>> pd['C']['Part3'] = FemMesh(topo=topo[-10:])

    See also
    --------
    :class:`polymesh.PolyData`
    :class:`polymesh.CellData`

    """

    dofs = ('UX', 'UY', 'UZ', 'ROTX', 'ROTY', 'ROTZ')
    NDOFN = 6
    _point_class_ = PointData

    def __init__(self, *args, model=None, fixity=None, loads=None, body_loads=None,
                 strain_loads=None, t=None, density=None, mass=None,
                 cell_fields=None, point_fields=None, activity=None,
                 connectivity=None, **kwargs):
        # fill up data objects with obvious data
        pkeys = self.__class__._point_class_._attr_map_
        point_fields = {} if point_fields is None else point_fields
        point_fields[pkeys['loads']] = loads
        point_fields[pkeys['mass']] = mass
        point_fields[pkeys['fixity']] = fixity
        point_fields[pkeys['loads']] = loads
        ckeys = CellData._attr_map_
        cell_fields = {} if cell_fields is None else cell_fields
        cell_fields[ckeys['loads']] = body_loads
        cell_fields[ckeys['strain-loads']] = strain_loads
        cell_fields[ckeys['density']] = density
        cell_fields[ckeys['t']] = t
        cell_fields[ckeys['connectivity']] = connectivity
        super().__init__(*args, point_fields=point_fields,
                         cell_fields=cell_fields, **kwargs)
        # nodal data can only be provided for the root object
        if not self.is_root():
            assert loads is None, "At object creation, nodal loads can only \
                be provided at the top level."
            assert fixity is None, "At object creation, fixity information \
                can only be provided at the top level."
            assert mass is None, "At object creation, nodal masses can only \
                be provided at the top level."
        # it is determined by the size of `activity` whether it refers to
        # cells or points
        if isinstance(activity, np.ndarray):
            nA = activity.shape[0]
            if self.celldata is not None:
                N = len(self.celldata)
                if nA == N:
                    self.celldata.activity = activity
                    nA = -1
            if nA > 0 and self.pointdata is not None:
                N = len(self.pointdata)
                if nA == N:
                    self.pointdata.activity = activity
                    nA = -1
            assert nA < 0
        self._model = model

    def cells_coords(self, *args, points=None, cells=None, **kwargs) \
            -> Union[np.ndarray, List[np.ndarray]]:
        """
        Returns the coordinates of the cells as a 3d numpy array if the 
        subblocks form a regular mesh, or a list of such arrays.
        """
        if points is None and cells is None:
            return super().cells_coords(*args, **kwargs)
        else:
            blocks = self.cellblocks(inclusive=True)
            kwargs.update(points=points, squeeze=False)
            if cells is not None:
                kwargs.update(cells=cells)
                res = {}
                def foo(b): return b.celldata.coords(*args, **kwargs)
                [res.update(d) for d in map(foo, blocks)]
                return res
            else:
                def foo(b): return b.celldata.coords(*args, **kwargs)
                return np.vstack(list(map(foo, blocks)))

    def element_dof_numbering(self, *args, **kwargs) -> np.ndarray:
        """
        Returns global ids of the local degrees of freedoms for each cell.
        """
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.global_dof_numbering()
        return np.vstack(list(map(foo, blocks)))

    def elastic_stiffness_matrix(self, *args, sparse=True, 
                                 **kwargs) -> Union[np.ndarray, coo_matrix]:
        """
        Returns the elastic stiffness matrix in dense or sparse format.

        Parameters
        ----------
        sparse : bool, Optional
            Returns the matrix in a sparse format. Default is True.

        Returns
        -------
        :class:`numpy.ndarray` or :class:`scipy.sparse.coo_matrix`

        """
        if sparse:
            return self.elastic_stiffness_matrix_coo(*args, **kwargs)
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.elastic_stiffness_matrix()
        return np.vstack(list(map(foo, blocks)))

    def elastic_stiffness_matrix_coo(self, *args, eliminate_zeros=True,
                                     sum_duplicates=True, **kwargs):
        """
        Elastic stiffness matrix in coo format.

        Parameters
        ----------
        eliminate_zeros : bool, Optional
            Eliminates zero entries. Default is True.

        sum_duplicates : bool, Optional
            Sums duplicate entries. Default is True.

        Returns
        -------
        :class:`scipy.sparse.coo_matrix`
            The stiffness matrix in sparse COO format.

        """
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.elastic_stiffness_matrix_coo()
        K = np.sum(list(map(foo, blocks))).tocoo()
        if eliminate_zeros:
            K.eliminate_zeros()
        if sum_duplicates:
            K.sum_duplicates()
        return K

    def masses(self, *args, **kwargs) -> np.ndarray:
        """
        Returns cell masses of the cells as a 1d numpy array.
        """
        blocks = self.cellblocks(*args, inclusive=True, **kwargs)
        vmap = map(lambda b: b.celldata.masses(), blocks)
        return np.concatenate(list(vmap))

    def mass(self, *args, **kwargs) -> float:
        """
        Returns the total mass as a float.
        """
        return np.sum(self.masses(*args, **kwargs))

    def consistent_mass_matrix(self, *args, sparse=True, **kwargs):
        """
        Returns the consistent mass matrix of the mesh with either dense 
        or sparse layout.

        Parameters
        ----------
        sparse : bool, Optional
            Returns the matrix in a sparse format. Default is True.

        Notes
        -----
        If there are nodal masses defined, only sparse output is 
        available at the moment.

        """
        if sparse:
            return self.consistent_mass_matrix_coo(*args, **kwargs)
        else:
            dbkey = self.__class__._point_class_._attr_map_['mass']
            if dbkey in self.root().pointdata.fields:
                return self.consistent_mass_matrix_coo(*args, **kwargs)
        # distributed masses (cells)
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.consistent_mass_matrix()
        return np.vstack(list(map(foo, blocks)))

    def consistent_mass_matrix_coo(self, *args, eliminate_zeros=True,
                                   sum_duplicates=True, distribute=False, **kwargs):
        """
        Returns the mass matrix in coo format. If `distribute` is set 
        to `True`, nodal masses are distributed over neighbouring cells 
        and handled as self-weight is.

        Parameters
        ----------
        eliminate_zeros : bool, Optional
            Eliminates zero entries. Default is True.

        sum_duplicates : bool, Optional
            Sums duplicate entries. Default is True.

        distribute : bool, Optional
            If True, nodal masses are distributed over neighbouring cells 
            and are handled as self-weight is. Default is True.

        Returns
        -------
        scipy.sparse.coo_matrix
            The mass matrix in sparse COO format.

        """
        # distributed masses (cells)
        blocks = list(self.cellblocks(inclusive=True))
        def foo(b): return b.celldata.consistent_mass_matrix_coo()
        M = np.sum(list(map(foo, blocks))).tocoo()
        # nodal masses
        pd = self.root().pointdata
        dbkey = self.__class__._point_class_._attr_map_['mass']
        if dbkey in pd.fields:
            d = pd.mass
            if distribute:
                v = self.volumes()
                edata = list(
                    map(lambda b: b.celldata.pull(data=d, avg=v), blocks))

                def foo(bv): return bv[0].celldata.consistent_mass_matrix_coo(
                    values=bv[1])
                moo = map(lambda i: (blocks[i], edata[i]), range(len(blocks)))
                M += np.sum(list(map(foo, moo))).tocoo()
            else:
                ndof = self.__class__.NDOFN
                M += fem_nodal_mass_matrix_coo(values=d, eliminate_zeros=eliminate_zeros,
                                               sum_duplicates=sum_duplicates, ndof=ndof)
        if eliminate_zeros:
            M.eliminate_zeros()
        if sum_duplicates:
            M.sum_duplicates()
        return M

    def penalty_matrix_coo(self, *args, eliminate_zeros=True,
                           sum_duplicates=True, ensure_comp=False,
                           distribute=False, **kwargs) -> coo_matrix:
        """
        A penalty matrix that enforces Dirichlet boundary conditions. 
        Returns a scipy sparse matrix in coo format.
        """
        fixity = self.root().pointdata.fixity
        K_coo = fem_penalty_matrix_coo(values=fixity, eliminate_zeros=eliminate_zeros,
                                       sum_duplicates=sum_duplicates)
        return K_coo.tocoo()

    def approximation_matrix_coo(self, *args, eliminate_zeros=True,
                                 **kwargs) -> coo_matrix:
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.approximation_matrix_coo()
        res = np.sum(list(map(foo, blocks))).tocoo()
        if eliminate_zeros:
            res.eliminate_zeros()
        return res

    def nodal_approximation_matrix_coo(self, *args, eliminate_zeros=True, 
                                       **kwargs) -> coo_matrix:
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.nodal_approximation_matrix_coo()
        res = np.sum(list(map(foo, blocks))).tocoo()
        if eliminate_zeros:
            res.eliminate_zeros()
        return res

    @squeeze(True)
    def load_vector(self, *args, **kwargs) -> np.ndarray:
        """
        Returns the nodal load vector.

        """
        # concentrated nodal loads
        nodal_data = self.root().pointdata.loads
        nodal_data = atleast3d(nodal_data, back=True)  # (nP, nDOF, nRHS)
        res = fem_load_vector(values=nodal_data, squeeze=False)
        # cells
        blocks = list(self.cellblocks(inclusive=True))
        # body loads
        def foo(b): return b.celldata.body_load_vector(squeeze=False)
        try:
            m = filter(lambda i : i is not None, map(foo, blocks))
            res += np.sum(list(m), axis=0)
        except Exception:
            pass
        # strain loads
        def foo(b): return b.celldata.strain_load_vector(squeeze=False)
        try:
            m = filter(lambda i : i is not None, map(foo, blocks))
            res += np.sum(list(m), axis=0)
        except Exception:
            pass
        return res

    def prostproc_dof_solution(self, *args, **kwargs):
        """
        Calculates approximate solution of the primary variables as a list of arrays
        for each block in the mesh.
        """
        blocks = self.cellblocks(inclusive=True)
        dofsol = self.root().pointdata.dofsol
        def foo(b): return b.celldata.prostproc_dof_solution(dofsol=dofsol)
        list(map(foo, blocks))

    @squeeze(True)
    def nodal_dof_solution(self, *args, flatten=False, **kwargs) -> np.ndarray:
        """
        Returns nodal degree of freedom solution.
        """
        dofsol = self.root().pointdata.dofsol
        if flatten:
            if len(dofsol.shape) == 2:
                return dofsol.flatten()
            else:
                nN, nDOFN, nRHS = dofsol.shape
                return dofsol.reshape((nN * nDOFN, nRHS))
        else:
            return dofsol

    @squeeze(True)
    def cell_dof_solution(self, *args, cells=None, flatten=True,
                          squeeze=True, **kwargs) -> np.ndarray:
        """
        Returns degree of freedom solution for each cell.
        """
        blocks = self.cellblocks(inclusive=True)
        kwargs.update(flatten=flatten, squeeze=False)
        if cells is not None:
            kwargs.update(cells=cells)
            res = {}
            def foo(b): return b.celldata.dof_solution(*args, **kwargs)
            [res.update(d) for d in map(foo, blocks)]
            return res
        else:
            def foo(b): return b.celldata.dof_solution(*args, **kwargs)
            return np.vstack(list(map(foo, blocks)))

    @squeeze(True)
    def strains(self, *args, cells=None, squeeze=True, **kwargs) -> np.ndarray:
        """
        Returns strains for each cell.
        """
        blocks = self.cellblocks(inclusive=True)
        kwargs.update(squeeze=False)
        if cells is not None:
            kwargs.update(cells=cells)
            res = {}
            def foo(b): return b.celldata.strains(*args, **kwargs)
            [res.update(d) for d in map(foo, blocks)]
            return res
        else:
            def foo(b): return b.celldata.strains(*args, **kwargs)
            return np.vstack(list(map(foo, blocks)))

    @squeeze(True)
    def external_forces(self, *args, cells=None, flatten=True, squeeze=True, 
                        **kwargs) -> np.ndarray:
        """
        Returns external forces for each cell.
        """
        blocks = self.cellblocks(inclusive=True)
        kwargs.update(flatten=flatten, squeeze=False)
        if cells is not None:
            kwargs.update(cells=cells)
            res = {}
            def foo(b): return b.celldata.external_forces(*args, **kwargs)
            [res.update(d) for d in map(foo, blocks)]
            return res
        else:
            def foo(b): return b.celldata.external_forces(*args, **kwargs)
            return np.vstack(list(map(foo, blocks)))
    
    @squeeze(True)
    def internal_forces(self, *args, cells=None, flatten=True, squeeze=True, 
                        **kwargs) -> np.ndarray:
        """
        Returns internal forces for each cell.
        """
        blocks = self.cellblocks(inclusive=True)
        kwargs.update(flatten=flatten, squeeze=False)
        if cells is not None:
            kwargs.update(cells=cells)
            res = {}
            def foo(b): return b.celldata.internal_forces(*args, **kwargs)
            [res.update(d) for d in map(foo, blocks)]
            return res
        else:
            def foo(b): return b.celldata.internal_forces(*args, **kwargs)
            return np.vstack(list(map(foo, blocks)))

    def stresses_at_cells_nodes(self, *args, **kwargs) -> np.ndarray:
        """
        Returns stresses at all nodes of all cells.
        """
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.stresses_at_nodes(*args, **kwargs)
        return np.vstack(list(map(foo, blocks)))

    def stresses_at_centers(self, *args, **kwargs) -> np.ndarray:
        """
        Returns stresses at the centers of all cells.
        """
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.stresses_at_centers(*args, **kwargs)
        return np.squeeze(np.vstack(list(map(foo, blocks))))

    @property
    def model(self):
        """
        Returns the attached model object if there is any.
        """
        if self._model is not None:
            return self._model
        else:
            if self.is_root():
                return self._model
            else:
                return self.parent.model

    def material_stiffness_matrix(self):
        """
        Returns the model of the mesh if there is any.
        This is not important if all the celldata in the mesh
        are filled up with appropriate model stiffness matrices.
        Otherwise, the model stiffness matrix is calcualated on the 
        mesh level during the initialization stage and is spread 
        among the subblocks.
        """
        m = self.model
        if isinstance(m, np.ndarray):
            return m
        else:
            try:
                return m.stiffness_matrix()
            except Exception:
                raise RuntimeError("Invalid model type {}".format(type(m)))

    def postprocess(self, *args, **kwargs):
        """
        General postprocessing. This may be reimplemented in subclasses,
        currently nothing happens here.
        """
        pass
    
    
class SolidMesh(FemMesh):
    """
    A data class dedicated to simple 3d solid cells with 3 degrees of freedom per node.    
    
    See also
    --------
    :class:`TET4`
    :class:`TET10`
    :class:`H8`
    :class:`H27`
    """

    dofs = ('UX', 'UY', 'UZ')
    NDOFN = 3
    
    _cell_classes_ = {
        4: TET4,
        10: TET10,
        8: H8,
        27: H27,
    }

    