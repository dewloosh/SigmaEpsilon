# -*- coding: utf-8 -*-
from typing import Union, List, Iterable, Collection
import logging

from scipy.sparse import coo_matrix
import numpy as np
from numpy import ndarray

from neumann import squeeze
from neumann.array import atleast3d

from polymesh import PolyData

from .pointdata import PointData
from .preproc import (nodal_load_vector, essential_penalty_matrix,
                      nodal_mass_matrix)
from .cells import TET4, TET10, H8, H27, FiniteElement
from .metamesh import ABC_FemMesh
from .constants import DEFAULT_DIRICHLET_PENALTY


class FemMesh(PolyData, ABC_FemMesh):
    """
    A descendant of :class:`polymesh.PolyData` to handle polygonal meshes for 
    the **Finite Element Method**.

    Parameters
    ----------
    pd : polymesh.PolyData or polymesh.CellData, Optional
        A PolyData or a CellData instance. Dafault is None.
    cd : polymesh.CellData, Optional
        A CellData instance, if the first argument is provided. Dafault is None.

    Note
    ----
    The suggested way of usage is to create the pointdata and celldata
    objects in advance, and provide them as the firts argument when building 
    the model.

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

    def __init__(self, *args, model=None, cell_fields: dict = None,
                 point_fields: dict = None, **kwargs):
        point_fields = {} if point_fields is None else point_fields
        cell_fields = {} if cell_fields is None else cell_fields
        super().__init__(*args, point_fields=point_fields,
                         cell_fields=cell_fields, **kwargs)
        self._model = model

    @property
    def pd(self) -> PointData:
        """
        Returns the attached pointdata.
        
        See Also
        --------
        class:`sigmaepsilon.solid.fem.pointdata.PointData`
        
        """
        return self.pointdata

    @property
    def cd(self) -> FiniteElement:
        """
        Returns the attached celldata.
        
        See Also
        --------
        class:`sigmaepsilon.solid.fem.cells.elem.FiniteElement`
        class:`sigmaepsilon.solid.fem.cells.celldata.CellData`
        
        """
        return self.celldata
    
    def cellblocks(self, *args, **kwargs) -> Collection['FemMesh']:
        """
        Returns an iterable over blocks with cell data.
        """
        return filter(lambda i: i.celldata is not None, 
                      self.blocks(*args, **kwargs))
                    
    def cells_coords(self, *args, points: Iterable = None, 
                     cells: Iterable = None,
                     **kwargs) -> Union[ndarray, List[ndarray]]:
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
                def foo(b:FemMesh): 
                    return b.cd.coords(*args, **kwargs)
                [res.update(d) for d in map(foo, blocks)]
                return res
            else:
                def foo(b:FemMesh): 
                    return b.cd.coords(*args, **kwargs)
                return np.vstack(list(map(foo, blocks)))

    def element_dof_numbering(self) -> ndarray:
        """
        Returns global ids of the local degrees of freedoms for each cell.
        """
        blocks = self.cellblocks(inclusive=True)
        def foo(b:FemMesh): 
            return b.cd.global_dof_numbering()
        return np.vstack(list(map(foo, blocks)))
    
    def element_fixity(self) -> ndarray:
        """
        Returns element fixity data.
        """
        fixity = []
        for b in self.cellblocks(inclusive=True):
            if b.has_fixity:
                fixity.append(b.cd.fixity.astype(float))
            else:
                nE, nNE, nDOF = len(b.cd), b.cd.NNODE, self.NDOFN
                fixity.append(np.ones(nE, nNE, nDOF))
        return np.vstack(fixity)
    
    def direction_cosine_matrix(self, target:str='global'):
        blocks = self.cellblocks(inclusive=True)
        def foo(b:FemMesh): 
            return b.cd.direction_cosine_matrix(target=target)
        return np.vstack(list(map(foo, blocks)))

    def elastic_stiffness_matrix(self, *, 
                                 eliminate_zeros: bool = True,
                                 sum_duplicates: bool = True, 
                                 sparse: bool = False,
                                 transform:bool = True,
                                 **kwargs) -> Union[ndarray, coo_matrix]:
        """
        Returns the elastic stiffness matrix in dense or sparse format.

        Parameters
        ----------
        sparse : bool, Optional
            If True, the result is a sparse matrix. Default is False.
        eliminate_zeros : bool, Optional
            Eliminates zero entries. Only if 'sparse' is True. Default is True.
        sum_duplicates : bool, Optional
            Sums duplicate entries. Only if 'sparse' is True. Default is True.
        transform : bool, Optional
            If True, local matrices are transformed to the global frame.
            Default is True.
            
        Returns
        -------
        numpy.ndarray or scipy.sparse.coo_matrix

        """
        blocks = self.cellblocks(inclusive=True)
        if sparse:
            assert transform, "Must transform for sparse layout."
            def foo(b:FemMesh): 
                return b.cd.elastic_stiffness_matrix(sparse=True, transform=True)
            K = np.sum(list(map(foo, blocks))).tocoo()
            if eliminate_zeros:
                K.eliminate_zeros()
            if sum_duplicates:
                K.sum_duplicates()
            return K
        else:
            def foo(b:FemMesh): 
                return b.cd.elastic_stiffness_matrix(transform=transform)
            return np.vstack(list(map(foo, blocks)))

    def masses(self, *args, **kwargs) -> ndarray:
        """
        Returns masses of the cells as a 1d numpy array.
        """
        blocks = self.cellblocks(*args, inclusive=True, **kwargs)
        def foo(b:FemMesh): 
            return b.cd.masses()
        vmap = map(foo, blocks)
        return np.concatenate(list(vmap))

    def mass(self, *args, **kwargs) -> float:
        """
        Returns the total mass of the structure.
        """
        return np.sum(self.masses(*args, **kwargs))

    def consistent_mass_matrix(self, *, eliminate_zeros: bool = True,
                               sum_duplicates: bool = True, sparse: bool = False,
                               transform:bool = True,**kwargs) \
                                   -> Union[ndarray, coo_matrix]:
        """
        Returns the stiffness-consistent mass matrix as a dense or a sparse 
        matrix.

        Parameters
        ----------
        sparse : bool, Optional
            If True, the result is a sparse matrix. Default is False.
        eliminate_zeros : bool, Optional
            Eliminates zero entries. Only if 'sparse' is True. Default is True.
        sum_duplicates : bool, Optional
            Sums duplicate entries. Only if 'sparse' is True. Default is True.
        transform : bool, Optional
            If True, local matrices are transformed to the global frame.
            Default is True.
            
        Returns
        -------
        numpy.ndarray or scipy.sparse.coo_matrix
            The mass matrix as a 3d dense or a 2d sparse array.

        """
        # distributed masses (cells)
        blocks = self.cellblocks(inclusive=True)
        if sparse:
            def foo(b:FemMesh): 
                return b.cd.consistent_mass_matrix(sparse=True, transform=True)
            M = np.sum(list(map(foo, blocks))).tocoo()
            if eliminate_zeros:
                M.eliminate_zeros()
            if sum_duplicates:
                M.sum_duplicates()
            return M
        else:
            def foo(b:FemMesh): 
                return b.cd.consistent_mass_matrix(transform=transform)
            return np.vstack(list(map(foo, blocks)))

    def mass_matrix(self, *, eliminate_zeros: bool = True,
                    sum_duplicates: bool = True, **kwargs) -> coo_matrix:
        """
        Returns the mass matrix in coo format.The resulting matrix is the 
        sum of the consistent mass matrix and the nodal mass matrix.

        Parameters
        ----------
        eliminate_zeros : bool, Optional
            Eliminates zero entries. Default is True.
        sum_duplicates : bool, Optional
            Sums duplicate entries. Default is True.

        Returns
        -------
        scipy.sparse.coo_matrix
            The mass matrix in sparse COO format.

        """
        # distributed masses (cells)
        M = self.consistent_mass_matrix(sparse=True,
                                        transform=True,
                                        eliminate_zeros=False,
                                        sum_duplicates=False)
        # nodal masses
        M += self.nodal_mass_matrix(eliminate_zeros=False,
                                    sum_duplicates=False)
        if eliminate_zeros:
            M.eliminate_zeros()
        if sum_duplicates:
            M.sum_duplicates()
        return M.tocoo()

    def nodal_mass_matrix(self, *, eliminate_zeros: bool = False,
                          sum_duplicates: bool = False, **kw):
        """
        Returns the mass matrix from nodal masses only, in coo format.

        Parameters
        ----------
        eliminate_zeros : bool, Optional
            Eliminates zero entries. Default is True.
        sum_duplicates : bool, Optional
            Sums duplicate entries. Default is True.

        Returns
        -------
        scipy.sparse.coo_matrix
            The nodal mass matrix in sparse COO format.

        """
        pd = self.root().pointdata
        nN = len(pd)
        nDOF = self.__class__.NDOFN
        try:
            m = pd.mass
            M = nodal_mass_matrix(values=m, eliminate_zeros=eliminate_zeros,
                                  sum_duplicates=sum_duplicates, ndof=nDOF)
            return M
        except Exception as e:
            nX = nN * nDOF
            return coo_matrix(shape=(nX, nX))

    def essential_penalty_matrix(self, fixity: ndarray = None, *,
                                 eliminate_zeros: bool = True,
                                 sum_duplicates: bool = True,
                                 penalty: float = DEFAULT_DIRICHLET_PENALTY,
                                 **kwargs) -> coo_matrix:
        """
        A penalty matrix that enforces Dirichlet boundary conditions. 
        Returns a scipy sparse matrix in coo format.

        Parameters
        ----------
        fixity : numpy.ndarray, Optional
            The fixity values as a 2d array. The length of the first axis must 
            equal the number of nodes, the second the number of DOFs of the model.
            If not specified, the data is assumed to be stored in the attached 
            pointdata with the same key. Default is None.
        penalty : float, Optional
            The Courant-type penalty value.
            Default is `sigmaepsilon.solid.fem.constants.DEFAULT_DIRICHLET_PENALTY`.
        eliminate_zeros : bool, Optional
            Eliminates zeros from the matrix. Default is True.
        eliminate_zeros : bool, Optional
            Summs duplicate entries in the matrix. Default is True.

        Returns
        -------
        scipy.sparse.coo_matrix

        """
        root = self.root()
        pd = root.pointdata
        N = len(pd) * root.NDOFN
        K_coo = coo_matrix(([0.], ([0], [0])), shape=(N, N))
        try:
            fixity = pd.fixity
            K_coo += essential_penalty_matrix(fixity=fixity, pfix=penalty,
                                              eliminate_zeros=eliminate_zeros,
                                              sum_duplicates=sum_duplicates)
        except AttributeError:
            logging.info('No FIXITY information in pointdata.')
        except Exception as e:
            logging.error(f'Exception {e} occured', exc_info=True)
        return K_coo.tocoo()

    @squeeze(True)
    def nodal_load_vector(self, *args, **kwargs) -> ndarray:
        """
        Returns the nodal load vector.

        Returns
        -------
        numpy.ndarray

        """
        # concentrated nodal loads
        nodal_data = self.root().pointdata.loads
        nodal_data = atleast3d(nodal_data, back=True)
        # (nP, nDOF, nRHS)
        res = nodal_load_vector(nodal_data, squeeze=False)
        return res
    
    @squeeze(True)
    def cell_load_vector(self, *, transform: bool = True, 
                         assemble: bool = False, **kwargs) -> ndarray:
        """
        Returns the nodal load vector from body and strain loads.
        
        Parameters
        ----------
        assemble : bool, Optional
            If True, the values are returned with a matching shape to 
            the total system. Default is False.
        transform : bool, Optional
            If True, local matrices are transformed to the global frame.
            Default is False.       

        Returns
        -------
        numpy.ndarray
            The shape depends on the arguments.
            
        """
        if assemble:
            assert transform, "Must transform before assembly."
        blocks = list(self.cellblocks(inclusive=True))
        params = dict(
            squeeze=False,
            transform=transform,
            assemble=assemble
        )
        params.update(**kwargs)
        def foo(b:FemMesh): 
            return b.cd.load_vector(**params)
        return np.vstack(list(map(foo, blocks)))

    def prostproc_dof_solution(self, *args, **kwargs):
        """
        Calculates approximate solution of the primary variables as a list 
        of arrays for each block in the mesh.
        """
        blocks = self.cellblocks(inclusive=True)
        dofsol = self.root().pointdata.dofsol
        def foo(b:FemMesh): 
            return b.cd.prostproc_dof_solution(dofsol=dofsol)
        list(map(foo, blocks))
        
    def condensate_cell_fixity(self) -> 'FemMesh':
        """
        Performs static condensation of the system equations to account for
        cell fixity. Returns the mesh object for continuation.
        """
        [b.cd.condensate() for b in self.cellblocks(inclusive=True)]
        return self

    @squeeze(True)
    def nodal_dof_solution(self, *, flatten: bool = False, **kw) -> ndarray:
        """
        Returns nodal degree of freedom solution.
        
        Parameters
        ----------
        flatten: bool, Optional
            If True, nodal results are flattened. Default is False.

        Returns
        -------
        numpy.ndarray

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
    def cell_dof_solution(self, *args, cells:Iterable=None, 
                          flatten: bool = True, **kwargs) -> ndarray:
        """
        Returns degree of freedom solution for each cell or just some.

        Parameters
        ----------
        cells : Iterable, Optional
            Indices of cells for which data is requested.
            If specified, the result is a dictionary with the indices
            being the keys. Default is None.
        flatten: bool, Optional
            If True, nodal results are flattened. Default is True.
        squeeze : bool, Optional
            If True, dummy axes are removed. This may be relevant if you
            only have one load case or element, but still want to have
            results with predictable shape. Default is True.

        Returns
        -------
        numpy.ndarray or dict
            A dictionary if cell indices are specified, or a NumPy array.

        """
        kwargs.update(flatten=flatten, squeeze=False)
        if cells is not None:
            def f(b:FemMesh, cid:int): 
                return b.cd.dof_solution(*args, cells=[cid], **kwargs)[0]
            boc = self.blocks_of_cells(i=cells)
            return {cid : f(boc[cid], cid) for cid in cells}
        else:
            blocks = self.cellblocks(inclusive=True)
            def foo(b:FemMesh): 
                return b.cd.dof_solution(*args, **kwargs)
            return np.vstack(list(map(foo, blocks)))

    @squeeze(True)
    def strains(self, *args, cells:Iterable=None, **kwargs) -> ndarray:
        """
        Returns strains for each cell or just some.

        Parameters
        ----------
        cells : Iterable, Optional
            Indices of cells for which data is requested.
            If specified, the result is a dictionary with the indices
            being the keys. Default is None.
        flatten: bool, Optional
            If True, nodal results are flattened. Default is True.
        squeeze : bool, Optional
            If True, dummy axes are removed. This may be relevant if you
            only have one load case or element, but still want to have
            results with predictable shape. Default is True.

        Returns
        -------
        numpy.ndarray or dict
            A dictionary if cell indices are specified, or a NumPy array.

        """
        kwargs.update(squeeze=False)
        if cells is not None:
            def f(b:FemMesh, cid:int): 
                return b.cd.strains(*args, cells=[cid], **kwargs)[0]
            boc = self.blocks_of_cells(i=cells)
            return {cid : f(boc[cid], cid) for cid in cells}
        else:
            blocks = self.cellblocks(inclusive=True)
            def foo(b:FemMesh): 
                return b.cd.strains(*args, **kwargs)
            return np.vstack(list(map(foo, blocks)))

    @squeeze(True)
    def external_forces(self, *args, cells:Iterable=None, 
                        flatten: bool = True, **kwargs) -> ndarray:
        """
        Returns external forces for each cell or just some.

        Parameters
        ----------
        cells : Iterable, Optional
            Indices of cells for which external forces are requested.
            If specified, the result is a dictionary with the indices
            being the keys. Default is None.
        flatten: bool, Optional
            If True, nodal results are flattened. Default is True.
        squeeze : bool, Optional
            If True, dummy axes are removed. This may be relevant if you
            only have one load case or element, but still want to have
            results with predictable shape. Default is True.

        Returns
        -------
        numpy.ndarray or dict
            A dictionary if cell indices are specified, or a NumPy array.

        """
        kwargs.update(flatten=flatten, squeeze=False)
        if cells is not None:
            def f(b:FemMesh, cid:int): 
                return b.cd.external_forces(*args, cells=[cid], **kwargs)[0]
            boc = self.blocks_of_cells(i=cells)
            return {cid : f(boc[cid], cid) for cid in cells}
        else:
            blocks = self.cellblocks(inclusive=True)
            def foo(b:FemMesh): 
                return b.cd.external_forces(*args, **kwargs)
            return np.vstack(list(map(foo, blocks)))

    @squeeze(True)
    def internal_forces(self, *args, cells:Iterable=None, 
                        flatten: bool = True, **kwargs) -> ndarray:
        """
        Returns internal forces for each cell or just some.
        
        Parameters
        ----------
        cells : Iterable, Optional
            Indices of cells for which internal forces are requested.
            If specified, the result is a dictionary with the indices
            being the keys. Default is None.
        flatten: bool, Optional
            If True, nodal results are flattened. Default is True.
        squeeze : bool, Optional
            If True, dummy axes are removed. This may be relevant if you
            only have one load case or element, but still want to have
            results with predictable shape. Default is True.

        Returns
        -------
        numpy.ndarray or dict
            A dictionary if cell indices are specified, or a NumPy array.

        """
        kwargs.update(flatten=flatten, squeeze=False)
        if cells is not None:
            def f(b:FemMesh, cid:int): 
                return b.cd.internal_forces(*args, cells=[cid], **kwargs)[0]
            boc = self.blocks_of_cells(i=cells)
            return {cid : f(boc[cid], cid) for cid in cells}
        else:
            blocks = self.cellblocks(inclusive=True)
            def foo(b:FemMesh): 
                return b.cd.internal_forces(*args, **kwargs)
            return np.vstack(list(map(foo, blocks)))

    def stresses_at_cells_nodes(self, *args, **kwargs) -> ndarray:
        """
        Returns stresses at all nodes of all cells.
        """
        blocks = self.cellblocks(inclusive=True)
        def foo(b:FemMesh): 
            return b.cd.stresses_at_nodes(*args, **kwargs)
        return np.vstack(list(map(foo, blocks)))

    def stresses_at_centers(self, *args, **kwargs) -> ndarray:
        """
        Returns stresses at the centers of all cells.

        Returns
        -------
        numpy.ndarray

        """
        blocks = self.cellblocks(inclusive=True)
        def foo(b:FemMesh): 
            return b.cd.stresses_at_centers(*args, **kwargs)
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

    def material_stiffness_matrix(self) -> ndarray:
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
    A data class dedicated to simple 3d solid cells with 3 degrees of 
    freedom per node.    

    See Also
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
