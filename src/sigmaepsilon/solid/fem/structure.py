# -*- coding: utf-8 -*-
from typing import Union, Tuple, Callable

import numpy as np
from scipy.sparse import coo_matrix

from linkeddeepdict import LinkedDeepDict

from dewloosh.core.wrapping import Wrapper

from neumann import squeeze
from neumann.array import repeat

from .mesh import FemMesh
from .femsolver import FemSolver


__all__ = ['Structure']


def flatten_pd(default=True):
    def decorator(fnc: Callable):
        def inner(*args, **kwargs):
            if kwargs.get('flatten', default):
                x = fnc(*args, **kwargs)
                if len(x.shape) == 2:
                    return x.flatten()
                else:
                    nN, nDOFN, nRHS = x.shape
                    return x.reshape((nN * nDOFN, nRHS))
            else:
                return fnc(*args, **kwargs)
        inner.__doc__ = fnc.__doc__
        return inner
    return decorator


class Structure(Wrapper):
    """
    A higher level class to manage solid structures.

    Parameters
    ----------
    mesh : :class:`FemMesh`
        A finite element mesh.

    """

    def __init__(self, *args, mesh: FemMesh = None, constraints=None, **kwargs):
        if not isinstance(mesh, FemMesh):
            raise NotImplementedError
        super().__init__(wrap=mesh)
        assert mesh is not None, "Some kind of a finite element mesh must be \
            provided with keyword 'mesh'!"
        self.summary = LinkedDeepDict()
        self.solver = 'scipy'
        self.Solver = None
        self._constraints = constraints

    @property
    def mesh(self) -> FemMesh:
        """
        Returns the underlying mesh object.
        """
        return self._wrapped

    @mesh.setter
    def mesh(self, value: FemMesh):
        """
        Sets the underlying mesh object.
        """
        self._wrapped = value

    @property
    def Solver(self) -> FemSolver:
        """
        Returns the solver object, which is initialized in the preprocessing stage.

        See also
        --------
        :func:`preprocess`

        Returns
        -------
        :class:`.femsolver.FemSolver`

        """
        return self._solver

    @Solver.setter
    def Solver(self, value: Solver):
        self._solver = value

    def initialize(self, *args, **kwargs) -> 'Structure':
        """
        Initializes the structure. This includes data initialization 
        for the cells and happens during the preprocessor. Returns the object 
        for continuation.

        See also
        --------
        :func:`preprocess`

        """
        self.summary = LinkedDeepDict()
        blocks = self.mesh.cellblocks(inclusive=True)
        for block in blocks:
            nE = len(block.celldata)

            # populate material stiffness matrices
            if not 'mat' in block.celldata.fields:
                C = block.material_stiffness_matrix()
                if not len(C.shape) == 3:
                    C = repeat(block.material_stiffness_matrix(), nE)
                block.celldata._wrapped['mat'] = C

            # populate frames
            if not 'frames' in block.celldata.fields:
                frames = repeat(block.frame.show(), nE)
                block.celldata._wrapped['frames'] = frames
        return self

    def preprocess(self, *args, **kwargs) -> 'Structure':
        """
        Initializes the database and converts the discription into standard form.
        Returns the object for continuation.

        See also
        --------
        :func:`initialize`
        :func:`to_standard_form`

        """
        self.initialize(*args, **kwargs)
        # self.mesh.nodal_distribution_factors(store=True, key='ndf')  # sets mesh.celldata.ndf
        self.Solver = self.to_standard_form(*args, **kwargs)
        return self

    def process(self, *args, summary=False, **kwargs) -> 'Structure':
        """
        Performs a linear elastostatic solution and returns the structure
        and returns the object for continuation.

        Parameters
        ----------
        summary : bool, Optional
            Appends related data to the summary. Default is False.

        """
        self.Solver.linsolve(*args, summary=True, **kwargs)
        if summary:
            self.summary['linsolve'] = self.Solver.summary[-1]
        return self

    def to_standard_form(self, *args, ensure_comp=False, solver=FemSolver, **kwargs) -> Solver:
        """
        Returns a solver of the problem in standard form. Creation of the solver happens
        during the preprocessing stage.
        """
        mesh = self._wrapped
        f = mesh.load_vector()
        Kp_coo = mesh.penalty_matrix_coo(ensure_comp=ensure_comp, **kwargs)
        K_bulk = mesh.elastic_stiffness_matrix(*args, sparse=False, **kwargs)
        gnum = mesh.element_dof_numbering()
        solvertype = solver if solver is not None else FemSolver
        return solvertype(K_bulk, Kp_coo, f, gnum, regular=False)

    def linsolve(self, *args, summary=False, postproc=True, **kwargs) -> 'Structure':
        """
        Performs a linear elastostatic calculation with pre- and 
        post-processing.

        Parameters
        ----------
        summary : bool, Optional
            Appends to the overall summary if True. This is available
            as `obj.summary`. Default is False.

        postproc : bool, Optional
            To do postprocessing or not. Default is True.

        """
        self.preprocess(*args, summary=summary, **kwargs)
        self.process(*args, summary=summary, **kwargs)
        if postproc:
            return self.postprocess(*args, summary=summary, **kwargs)
        else:
            return self

    def natural_circular_frequencies(self, *args, **kwargs) -> Tuple:
        """
        Returns natural circular frequencies. The call forwards all parameters
        to :func:`sigmaepslion.solid.fem.dyn.natural_circular_frequencies`, 
        see the docs there for the details. 

        Parameters
        ----------
        return_vectors : bool, Optional
            To return eigenvectors or not. Default is False.

        See also
        --------
        :func:`sigmaepslion.solid.fem.dyn.natural_circular_frequencies`

        Returns
        -------
        :class:`numpy.ndarray`
            The natural circular frequencies.

        :class:`numpy.ndarray`
            The eigenvectors, only if 'return_vectors' is True.

        """
        if self.Solver.M is not None:
            self.consistent_mass_matrix()
        return self.Solver.natural_circular_frequencies(*args, **kwargs)

    def elastic_stiffness_matrix(self, *args, **kwargs) -> Union[np.ndarray, coo_matrix]:
        """
        Returns the elastic stiffness matrix of the structure in dense or sparse format.

        Returns
        -------
        :class:`numpy.ndarray` or :class:`scipy.sparse.coo_matrix`

        """
        return self.mesh.elastic_stiffness_matrix(*args, **kwargs)

    def penalty_stiffness_matrix(self, *args, **kwargs) -> coo_matrix:
        """
        Returns the penalty stiffness matrix of the structure.

        Returns
        -------
        :class:`scipy.sparse.coo_matrix`

        """
        return self.mesh.penalty_matrix_coo(*args, **kwargs)

    def consistent_mass_matrix(self, *args, **kwargs) -> Union[np.ndarray, coo_matrix]:
        """
        Returns the consistent mass matrix of the structure.

        Notes
        -----
        If there are nodal masses defined, only sparse output is 
        available at the moment.

        Returns
        -------
        :class:`numpy.ndarray` or :class:`scipy.sparse.coo_matrix`

        """
        M = self.mesh.consistent_mass_matrix(*args, **kwargs)
        if self.Solver is not None:
            if self.Solver.M is None:
                self.Solver.M = M
        return M

    @flatten_pd(False)
    @squeeze(True)
    def nodal_dof_solution(self, *args, flatten: bool = False, squeeze: bool = True,
                           store: str = None, **kwargs) -> np.ndarray:
        """
        Returns the vector of nodal displacements and optionally stores the result
        with a specified key.

        Parameters
        ----------
        flatten : bool, Optional
            If True, the result is a 1d array, otherwise 2d. Default is False.

        squeeze : bool, Optional
            Calls :func:`numpy.squeeze` on the result. This might be relevant for 
            single element operations, or if there is only one load case, etc. Then,
            it depends on the enviroment if a standard shape is desirable to maintain or not.
            Default is True.

        store : str, Optional
            If a string is provided, the resulting data can be later accessed as an attribute
            of the pointdata of the mesh object. Default is None.

        """
        mesh = self._wrapped
        nDOFN = mesh.NDOFN
        nN = mesh.number_of_points()

        # store dof solution
        u = self.Solver.u
        nRHS = 1 if len(u.shape) == 1 else u.shape[-1]
        if nRHS == 1:
            res = np.reshape(u, (nN, nDOFN))
        else:
            res = np.reshape(u, (nN, nDOFN, nRHS))

        if isinstance(store, str):
            mesh.pointdata[store] = res

        return res

    @flatten_pd(False)
    @squeeze(True)
    def reaction_forces(self, *args, flatten: bool = False, squeeze: bool = True,
                        store: str = None, **kwargs) -> np.ndarray:
        """
        Returns the vector of reaction forces.

        Parameters
        ----------
        flatten : bool, Optional
            If True, the result is a 1d array, otherwise 2d. Default is False.

        squeeze : bool, Optional
            Calls :func:`numpy.squeeze` on the result. This might be relevant for 
            single element operations, or if there is only one load case, etc. Then,
            it depends on the enviroment if a standard shape is desirable to maintain or not.
            Default is True.

        store : str, Optional
            If a string is provided, the resulting data can be later accessed as an attribute
            of the pointdata of the mesh object. Default is None.

        """
        mesh = self._wrapped
        nDOFN = mesh.NDOFN
        nN = mesh.number_of_points()

        # store dof solution
        r = self.Solver.r
        nRHS = 1 if len(r.shape) == 1 else r.shape[-1]
        if nRHS == 1:
            res = np.reshape(r, (nN, nDOFN))
        else:
            res = np.reshape(r, (nN, nDOFN, nRHS))

        if isinstance(store, str):
            mesh.pointdata[store] = res

        return res

    @flatten_pd(False)
    @squeeze(True)
    def nodal_forces(self, *args, flatten: bool = False, squeeze: bool = True,
                     store: str = None, **kwargs) -> np.ndarray:
        """
        Returns the vector of nodal forces.

        Parameters
        ----------
        flatten : bool, Optional
            If True, the result is a 1d array, otherwise 2d. Default is False.

        squeeze : bool, Optional
            Calls :func:`numpy.squeeze` on the result. This might be relevant for 
            single element operations, or if there is only one load case, etc. Then,
            it depends on the enviroment if a standard shape is desirable to maintain or not.
            Default is True.

        store : str, Optional
            If a string is provided, the resulting data can be later accessed as an attribute
            of the pointdata of the mesh object. Default is None.

        """

        mesh = self._wrapped
        nDOFN = mesh.NDOFN
        nN = mesh.number_of_points()

        # store nodal loads
        f = self.Solver.f
        nRHS = 1 if len(f.shape) == 1 else f.shape[-1]
        if nRHS == 1:
            res = np.reshape(f, (nN, nDOFN))
        else:
            res = np.reshape(f, (nN, nDOFN, nRHS))

        if isinstance(store, str):
            mesh.pointdata[store] = res

        return res

    @squeeze(True)
    def internal_forces(self, *args, flatten=False, squeeze=True, **kwargs) -> np.ndarray:
        """
        Returns the internal forces for one or more elements.
        """
        return self.mesh.internal_forces(*args, flatten=flatten, squeeze=False, **kwargs)

    def postprocess(self, *args, cleanup=False, **kwargs) -> 'Structure':
        """
        Runs general postprocessing of the solution data. Returns the object for continuation.

        The calculations produce the following point-related data, stored in the pointdata
        object of the root object of the finite element mesh:

            - nodal dof solution, available with key 'dofsol'

            - vector of nodal forces, available with key 'forces'

            - vector of reaction forces, available with key 'reactions'

        For information on the generated cell-related data see :func:`sigmaepsilon.solid.fem.mesh.FemMesh.postprocess`.

        Parameters
        ----------
        *args : tuple, Optional
            Optional positional arguments to the postprocessing engine of the mesh.
            Only if postproc is True.

        **kwargs : dict, Optional
            Optional keyword arguments to the postprocessing engine of the mesh.
            Only if postproc is True.

        cleanup : bool, Optional
            If True, data related to the calculation is freed up and the model
            needs to be reinitialized to run calculations again. Default is False.

        See also
        --------
        :class:`sigmaepsilon.solid.fem.mesh.FemMesh`

        Note
        ----
        This can be triggered automatically during linear solution when calling :func:`linsolve`
        with `postproc=True`.

        Returns
        -------
        :class:`Structure`

        """
        mesh = self._wrapped
        mesh.postprocess(*args, **kwargs)

        # clean up
        _ = self.cleanup() if cleanup else None

        return self

    def cleanup(self) -> 'Structure':
        """
        Destroys the solver and returns the object for continuation.

        Returns
        -------
        :class:`Structure`

        """
        self.Solver = None
        return self
