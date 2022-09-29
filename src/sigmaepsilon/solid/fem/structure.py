# -*- coding: utf-8 -*-
from typing import Union, Tuple

import numpy as np
from scipy.sparse import coo_matrix

from linkeddeepdict import LinkedDeepDict

from dewloosh.core.wrapping import Wrapper

from neumann import squeeze
from neumann.array import repeat

from .mesh import FemMesh
from .femsolver import FemSolver


__all__ = ['Structure']


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

    def linsolve(self, *args, summary=False, **kwargs) -> 'Structure':
        """
        Performs a linear elastostatic calculation with pre- and 
        post-processing.

        Parameters
        ----------
        summary : bool, Optional
            Appends to the overall summary if True. This is available
            as `obj.summary`. Default is False.

        """
        self.preprocess(*args, summary=summary, **kwargs)
        self.process(*args, summary=summary, **kwargs)
        return self.postprocess(*args, summary=summary, **kwargs)

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

    @squeeze(True)
    def nodal_dof_solution(self, *args, flatten=False, squeeze=True, **kwargs) -> np.ndarray:
        """
        Returns the vector of nodal displacements.
        """
        return self.mesh.nodal_dof_solution(*args, flatten=flatten, squeeze=False, **kwargs)

    @squeeze(True)
    def reaction_forces(self, *args, flatten=False, squeeze=True, **kwargs) -> np.ndarray:
        """
        Returns the vector of reaction forces.
        """
        return self.mesh.reaction_forces(*args, flatten=flatten, squeeze=False, **kwargs)

    @squeeze(True)
    def nodal_forces(self, *args, flatten=False, squeeze=True, **kwargs) -> np.ndarray:
        """
        Returns the vector of nodal forces.
        """
        return self.mesh.nodal_forces(*args, flatten=flatten, squeeze=False, **kwargs)

    @squeeze(True)
    def internal_forces(self, *args, flatten=False, squeeze=True, **kwargs) -> np.ndarray:
        """
        Returns the internal forces for one or more elements.
        """
        return self.mesh.internal_forces(*args, flatten=flatten, squeeze=False, **kwargs)

    def postprocess(self, *args, summary=True, cleanup=False, **kwargs) -> 'Structure':
        """
        Does basic postprocessing. This is triggered automatically during 
        linear solution. Returns the object for continuation.
        """
        mesh = self._wrapped
        nDOFN = mesh.NDOFN
        nN = mesh.number_of_points()

        # store dof solution
        u = self.Solver.u
        nRHS = 1 if len(u.shape) == 1 else u.shape[-1]
        if nRHS == 1:
            mesh.pointdata['dofsol'] = np.reshape(u, (nN, nDOFN))
        else:
            mesh.pointdata['dofsol'] = np.reshape(u, (nN, nDOFN, nRHS))

        # store nodal loads
        f = self.Solver.f
        nRHS = 1 if len(f.shape) == 1 else f.shape[-1]
        if nRHS == 1:
            mesh.pointdata['forces'] = np.reshape(f, (nN, nDOFN))
        else:
            mesh.pointdata['forces'] = np.reshape(f, (nN, nDOFN, nRHS))

        # store dof solution
        r = self.Solver.r
        nRHS = 1 if len(r.shape) == 1 else r.shape[-1]
        if nRHS == 1:
            mesh.pointdata['reactions'] = np.reshape(r, (nN, nDOFN))
        else:
            mesh.pointdata['reactions'] = np.reshape(r, (nN, nDOFN, nRHS))

        mesh.postprocess(*args, **kwargs)

        # clean up
        _ = self.cleanup() if cleanup else None

        return self

    def cleanup(self) -> 'Structure':
        """
        Destroys the solver and returns the object for continuation.
        """
        self.Solver = None
        return self
