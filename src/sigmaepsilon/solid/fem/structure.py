# -*- coding: utf-8 -*-
import numpy as np
from typing import Tuple

from linkeddeepdict import LinkedDeepDict
from dewloosh.core.wrapping import Wrapper
from neumann import squeeze
from neumann.array import repeat

from .mesh import FemMesh

from .femsolver import Newton, Solver


__all__ = ['Structure']


class Structure(Wrapper):

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
    def mesh(self):
        """
        Returns the underlying mesh.
        """
        return self._wrapped

    @mesh.setter
    def mesh(self, value: FemMesh):
        self._wrapped = value
    
    @property
    def Solver(self) -> Newton:
        return self._solver
    
    @Solver.setter
    def Solver(self, value : Solver):
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
        #self.mesh.nodal_distribution_factors(store=True, key='ndf')  # sets mesh.celldata.ndf
        self.Solver = self.to_standard_form(*args, **kwargs)
        return self

    def to_standard_form(self, *args, ensure_comp=False, solver=Newton, **kwargs) -> Solver:
        """
        Returns a solver of the problem in standard form. Creation of the solver happens
        during the preprocessing stage.
        """
        mesh = self._wrapped
        f = mesh.load_vector()
        Kp_coo = mesh.penalty_matrix_coo(ensure_comp=ensure_comp, **kwargs)
        K_bulk = mesh.stiffness_matrix(*args, sparse=False, **kwargs)
        gnum = mesh.element_dof_numbering()
        solvertype = solver if solver is not None else Newton
        return solvertype(K_bulk, Kp_coo, f, gnum, regular=False)

    def linsolve(self, *args, summary=False, **kwargs) -> 'Structure':
        """
        Performs a linear elastostatic calculation with pre- and 
        post-processing
        
        Parameters
        ----------
        summary : bool, Optional
            Controls basic logging.
            
        Returns
        -------
        Structure
        
        """
        self.preprocess(*args, summary=summary, **kwargs)
        self.process(*args, summary=summary, **kwargs)
        return self.postprocess(*args, summary=summary, **kwargs)

    def modes_of_vibration(self, *args, normalize=True, around=None, 
                           distribute_nodal_masses=False, as_dense=False, 
                           **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """        
        Calculates natural modes of vibration and the corresponding eigenvalue
        for each mode.
        
        Parameters
        ----------
        normalize : bool, Optional
            If True, the returned eigenvectors are normlized to the mass matrix.
            Default is True.
            
        as_dense : bool, Optional
            Controls whether the mass matrix is handles as a dense or a sparse matrix.
            Default is False. 
        
        distribute_nodal_masses : bool, Optional
            If True, nodal masses are distributed over the neighbouring elements
            and handled similary to self-weight. Default is False. 
            Only when `mode` is 'M'. Default is False.
            
        mode : str, Optional
            Mode of normalization, if 'normalize' is `True`. Default is 'M', which
            means normalization to the mass matrix.
        
        around : float, Optional
            A target (possibly an approximation) value around which eigenvalues
            are searched. Default is None. 
        
        Returns
        -------
        np.ndarray
            Eigenvalues (natural circular frequencies).
        
        np.ndarray
            Eigenvectors (natural modes of vibrations).
            
        """
        M = self.mesh.mass_matrix(distribute=distribute_nodal_masses)
        self.Solver.M = M
        if around is not None:
            assert not as_dense
            sigma = (np.pi * 2 * around)**2
            kwargs['sigma'] = sigma
        return self.Solver.modes_of_vibration(*args, normalize=normalize, 
                                              as_dense=as_dense, **kwargs)
        
    def effective_modal_masses(self, *args, action=None, **kwargs) -> np.ndarray:
        """
        Returns effective modal masses of several modes of vibration.
        
        Parameters
        ----------
        action : Iterable
            1d iterable, with a length matching the dof layout of the structure.
            
        Notes
        -----
        1) The sum of all effective masses equals the total mass of the structure.
        2) This requires natural modes of vibration to be calculated.
             
        Returns
        -------
        numpy array
            An array of effective mass values.
            
        See also
        --------
        :func:`modes_of_vibration`
        
        """
        return self.Solver.effective_modal_masses(*args, action=action, **kwargs)
    
    def modal_participation_factors(self, *args, action=None, **kwargs) -> np.ndarray:
        """
        Returns modal participation factors for several actions.
        
        Parameters
        ----------
        action : Iterable
            1d iterable, with a length matching the dof layout of the structure.
            
        Notes
        -----
        1) This requires natural modes of vibration to be calculated.
             
        Returns
        -------
        numpy array
            An array of effective mass values.
            
        See also
        --------
        :func:`modes_of_vibration`
        
        """
        return self.Solver.modal_participation_factors(*args, action=action, **kwargs)
    
    def process(self, *args, summary=False, **kwargs):
        """
        Performs a linear elastostatic solution and returns the structure.
        
        Parameters
        ----------
        summary : bool, Optional
            Appends related data to the summary. Default is False.
            
        """
        self.Solver.linsolve(*args, summary=True, **kwargs)
        if summary:
            self.summary['linsolve'] = self.Solver.summary[-1]
        return self
        
    def stiffness_matrix(self, *args, **kwargs) -> np.ndarray:
        """
        Returns the stiffness matrix of the structure.
        """
        return self.mesh.stiffness_matrix(*args, **kwargs)
    
    def penalty_stiffness_matrix(self, *args, **kwargs):
        """
        Returns the penalty stiffness matrix of the structure.
        """
        return self.mesh.penalty_matrix_coo(*args, **kwargs)
    
    def mass_matrix(self, *args, **kwargs):
        """
        Returns the mass matrix of the structure.
        """
        return self.mesh.mass_matrix(*args, **kwargs)

    @squeeze(True)
    def nodal_dof_solution(self, *args, flatten=False, squeeze=True, **kwargs):
        """
        Returns the vector of nodal displacements.
        """
        return self.mesh.nodal_dof_solution(*args, flatten=flatten, squeeze=False, **kwargs)

    @squeeze(True)
    def reaction_forces(self, *args, flatten=False, squeeze=True, **kwargs):
        """
        Returns the vector of reaction forces.
        """
        return self.mesh.reaction_forces(*args, flatten=flatten, squeeze=False, **kwargs)

    @squeeze(True)
    def nodal_forces(self, *args, flatten=False, squeeze=True, **kwargs):
        """
        Returns the vector of nodal forces.
        """
        return self.mesh.nodal_forces(*args, flatten=flatten, squeeze=False, **kwargs)

    @squeeze(True)
    def internal_forces(self, *args, flatten=False, squeeze=True, **kwargs):
        """
        Returns the internal forces for one or more elements.
        """
        return self.mesh.internal_forces(*args, flatten=flatten, squeeze=False, **kwargs)

    def postprocess(self, *args, summary=True, cleanup=False, **kwargs) -> 'Structure':
        """
        Does basic postprocessing. This is triggered automatically during 
        solution. Returns the object for continuation.
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

    def cleanup(self):
        """
        Destroys the solver.
        """
        self.Solver = None