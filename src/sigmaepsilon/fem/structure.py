from typing import Tuple, Union, List

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix

from neumann import repeat

from .mesh import FemMesh
from .ebc import EssentialBoundaryCondition as EBC
from .femsolver import StaticSolver, DynamicSolver
from ..utils.fem.preproc import assemble_load_vector
from ..utils.fem.tr import tr_element_matrices_bulk, tr_nodal_loads_bulk
from ..utils.fem.dyn import effective_modal_masses, Rayleigh_quotient
from .pointdata import flatten_pd
from .constants import DEFAULT_DIRICHLET_PENALTY, DEFAULT_MASS_PENALTY_RATIO


__all__ = ["Structure"]


class Structure:
    """
    A higher level class to manage solid structures.

    Parameters
    ----------
    mesh : FemMesh, Optional
        A finite element mesh.
    essential_penalty : float, Optional
        Penalty parameter for the Courant penalty function used to enforce
        Dirichlet (essential) boundary conditions. If not provided, a default
        value of `~sigmaepsilon.fem.constants.DEFAULT_DIRICHLET_PENALTY`
        is used.
    mass_penalty_ratio : float, Optional
        Ratio of the penalty factors applied to the stiffness matrix (pK) and
        the mass matrix (pM) as pM/pK. If not provided, a default
        value of `~sigmaepsilon.fem.constants.DEFAULT_MASS_PENALTY_RATIO`
        is used.
    """

    def __init__(
        self,
        mesh: FemMesh,
        essential_penalty: float = DEFAULT_DIRICHLET_PENALTY,
        mass_penalty_ratio: float = DEFAULT_MASS_PENALTY_RATIO,
    ):
        if not isinstance(mesh, FemMesh):
            raise TypeError(
                "The type of 'mesh' is {},"
                " which is not a subclass "
                "of 'FemMesh'".format(type(mesh))
            )
        if not mesh.locked:
            mesh.lock(create_mappers=True)
        super().__init__()
        self._mesh = mesh
        self._static_solver = None
        self._dynamic_solver_ = None
        self._natural_circular_frequencies = None
        self._essential_penalty = essential_penalty
        self._mass_penalty_ratio = mass_penalty_ratio
        self.clear_constraints()

    @property
    def mesh(self) -> FemMesh:
        """
        Returns the underlying mesh object.
        """
        return self._mesh

    @mesh.setter
    def mesh(self, value: FemMesh):
        """
        Sets the underlying mesh object.
        """
        if not isinstance(value, FemMesh):
            raise TypeError(
                "The type of 'value' is {},"
                " which is not a subclass "
                "of 'FemMesh'".format(type(value))
            )
        if not value.locked:
            value.lock(create_mappers=True)
        self._mesh = value

    @property
    def constraints(self) -> List[EBC]:
        """
        Returns the constraints of the structure as a list. Note that
        this is an independent mechanism from the 'fixity' data defined
        for the :class:`~sigmaepsilon.fem.pointdata.PointData` object
        of the underlying mesh instance.
        """
        if self._constraints is None:
            self._constraints = []
        return self._constraints

    def clear_constraints(self):
        """
        Clears up the constraints. This does not affect the 'fixity'
        definitions in the pointdata of the mesh.

        Note
        ----
        You can also use the `clear` method of the constrain list to get
        rid of previously defined constraints.

        Examples
        --------
        >>> structure.clear_constraints()

        is equivalent to

        >>> structure.constraints.clear()
        """
        self._constraints = None

    @flatten_pd(False)
    def natural_modes_of_vibration(self, *, flatten: bool = False, **kwargs) -> ndarray:
        """
        Returns natural modes of vibration.

        Parameters
        ----------
        flatten : bool, Optional
            If True, the result is a 1d array, otherwise 2d. Default is False.

        Returns
        -------
        numpy.ndarray
        """
        return self.mesh.pd.vshapes

    def natural_circular_frequencies(
        self, *args, return_vectors: bool = False, **kwargs
    ) -> Union[Tuple[ndarray], ndarray]:
        """
        Returns natural circular frequencies. The call forwards all parameters
        to :func:`sigmaepslion.solid.fem.dyn.natural_circular_frequencies`,
        see the docs there for the details.

        Parameters
        ----------
        return_vectors : bool, Optional
            To return eigenvectors or not. Default is False.

        Returns
        -------
        numpy.ndarray
            The natural circular frequencies.
        numpy.ndarray
            The eigenvectors, only if 'return_vectors' is True.

        See also
        --------
        :func:`sigmaepslion.solid.fem.dyn.natural_circular_frequencies`
        """
        f = self._dynamic_solver_.natural_circular_frequencies(*args, **kwargs)
        if return_vectors:
            v = self.natural_modes_of_vibration(flatten=True)
            return f, v
        return f

    def elastic_stiffness_matrix(
        self,
        *args,
        penalize: bool = False,
        sparse: bool = False,
        transform: bool = False,
        **kwargs
    ) -> Union[ndarray, coo_matrix]:
        """
        Returns the elastic stiffness matrix of the structure with dense or
        sparse layout.

        Parameters
        ----------
        penalize : bool, Optional
            If True, the result is the sum of the stiffness matrix and the
            penalty matrix of the essential boundary conditions.
        sparse : bool, Optional
            If True, the result is a sparse matrix. Default is False.
        transform : bool, Optional
            If True, local matrices are transformed to the global frame.
            Default is True.

        Returns
        -------
        numpy.ndarray or scipy.sparse.coo_matrix
        """
        if penalize:
            assert transform, "Must transform to penalize."
            assert sparse, "Penalization is only available for sparse results."
        if sparse:
            assert transform, "Must transform for a sparse output."
        params = dict(sparse=sparse, transform=transform)
        params.update(**kwargs)
        M_sparse = self.mesh.elastic_stiffness_matrix(**params)
        if penalize:
            M_sparse += self.mesh.essential_penalty_matrix()
        return M_sparse.tocoo()

    def essential_penalty_matrix(self, *args, **kwargs) -> coo_matrix:
        """
        Returns the penalty stiffness matrix of the structure.

        Returns
        -------
        scipy.sparse.coo_matrix
        """
        return self.mesh.essential_penalty_matrix(*args, **kwargs).tocoo()

    def consistent_mass_matrix(
        self,
        *,
        penalize: bool = False,
        sparse: bool = False,
        transform: bool = False,
        **kwargs
    ) -> Union[coo_matrix, ndarray]:
        """
        Returns the stiffness-consistent mass matrix of the structure.

        Parameters
        ----------
        penalize : bool, Optional
            If True, the result is the sum of the mass matrix and the
            penalty matrix of the essential boundary conditions.
        sparse : bool, Optional
            If True, the result is a sparse matrix. Default is False.
        transform : bool, Optional
            If True, local matrices are transformed to the global frame.
            Default is True.

        Returns
        -------
        numpy.ndarray or scipy.sparse.coo_matrix
        """
        if penalize:
            assert transform, "Must transform to penalize."
            assert sparse, "Penalization is only available for sparse results."
        if sparse:
            assert transform, "Must transform for a sparse output."
        params = dict(sparse=sparse, transform=transform)
        params.update(**kwargs)
        return self.mesh.consistent_mass_matrix(**params)

    def nodal_mass_matrix(self) -> coo_matrix:
        """
        Returns the nodal mass matrix of the structure.

        Returns
        -------
        scipy.sparse.coo_matrix
        """
        return self.mesh.nodal_mass_matrix().tocoo()

    def mass_matrix(self, penalize: bool = False) -> coo_matrix:
        """
        Returns the total mass matrix of the structure, which is the
        sum of the stiffness-consistent mass matrix derived from the
        densities of the cells and the nodal masses.

        Parameters
        ----------
        penalize : bool, Optional
            If True, the result is the sum of the mass matrix and the
            penalty matrix emenating from the enforcing of the essential
            boundary conditions using a Courant penalty function.

        Returns
        -------
        scipy.sparse.coo_matrix
        """
        M_sparse = self.mesh.mass_matrix()
        if penalize:
            M_sparse += self.mesh.essential_penalty_matrix()
        return M_sparse.tocoo()

    @flatten_pd(False)
    def nodal_dof_solution(self, *, flatten: bool = False, **__) -> ndarray:
        """
        Returns the vector of nodal displacements and optionally stores the result
        with a specified key.

        Parameters
        ----------
        flatten : bool, Optional
            If True, the result is a 1d array, otherwise 2d. Default is False.

        Returns
        -------
        numpy.ndarray
        """
        return self.mesh.pd.dofsol

    @flatten_pd(False)
    def reaction_forces(self, *, flatten: bool = False, **__) -> ndarray:
        """
        Returns the vector of reaction forces.

        Parameters
        ----------
        flatten : bool, Optional
            If True, the result is a 1d array, otherwise 2d. Default is False.

        Returns
        -------
        numpy.ndarray
        """
        return self.mesh.pd.reactions

    @flatten_pd(False)
    def nodal_forces(self, *_, flatten: bool = False, **__) -> ndarray:
        """
        Returns the vector of nodal forces from all sources. This is calculated
        during static analysis.

        Parameters
        ----------
        flatten : bool, Optional
            If True, the result is a 1d array, otherwise 2d. Default is False.
        """
        return self.mesh.pd.forces

    def internal_forces(self, *args, flatten: bool = False, **kwargs) -> ndarray:
        """
        Returns the internal forces for one or more elements.
        """
        return self.mesh.internal_forces(*args, flatten=flatten, **kwargs)

    def external_forces(self, *args, flatten: bool = False, **kwargs) -> ndarray:
        """
        Returns the external forces for one or more elements.
        """
        return self.mesh.external_forces(*args, flatten=flatten, **kwargs)

    def linsolve(self, *args, **kwargs) -> "Structure":
        """
        Performs a linear elastostatic calculation with pre- and
        post-processing.
        """
        return self.linear_static_analysis(*args, **kwargs)

    def linear_static_analysis(self, *args, **kwargs) -> "Structure":
        """
        Performs a linear elastostatic calculation with pre- and
        post-processing.
        """
        self._preproc_linstat_(*args, **kwargs)
        self._proc_linstat_(*args, **kwargs)
        return self._postproc_linstat_(*args, **kwargs)

    def free_vibration_analysis(self, *args, **kwargs) -> "Structure":
        """
        Performs a linear elastostatic calculation with pre- and
        post-processing.
        """
        self._preproc_free_vib_(*args, **kwargs)
        self._proc_free_vib_(*args, **kwargs)
        return self._postproc_free_vib_(*args, **kwargs)

    def effective_modal_masses(self, actions: ndarray, modes: ndarray) -> ndarray:
        """
        Returns effective modal masses of several modes of vibration.
        The call forwards all parameters to
        :func:`sigmaepslion.solid.fem.dyn.effective_modal_masses`, see the
        docs there for the details.

        Parameters
        ----------
        actions : numpy.ndarray
            One or more nodal displacement vectors representing rigid
            body motions.
        modes : numpy.ndarray
            Modes of vibration.

        Returns
        -------
        numpy.ndarray
            An array of effective mass values.

        See also
        --------
        :func:`~sigmaepslion.solid.fem.dyn.effective_modal_masses`
        """
        M_sparse = self.mesh.mass_matrix(penalize=False)
        return effective_modal_masses(M_sparse, actions, modes)

    def Rayleigh_quotient(self, u: ndarray, f: ndarray) -> ndarray:
        """
        Returns Rayleigh's quotient. The call forwards all parameters
        to :func:`~sigmaepslion.solid.fem.dyn.Rayleigh_quotient`, see the
        docs there for the details. If there are no actions specified,
        the function feeds it with the results from a linear elastic solution.

        Parameters
        ----------
        u : numpy.ndarray
            One or more nodal displacement vectors.
        f : numpy.ndarray
            One or more nodal load vectors.

        Returns
        -------
        float or numpy.ndarray
            One or more floats, depending on the input.

        See also
        --------
        :func:`~sigmaepslion.solid.fem.dyn.Rayleigh_quotient`
        """
        M_sparse = self.mass_matrix(penalize=True)
        return Rayleigh_quotient(M_sparse, u, f=f)

    def _initialize_(self, *_, **__) -> "Structure":
        """
        Initializes the structure. This includes data initialization
        for the cells and happens during the preprocessor. Returns the object
        for continuation.

        See also
        --------
        :func:`preprocess`
        """
        blocks = self.mesh.cellblocks(inclusive=True)
        for block in blocks:
            nE = len(block.celldata)
            # populate frames
            if not block.celldata.has_frames:
                frames = repeat(block.frame.show(), nE)
                block.celldata.frames = frames
        return self

    def _assemble_linstat_(self, *_, **__):
        mesh = self.mesh
        jagged = mesh.is_jagged()

        gnum = mesh.element_dof_numbering()
        # get raw data
        mesh.nodal_load_vector()
        if jagged:
            mesh.cell_load_vector(assemble=True, transform=True)
        else:
            mesh.cell_load_vector(assemble=False, transform=False)
        mesh.elastic_stiffness_matrix(sparse=False, transform=False, _jagged=jagged)
        # condensate
        mesh.condensate_cell_fixity()
        # get condensated data
        f_nodal = mesh.nodal_load_vector()
        if jagged:
            f_bulk = mesh.cell_load_vector(assemble=True, transform=True)
            K_bulk = mesh.elastic_stiffness_matrix(
                sparse=False,
                transform=True,
                _jagged=jagged,
            )
        else:
            f_bulk = mesh.cell_load_vector(assemble=False, transform=False)
            K_bulk = mesh.elastic_stiffness_matrix(
                sparse=False, transform=False, _jagged=jagged
            )

        if not jagged:
            # transform to global
            dcm = mesh.direction_cosine_matrix(target="global")
            K_bulk = tr_element_matrices_bulk(K_bulk, dcm)
            f_bulk = tr_nodal_loads_bulk(f_bulk, dcm)
            # assemble nodal load vector, the stiffness matrix gets assembled
            # in the solver
            nX = len(mesh.pd) * mesh.NDOFN
            f_bulk = assemble_load_vector(f_bulk, gnum, nX)
        f = f_nodal + f_bulk
        # penalize essential boundary conditions
        penalty = self._essential_penalty
        Kp = mesh.essential_penalty_matrix(penalty=penalty)
        fp = np.zeros(f.shape[0], dtype=float)
        for c in self.constraints:
            Kpc, fpc = c.assemble(mesh)
            Kp += Kpc
            fp += fpc
        if len(f.shape) == 1:
            f[:] += fp
        else:
            for i in range(f.shape[1]):
                f[:, i] += fp
        # create solver
        self._static_solver_ = StaticSolver(K_bulk, Kp.tocoo(), f, gnum, mesh=mesh)
        return self

    def _assemble_free_vib_(self, *_, **__):
        mesh = self.mesh
        gnum = mesh.element_dof_numbering()
        # get raw data
        mesh.nodal_load_vector()
        mesh.cell_load_vector(assemble=False, transform=False)
        mesh.elastic_stiffness_matrix(sparse=False, transform=False)
        mesh.consistent_mass_matrix(sparse=False, transform=False)
        # condensate
        mesh.condensate_cell_fixity()
        # get condensated data
        M_sparse = mesh.mass_matrix()
        K_bulk = mesh.elastic_stiffness_matrix(sparse=False, transform=False)
        # transform to global
        dcm = mesh.direction_cosine_matrix(target="global")
        K_bulk = tr_element_matrices_bulk(K_bulk, dcm)
        # penalize essential boundary conditions
        penalty = self._essential_penalty
        Kp = mesh.essential_penalty_matrix(penalty=penalty)
        for c in self.constraints:
            Kpc, _ = c.assemble(mesh)
            Kp += Kpc
        # create solver
        penalty_ratio = self._mass_penalty_ratio
        self._dynamic_solver_ = DynamicSolver(
            K_bulk,
            Kp.tocoo(),
            M_sparse,
            gnum,
            regular=False,
            mesh=mesh,
            penalty_ratio=penalty_ratio,
        )
        return self

    def _preproc_linstat_(self, *args, **kwargs) -> "Structure":
        self._initialize_(*args, **kwargs)
        self._assemble_linstat_(*args, **kwargs)
        return self

    def _preproc_free_vib_(self, *args, **kwargs) -> "Structure":
        self._initialize_(*args, **kwargs)
        self._assemble_free_vib_(*args, **kwargs)
        return self

    def _proc_linstat_(self, *args, **kwargs) -> "Structure":
        self._static_solver_.solve(*args, **kwargs)
        return self

    def _proc_free_vib_(self, *args, **kwargs) -> "Structure":
        self._dynamic_solver_.natural_circular_frequencies(*args, **kwargs)
        return self

    def _postproc_linstat_(self, *args, cleanup: bool = False, **kwargs) -> "Structure":
        solver = self._static_solver_
        mesh = self.mesh
        nDOFN = mesh.NDOFN
        nN = len(mesh.pd)
        # dof solution
        u = solver.u
        nRHS = 1 if len(u.shape) == 1 else u.shape[-1]
        if nRHS == 1:
            u = np.reshape(u, (nN, nDOFN))
        else:
            u = np.reshape(u, (nN, nDOFN, nRHS))
        self.mesh.pd.dofsol = u
        # reaction forces
        r = solver.r
        if nRHS == 1:
            r = np.reshape(r, (nN, nDOFN))
        else:
            r = np.reshape(r, (nN, nDOFN, nRHS))
        self.mesh.pd.reactions = r
        # nodal loads
        f = solver.f
        if nRHS == 1:
            f = np.reshape(f, (nN, nDOFN))
        else:
            f = np.reshape(f, (nN, nDOFN, nRHS))
        self.mesh.pd.forces = f
        # postproc block results
        self.mesh.postprocess(*args, **kwargs)
        if cleanup:
            self.cleanup()
        return self

    def _postproc_free_vib_(self, *, cleanup: bool = False, **__) -> "Structure":
        mesh = self.mesh
        solver = self._dynamic_solver_
        self._natural_circular_frequencies = solver.frequencies
        vshapes = solver.modal_shapes
        nN = len(mesh.pd)
        nDOF = mesh.__class__.NDOFN
        nS = vshapes.shape[-1]
        mesh.pd.vshapes = vshapes.reshape(nN, nDOF, nS)
        if cleanup:
            self.cleanup()
        return self

    def cleanup(self) -> "Structure":
        """
        Destroys the solver and returns the object for continuation.

        Returns
        -------
        Structure
        """
        self._static_solver_ = None
        self._dynamic_solver_ = None
        return self
