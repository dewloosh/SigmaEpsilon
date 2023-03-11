from typing import Union

import numpy as np
from numpy import ndarray
from scipy.sparse import spmatrix
from scipy.sparse import coo_matrix as npcoo, csc_matrix as npcsc
import time

from linkeddeepdict import LinkedDeepDict
from neumann import matrixform
from neumann.linalg.sparse import JaggedArray

from .mesh import FemMesh
from ..utils.fem.fem import irows_icols
from ..utils.fem.linsolve import solve_standard_form, unbox_lhs
from ..utils.fem.imap import index_mappers, box_spmatrix, box_rhs, box_dof_numbering
from ..utils.fem.dyn import natural_circular_frequencies
from .constants import DEFAULT_MASS_PENALTY_RATIO


class Solver:
    """
    Base Class for solvers for the Finite Element Method.
    """


class StaticSolver(Solver):
    """
    A class to perform solutions for linear elastostatics
    and dynamics.

    Parameters
    ----------
    K : numpy.ndarray or JaggedArray
        The stiffness matrix in in dense format.
    P : scipy.sparse.spmatrix
        The penalty stiffness matrix of the essential boundary conditions.
    f : numpy.ndarray
        The load vector as an 1d or 2d numpy array.
    gnum : numpy.ndarray or JaggedArray
        Global dof numbering as 2d integer array.
    imap : Union[dict, numpy.ndarray], Optional
        Index mapper. Default is None.
    regular : bool, Optional
        If True it is assumed that the structure is regular. Default is False.
    mesh : FemMesh, Optional
        The mesh the input data belongs to. Only necessary for
        nonlinear calculations. Default is None.
    """

    def __init__(
        self,
        K: Union[ndarray, JaggedArray],
        P: spmatrix,
        f: ndarray,
        gnum: Union[ndarray, JaggedArray],
        regular: bool = None,
        M: ndarray = None,
        mesh: FemMesh = None,
        **config
    ):
        self.K_bulk = K
        self.M_bulk = M
        self.P_sparse = P
        self.gnum = gnum
        self.f = f
        self.u = None  # this is filled up during solution
        self.config = config
        self.regular = regular
        self.imap = None
        self.mesh = mesh
        self.regular = regular
        self.READY = False
        self.summary = []
        self._summary = LinkedDeepDict()
        if self.regular is None and self.mesh is not None:
            self.regular = self.mesh.is_regular()
        self.core = self._box_()
        self.krows, self.kcols = None, None

    def update_stiffness(self, K_bulk: Union[ndarray, JaggedArray]):
        self.K_bulk = K_bulk
        self._preproc_(force=True)

    def _box_(self) -> "StaticSolver":
        if self.imap is None and not self.regular:
            loc_to_glob, glob_to_loc = index_mappers(self.gnum, return_inverse=True)
            gnum = box_dof_numbering(self.gnum, glob_to_loc)
            P_sparse = box_spmatrix(self.P_sparse, glob_to_loc)
            f = box_rhs(matrixform(self.f), loc_to_glob)
            self.imap = glob_to_loc
            return StaticSolver(self.K_bulk, P_sparse, f, gnum, regular=True)
        else:
            return self

    def _unbox_(self):
        assert not self.regular
        N = np.max(self.gnum) + 1
        self.u = unbox_lhs(self.core.u, self.imap, N=N)
        self.r = unbox_lhs(self.core.r, self.imap, N=N)

    def _preproc_(self, force: bool = False):
        if self.READY and not force:
            return
        _t0 = time.time()
        self.u = None
        self.N = int(np.max(self.gnum) + 1)
        if self.config.get("sparsify", False):
            if not isinstance(self.f, npcsc):
                self.f = npcsc(self.f)

        if not self.regular:
            self.core._preproc_()
            self.K_sparse = self.core.K_sparse
        else:
            if self.krows is None:
                self.krows, self.kcols = irows_icols(self.gnum)
                self.krows = self.krows.flatten()
                self.kcols = self.kcols.flatten()
                self.kshape = self.K_bulk.shape
            self.K_sparse = npcoo(
                (self.K_bulk.flatten(), (self.krows, self.kcols)),
                shape=(self.N, self.N),
                dtype=float,
            )
        _dt = time.time() - _t0
        self.READY = True
        self._summary["preproc", "regular"] = self.regular
        self._summary["preproc", "time"] = _dt
        self._summary["preproc", "N"] = self.N

    def _proc_(self, preproc: bool = False, solver: str = None):
        if preproc:
            self._preproc_(force=True)
        _t0 = time.time()
        if not self.regular:
            self.core._proc_()
            _dt = time.time() - _t0
            self._summary["proc"]["time"] = _dt
            return self
        K = self.K_sparse + self.P_sparse
        K.eliminate_zeros()
        K.sum_duplicates()
        self.u, summary = solve_standard_form(K, self.f, summary=True, solver=solver)
        self._summary["proc"] = summary

    def _postproc_(self):
        _t0 = time.time()
        if not self.regular:
            self.core._postproc_()
            self._summary["core"] = self.core._summary
            return self._unbox_()
        self.r = np.reshape(self.K_sparse.dot(self.u), self.f.shape) - self.f
        _dt = time.time() - _t0
        self._summary["postproc", "time"] = _dt

    def solve(self, *, solver: str = None) -> ndarray:
        """
        Performs a linear elastostatic analysis and returns the unknown
        coefficients as a numpy array.

        Parameters
        ----------
        solver : str, Optional
            The solver to use. Currently supported options are 'scipy'
            and 'pardiso'. If nothing is specified, we prefer 'pardiso' if
            it is around, otherwise the solver falls back to SciPy.

        Returns
        -------
        numpy.ndarray
            The vector of nodal displacements as a numpy array.
        """
        self._summary = LinkedDeepDict()
        self._preproc_()
        self._proc_(solver=solver)
        self._postproc_()
        self.summary.append(self._summary)
        return self.u


class DynamicSolver(Solver):
    """
    A class to perform solutions for linear elastostatics
    and dynamics.

    Parameters
    ----------
    K : numpy.ndarray
        The stiffness matrix in dense format.
    P : scipy.sparse.spmatrix
        The penalty stiffness matrix of the essential boundary conditions.
    M : scipy.sparse.spmatrix
        The mass matrix. Default is None.
    gnum : numpy.ndarray
        Global dof numbering as 2d integer array.
    regular : bool, Optional
        If True it is assumed that the structure is regular. Default is True.
    mesh : FemMesh, Optional
        The mesh the input data belongs to. Only necessary for
        nonlinear calculations. Default is None.
    penalty_ratio : float, Optional
        Ratio of the penalty factors applied to the stiffness matrix (pK) and
        the mass matrix (pM) as pM/pK. If not provided, a default
        value of `sigmaepsilon.fem.constants.DEFAULT_MASS_PENALTY_RATIO`
        is used.
    """

    def __init__(
        self,
        K: ndarray,
        P: spmatrix,
        M: spmatrix,
        gnum: ndarray,
        regular: bool = False,
        mesh: FemMesh = None,
        penalty_ratio: float = DEFAULT_MASS_PENALTY_RATIO,
        **config
    ):
        self.K_bulk = K
        self.K_sparse = None
        self.M_sparse = M
        self.P_sparse = P
        self.gnum = gnum
        self.frequencies = None
        self.modal_shapes = None
        self.config = config
        self.regular = regular
        self.imap = None
        self.mesh = mesh
        self.regular = regular
        self.READY = False
        self.summary = []
        self._summary = LinkedDeepDict()
        self.penalty_ratio = penalty_ratio
        self.core = self._box_()

    def _box_(self) -> "DynamicSolver":
        if self.imap is None and not self.regular:
            _, glob_to_loc = index_mappers(self.gnum, return_inverse=True)
            gnum = box_dof_numbering(self.gnum, glob_to_loc)
            P_sparse = box_spmatrix(self.P_sparse, glob_to_loc)
            M_sparse = box_spmatrix(self.M_sparse, glob_to_loc)
            self.imap = glob_to_loc
            return DynamicSolver(
                self.K_bulk,
                P_sparse,
                M_sparse,
                gnum,
                regular=True,
                penalty_ratio=self.penalty_ratio,
            )
        else:
            return self

    def _unbox_(self):
        assert not self.regular
        N = self.gnum.max() + 1
        self.modal_shapes = unbox_lhs(self.core.modal_shapes, self.imap, N=N)
        self.frequencies = self.core.frequencies

    def _preproc_(self, force: bool = False):
        if self.READY and not force:
            return
        _t0 = time.time()
        self.N = self.gnum.max() + 1
        self.krows, self.kcols = irows_icols(self.gnum)
        self.krows = self.krows.flatten()
        self.kcols = self.kcols.flatten()

        if not self.regular:
            self.core._preproc_()
            self.K_sparse = self.core.K_sparse
        else:
            self.K_sparse = npcoo(
                (self.K_bulk.flatten(), (self.krows, self.kcols)),
                shape=(self.N, self.N),
                dtype=self.K_bulk.dtype,
            )
        _dt = time.time() - _t0
        self.READY = True
        self._summary["preproc", "regular"] = self.regular
        self._summary["preproc", "time"] = _dt
        self._summary["preproc", "N"] = self.N

    def _proc_free_(self, preproc: bool = False, **kwargs):
        if preproc:
            self._preproc_(force=True)
        _t0 = time.time()
        if not self.regular:
            self.core._proc_free_(**kwargs)
            _dt = time.time() - _t0
            self._summary["proc-free"]["time"] = _dt
            return self
        K_sparse = self.K_sparse + self.P_sparse
        M_sparse = self.M_sparse + self.P_sparse * self.penalty_ratio
        K_sparse.eliminate_zeros()
        K_sparse.sum_duplicates()
        freqs, shapes = natural_circular_frequencies(
            K_sparse, M_sparse, return_vectors=True, **kwargs
        )
        self.frequencies = freqs
        self.modal_shapes = shapes

    def _postproc_(self):
        _t0 = time.time()
        if not self.regular:
            self.core._postproc_()
            self._summary["core"] = self.core._summary
            return self._unbox_()
        _dt = time.time() - _t0
        self._summary["postproc", "time"] = _dt

    def natural_circular_frequencies(self, *, return_vectors: bool = False, **kwargs):
        """
        Returns natural circular frequencies. The call forwards all parameters
        to :func:`sigmaepslion.solid.fem.dyn.natural_circular_frequencies`,
        see the docs there for the details.

        Parameters
        ----------
        return_vectors : bool, Optional
            To return eigenvectors or not. Default is False.
        penalty_ratio : float, Optional
            The penalty ratio. Only if 'penalize' is True.
            Default is `sigmaepsilon.fem.config.DEFAULT_PENALTY_RATIO`.

        See also
        --------
        :func:`sigmaepslion.solid.fem.dyn.natural_circular_frequencies`

        Returns
        -------
        numpy.ndarray
            The natural circular frequencies.
        numpy.ndarray
            The eigenvectors, only if 'return_vectors' is True.
        """
        if self.frequencies is None:
            self._preproc_()
            self._proc_free_(**kwargs)
            self._postproc_()
        if return_vectors:
            return self.frequencies, self.modal_shapes
        return self.frequencies
