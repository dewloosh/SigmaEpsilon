# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from scipy.sparse import spmatrix
from scipy.sparse import coo_matrix as npcoo, csc_matrix as npcsc
import time
from typing import Union

from linkeddeepdict import LinkedDeepDict
from neumann.array import atleast2d

from .mesh import FemMesh
from .utils import irows_icols_bulk
from .linsolve import (box_fem_data_bulk, solve_standard_form, 
                       unbox_lhs)
from .dyn import (effective_modal_masses, Rayleigh_quotient, 
                  natural_circular_frequencies)
from .constants import DEFAULT_PENALTY_RATIO

"""
PARDISO MATRIX TYPES

1
real and structurally symmetric

2
real and symmetric positive definite

-2
real and symmetric indefinite

3
complex and structurally symmetric

4
complex and Hermitian positive definite

-4
complex and Hermitian indefinite

6
complex and symmetric

11
real and nonsymmetric

13
complex and nonsymmetric

"""

class Solver:
    """ 
    Base Class for Solvers. This is only a placeholder at the moment 
    for future work to be done.
    """


class FemSolver(Solver):
    """
    A class to perform solutions for linear elastostatics
    and dynamics.

    Parameters
    ----------
    K : numpy.ndarray
        The stiffness matrix in in dense format.
    
    Kp : scipy.sparse.spmatrix
        The penalty stiffness matrix of the essential boundary conditions.
    
    f : numpy.ndarray
        The load vector as an 1d or 2d numpy array.
        
    gnum : numpy.ndarray
        Global dof numbering as 2d integer array.
    
    imap : Union[dict, numpy.ndarray], Optional
        Index mapper. Default is None.
        
    regular : bool, Optional
        If True it is assumed that the structure is regular. Default is True.
        
    M : scipy.sparse.spmatrix, Optional
        The mass matrix. Default is None.
        
    mesh : FemMesh, Optional
        The mesh the input data belongs to. Only necessary for
        nonlinear calculations. Default is None.
        
    """

    def __init__(self, K : ndarray, Kp : spmatrix, f : ndarray, gnum: ndarray, 
                 imap : Union[dict, ndarray]=None, regular:bool=True, 
                 M:Union[spmatrix, ndarray]=None, mesh:FemMesh=None, **config):
        self.K = K
        self.M = M
        self.Kp = Kp
        self.gnum = gnum
        self.f = f
        self.vmodes = None  # vibration modes : Tuple(vals, vecs)
        self.config = config
        self.regular = regular
        self.imap = imap
        self.mesh = mesh
        if imap is not None:
            # Ff imap.shape[0] > f.shape[0] it means that the inverse of
            # the mapping is given. It would require to store more data, but
            # it would enable to tell the total size of the equation system, that the
            # input is a representation of.
            assert imap.shape[0] == f.shape[0]
        self.regular = False if imap is not None else regular
        self.core = self.box()
        self.READY = False
        self.summary = []
        self._summary = LinkedDeepDict()
    
    def elastic_stiffness_matrix(self, sparse:bool=True, 
                                 penalize:bool=True) -> Union[ndarray, spmatrix]:
        """
        Returns the stiffness matrix.
        
        Parameters
        ----------
        sparse : bool, Optional
            Set this true to get the result as a sparse 2d matrix, otherwise
            a 3d dense matrix is retured with matrices for all the individual elements.
            Default is True.
            
        penalize: bool, Optional
            If this is True, what is returendd is the sum of the elastic stiffness matrix
            and the penalty stiffness matrix of the essential boundary conditions.
            Default is True.
        
        Returns
        -------
        Union[numpy.ndarray, scipy.linalg.spmatrix]
            The stiffness matrix.
            
        """
        if not sparse:
            assert not penalize, "The penalty matrix is only available in sparse format."
            return self.K
        else:
            if penalize:
                return self.Ke + self.Kp
            else:
                return self.Ke
            
    def mass_matrix(self, penalize:bool=True, DEFAULT_PENALTY_RATIO:float=DEFAULT_PENALTY_RATIO) -> spmatrix:
        """
        Returns the consistent mass matrix of the solver.
        
        Parameters
        ----------            
        penalize: bool, Optional
            If this is True, what is returend is the sum of the stiffness matrix
            and the Courant-type penalty stiffness matrix of the essential boundary 
            conditions. Default is True.
        
        DEFAULT_PENALTY_RATIO : float, Optional
            The penalty ratio. Only if 'penalize' is True. 
            Default is `sigmaepsilon.solid.fem.config.DEFAULT_PENALTY_RATIO`.
        
        Returns
        -------
        scipy.linalg.spmatrix
            The mass matrix.
            
        """
        if penalize:
            return self.M + self.Kp * DEFAULT_PENALTY_RATIO
        else:
            return self.M

    def box(self) -> 'FemSolver':
        if self.imap is None and not self.regular:
            Kp, gnum, f, imap = box_fem_data_bulk(self.Kp, self.gnum, self.f)
            self.imap = imap
            return FemSolver(self.K, Kp, f, gnum, regular=True)
        else:
            return self
        
    def __setattr__(self, name, value):
        self.__dict__[name] = value
        
    def unbox(self):
        assert not self.regular
        N = self.gnum.max() + 1
        self.u = unbox_lhs(self.core.u, self.imap, N=N)
        self.r = unbox_lhs(self.core.r, self.imap, N=N)

    def preproc(self, force:bool=False):
        _t0 = time.time()
        if self.READY and not force:
            return
        self.N = self.gnum.max() + 1
        if self.config.get('sparsify', False):
            if not isinstance(self.f, npcsc):
                self.f = npcsc(self.f)
        self.krows, self.kcols = irows_icols_bulk(self.gnum)
        self.krows = self.krows.flatten()
        self.kcols = self.kcols.flatten()

        if not self.regular:
            self.core.preproc()
            self.Ke = self.core.Ke
        else:
            self.Ke = npcoo((self.K.flatten(), (self.krows, self.kcols)),
                            shape=(self.N, self.N), dtype=self.K.dtype)
        self.READY = True
        _dt = time.time() - _t0
        self._summary['preproc', 'regular'] = self.regular
        self._summary['preproc', 'time'] = _dt
        self._summary['preproc', 'N'] = self.N

    def update_stiffness(self, K:ndarray):
        self.K = K
        self.preproc(force=True)

    def proc(self, preproc:bool=False, solver:str=None):
        if preproc:
            self.preproc(force=True)
        _t0 = time.time()
        if not self.regular:
            self.core.proc()
            _dt = time.time() - _t0
            self._summary['proc']['time'] = _dt
            return self
        Kcoo = self.Ke + self.Kp
        Kcoo.eliminate_zeros()
        Kcoo.sum_duplicates()
        self.u, summary = solve_standard_form(
            Kcoo, self.f, summary=True, solver=solver)
        self._summary['proc'] = summary

    def postproc(self):
        _t0 = time.time()
        if not self.regular:
            self.core.postproc()
            self._summary['core'] = self.core._summary
            return self.unbox()
        self.r = np.reshape(self.Ke.dot(self.u), self.f.shape) - self.f
        _dt = time.time() - _t0
        self._summary['postproc', 'time'] = _dt

    def linsolve(self, *, solver:str=None) -> ndarray:
        """
        Performs a linear elastostatic analysis and returns the unknown
        coefficients as a numpy array.
        
        Parameters
        ----------
        solver : str, Optional
            The solver to use. Currently supported options are 'scipy' and 'pardiso'. 
            If nothing is specified, we prefer 'pardiso' if it is around, otherwise the 
            solver falls back to SciPy.
        
        Returns
        -------
        numpy.ndarray
            The vector of nodal displacements as a numpy array.
            
        """
        self._summary = LinkedDeepDict()
        self.preproc()
        self.proc(solver=solver)
        self.postproc()
        self.summary.append(self._summary)
        return self.u
        
    def natural_circular_frequencies(self, *args, return_vectors:bool=False, 
                                     DEFAULT_PENALTY_RATIO:float=DEFAULT_PENALTY_RATIO, **kwargs):
        """
        Returns natural circular frequencies. The call forwards all parameters
        to :func:`sigmaepslion.solid.fem.dyn.natural_circular_frequencies`, 
        see the docs there for the details. 
        
        Parameters
        ----------
        return_vectors : bool, Optional
            To return eigenvectors or not. Default is False.
            
        DEFAULT_PENALTY_RATIO : float, Optional
            The penalty ratio. Only if 'penalize' is True. 
            Default is `sigmaepsilon.solid.fem.config.DEFAULT_PENALTY_RATIO`.
                
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
        K = self.elastic_stiffness_matrix(penalize=True)
        M = self.mass_matrix(penalize=True, DEFAULT_PENALTY_RATIO=DEFAULT_PENALTY_RATIO)
        self.vmodes = natural_circular_frequencies(K, M, *args, return_vectors=True, **kwargs)
        if return_vectors:
            return self.vmodes
        else:
            return self.vmodes[0] 

    def effective_modal_masses(self, *, action:ndarray=None, modes:ndarray=None) -> ndarray:
        """
        Returns effective modal masses of several modes of vibration. 
        The call forwards all parameters to 
        :func:`sigmaepslion.solid.fem.dyn.effective_modal_masses`, see the 
        docs there for the details.
        
        See also
        --------
        :func:`sigmaepslion.solid.fem.dyn.effective_modal_masses`
        
        Returns
        -------
        numpy.ndarray
            An array of effective mass values.
            
        """
        if action is None:
            raise TypeError("No action is provided!")
        vecs = self.vmodes[1] if modes is None else modes
        return effective_modal_masses(self.M, action, vecs)
    
    def Rayleigh_quotient(self, u:ndarray=None, f:ndarray=None) -> ndarray:
        """
        Returns Rayleigh's quotient. The call forwards all parameters
        to :func:`sigmaepslion.solid.fem.dyn.Rayleigh_quotient`, see the 
        docs there for the details. If there are no actions specified, 
        the function feeds it with the results from a linear elastic solution.
        
        See also
        --------
        :func:`sigmaepslion.solid.fem.dyn.Rayleigh_quotient`
        
        Returns
        -------
        float or numpy.ndarray
            One or more floats.
            
        """
        M = self.mass_matrix(penalize=True)
        if isinstance(f, ndarray) and u is None:
            K = self.elastic_stiffness_matrix(penalize=True)
            K.eliminate_zeros()
            K.sum_duplicates()
            u = atleast2d(solve_standard_form(K, f), back=True)
        elif isinstance(u, ndarray) and f is None:
            K = self.elastic_stiffness_matrix(penalize=True)
            K.eliminate_zeros()
            K.sum_duplicates()
            f = K @ u
        elif f is None and u is None:
            assert self.u is not None, "A linear solution must be calculated prior to this."
            u = self.u
            return self.Rayleigh_quotient(u=u)    
        return Rayleigh_quotient(M, u, f=f)
                