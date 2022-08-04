# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import coo_matrix as npcoo, csc_matrix as npcsc
import time

from sigmaepsilon.core import DeepDict

from .utils import irows_icols_bulk
from .linsolve import box_fem_data_bulk, solve_standard_form, unbox_lhs
from .eigsolve import eig_sparse, eig_dense, calc_eig_res
from .dyn import effective_modal_masses

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

class FemSolver:

    def __init__(self, K, Kp, f, gnum, imap=None, regular=True, M=None, **config):
        self.K = K
        self.M = M
        self.Kp = Kp
        self.gnum = gnum
        self.f = f
        self.vmodes = None  # vibration modes : Tuple(vals, vecs)
        self.config = config
        self.regular = regular
        self.imap = imap
        if imap is not None:
            # Ff imap.shape[0] > f.shape[0] it means that the inverse of
            # the mapping is given. It would require to store more data, but
            # it would enable to tell the total size of the equation system, that the
            # input is a representation of.
            assert imap.shape[0] == f.shape[0]
        self.regular = False if imap is not None else regular
        self.core = self.encode()
        self.READY = False
        self.summary = []
        self._summary = DeepDict()

    def encode(self) -> 'FemSolver':
        if self.imap is None and not self.regular:
            Kp, gnum, f, imap = box_fem_data_bulk(self.Kp, self.gnum, self.f)
            self.imap = imap
            return FemSolver(self.K, Kp, f, gnum, regular=True)
        else:
            return self

    def preproc(self, force=False):
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

    def update_stiffness(self, K):
        self.K = K
        self.preproc(force=True)

    def proc(self, preproc=False, solver=None):
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
            return self.decode()
        self.r = np.reshape(self.Ke.dot(self.u), self.f.shape) - self.f
        _dt = time.time() - _t0
        self._summary['postproc', 'time'] = _dt

    def linsolve(self, *args, solver=None, summary=False, **kwargs):
        self._summary = DeepDict()

        self.preproc()
        self.proc(solver=solver)
        self.postproc()

        self.summary.append(self._summary)
        
        if summary:
            return self.u, self.summary
        else:
            return self.u

    def decode(self):
        assert not self.regular
        N = self.gnum.max() + 1
        self.u = unbox_lhs(self.core.u, self.imap, N=N)
        self.r = unbox_lhs(self.core.r, self.imap, N=N)

    def natural_circular_frequencies(self, *args, k=10, return_vectors=False,
                                     maxiter=5000, normalize=True, as_dense=False, 
                                     **kwargs):
        """
        Returns the circular frequencies (\omega).
        """
        K = self.Ke + self.Kp
        if as_dense:
            vals, vecs = eig_dense(K, *args, M=self.M, nmode='M',
                                   normalize=normalize,
                                   return_residuals=False, **kwargs)
        else:
            vals, vecs = eig_sparse(K, *args, k=k, M=self.M, nmode='M',
                                    normalize=normalize, maxiter=maxiter,
                                    return_residuals=False, **kwargs)
        cfreqs = np.sqrt(vals)
        if return_vectors:
            return cfreqs, vecs
        return cfreqs

    def natural_cyclic_frequencies(self, *args, return_vectors=False, **kwargs):
        """
        Returns total oscillations done by the body in unit time (f).
        """
        kwargs['return_vectors'] = True
        vals, vecs = self.natural_circular_frequencies(*args, **kwargs)
        vals = vals / (2 * np.pi)
        if return_vectors:
            return vals, vecs
        return vals

    def natural_periods(self, *args, return_vectors=False, **kwargs):
        """
        Returns the times required to make a full cycle of vibration (T).
        """
        kwargs['return_vectors'] = True
        vals, vecs = self.natural_cyclic_frequencies(*args, **kwargs)
        vals = 1 / vals
        if return_vectors:
            return vals, vecs
        return vals

    def modes_of_vibration(self, *args, around=None, normalize=True, 
                           return_residuals=False, **kwargs):
        """
        Returns eigenvalues and eigenvectors as a tuple of two numpy arrays.        
        
        Notes
        -----
        Evalauated values are available as `obj.vmodes`.
        
        """
        if around is not None:
            sigma = (np.pi * 2 * around)**2
            kwargs['sigma'] = sigma
        self.vmodes = self.natural_cyclic_frequencies(
            *args, normalize=normalize, return_vectors=True, **kwargs)
        if return_residuals:
            vals, vecs = self.vmodes
            K = self.Ke + self.Kp
            r = calc_eig_res(K, self.M, vals, vecs)
            return self.vmodes, r
        return self.vmodes
    
    def effective_modal_masses(self, *args, action=None, modes=None, **kwargs):
        """
        Returns effective modal masses of several modes of vibration.
        
        Parameters
        ----------
        action : Iterable
            1d iterable, with a length matching the dof layout of the structure.
        
        modes : numpy array, Optional
            A matrix, whose columns are eigenmodes of interest.
            Default is None.
            
        Notes
        -----
        The sum of all effective masses equals the total mass of the structure.
            
        Returns
        -------
        numpy array
            An array of effective mass values.
        """
        if action is None:
            raise TypeError("No action is provided!")
        vecs = self.vmodes[1] if modes is None else modes
        return effective_modal_masses(self.M, action, vecs)
    
    def modal_participation_factors(self, *args, **kwargs):
        """
        Returns modal participation factors of several modes of vibration.
        
        The parameters are forwarded to `FemSolver.effective_modal_masses`.
        
        Notes
        -----
        The sum of all modal participation factors equals 1.
        
        Returns
        -------
        numpy array
            An array of values between 0 and 1.
        """
        m = self.effective_modal_masses(*args, **kwargs)
        return m / np.sum(m)
    
    def estimate_first_mode_of_vibration(self, *args, method='GA', **kwargs):
        pass
    
    def target_participation_factor(self, *args, action=None, **kwargs):
        """
        Returns the participation factor for the case when the structure
        hangs like a console under its own weight. 
        
        The idea of the modal response spectrum analysis suggests, that for 
        tall structures this shape should be close enough to the first mode of 
        vibration, thus provides means to estimate the largest modal participation 
        factor for practical scenarios.
        
        Parameters
        ----------
        action : Vector, Optional.
            An array specifying the direction of excitation. If not specified, the
            direction of action is estimated. Default is None.
                                
        Returns
        -------
        float
            The target modal participation factor.
        """
        pass
    
    def equivalent_nodal_forces(self, *args, action=None, **kwargs):
        pass