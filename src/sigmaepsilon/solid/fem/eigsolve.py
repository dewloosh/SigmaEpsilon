# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import norm
from scipy.sparse import isspmatrix as isspmatrix_np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh


def calc_eig_res(A, M, eigvals, eigvecs):
    """
    Returns the residuals of the generalized eigenvalue problem

    ``A * x[i] = w[i] * M * x[i]``
    
    according to
    
    ``res[i] = || (A - eigvals[i] * M) @ eigvecs[i] || / || A @ eigvecs[i] ||``
    
    """
    def rfunc(i): return norm((A - eigvals[i] * M) @ eigvecs[:, i]) / \
            norm(A @ eigvecs[:, i])
    return list(map(rfunc, range(len(eigvals))))


def normalize_eigenvectors(vecs, A):
    """
    Returns the eigenvectors normalized to the matrix `A`.
    """
    N = vecs.shape[-1]
    def qprod(i): return vecs[:, i].T @ A @ vecs[:, i]
    foo, rng = lambda i: vecs[:, i] / qprod(i), range(N)
    return np.stack(list(map(foo, rng))).T


def eig_dense(A, *args, M=None, normalize=False, nmode='A',
              return_residuals=False, **kwargs):
    """
    Returns all eigenvectors end eigenvalues for a dense Hermitian matrix. 

    The values are calculated by calling scipy.linalg.eigs.
    Extra keyword arguments are forwarded.

    Parameters
    ----------
    nmode : str, Optional
        'M' or 'G' for dynamic or stability analysis.

    normalize : bool, optional
        Controls normalization of the eigenvectors. See the notes below. 
        Default is False.

    Notes
    -----
    If `nmode` is 'M' and `normalize` is True, the eigenvectors are 
    normalized to M.

    """
    A_ = A.todense() if isspmatrix_np(A) else A
    if M is None:
        vals, vecs = eigh(A_, **kwargs)
    else:
        M_ = M.todense() if isspmatrix_np(M) else M
        vals, vecs = eigh(A_, b=M_, **kwargs)
    if normalize:
        if nmode == 'A':
            vecs = normalize_eigenvectors(vecs, A)
        elif nmode == 'M':
            vecs = normalize_eigenvectors(vecs, M)
        else:
            raise NotImplementedError()
    if return_residuals:
        r = calc_eig_res(A, M, vals, vecs)
        return vals, vecs, r
    return vals, vecs


def eig_sparse(A, *args, k=10, M=None, normalize=False, which='SM',
               maxiter=None, nmode='A', return_residuals=False, **kwargs):
    """
    Returns eigenvectors end eigenvalues for a sparse Hermitian matrix. 

    The values are calculated by calling scipy.sparse.linalg.eigsh, 
    which uses Arnoldi itrations. See references [1, 2] for more details. 
    Extra keyword arguments are forwarded.

    Parameters
    ----------
    k : int or str, Optional
        Number of eigendata to calculate. If `k` is a string, it must be 'all'. 
        Default is 10.

    nmode : str, Optional
        'M' or 'G' for dynamic or stability analysis.

    normalize : bool, optional
        Controls normalization of the eigenvectors. See the notes below. 
        Default is False.

    which : str, ['LM' | 'SM' | 'LR' | 'SR' | 'LI' | 'SI'], optional
        Which `k` eigenvectors and eigenvalues to find:
            'LM' : largest magnitude
            'SM' : smallest magnitude
            'LR' : largest real part
            'SR' : smallest real part
            'LI' : largest imaginary part
            'SI' : smallest imaginary part
        Default is 'SM'.

    Notes
    -----
    If `nmode` is 'M' and `normalize` is True, the eigenvectors are 
    normalized to M.

    References
    ----------
    .. [1] ARPACK Software, http://www.caam.rice.edu/software/ARPACK/
    .. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang, ARPACK USERS GUIDE:
        Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
        Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

    """
    vals, vecs = eigsh(A=A, k=k, M=M, which=which, maxiter=maxiter, **kwargs)
    if normalize:
        if nmode == 'A':
            vecs = normalize_eigenvectors(vecs, A)
        elif nmode == 'M':
            vecs = normalize_eigenvectors(vecs, M)
        else:
            raise NotImplementedError()
    if return_residuals:
        r = calc_eig_res(A, M, vals, vecs)
        return vals, vecs, r
    return vals, vecs
