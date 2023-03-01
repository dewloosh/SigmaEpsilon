import numpy as np
from numpy.linalg import norm
from scipy.sparse import isspmatrix as isspmatrix_np, spmatrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh


def calc_eig_res(A, M, eigvals, eigvecs) -> list:
    """
    Returns the residuals of the generalized eigenvalue problem

    .. math::
        :nowrap:

        \\begin{equation}
            \\mathbf{A} \\mathbf{x} = \\mathbf{v} \\mathbf{M} \\mathbf{x},
        \\end{equation}

    according to

    .. math::
        :nowrap:

        \\begin{equation}
            r_i = \\frac{|| (\\mathbf{A} - \\lambda_i \\mathbf{M}) ||}{|| \\mathbf{A} \\lambda_i ||},
        \\end{equation}

    Parameters
    ----------
    A : numpy.ndarray or scipy.linalg.sparse.spmatrix

    M : numpy.ndarray or scipy.linalg.sparse.spmatrix

    Returns
    -------
    list
        The residual for all eigenvectors.

    """

    def rfunc(i):
        return norm((A - eigvals[i] * M) @ eigvecs[:, i]) / norm(A @ eigvecs[:, i])

    return list(map(rfunc, range(len(eigvals))))


def normalize_eigenvectors(vecs, A):
    """
    Returns the eigenvectors normalized to the matrix `A`.

    Parameters
    ----------
    vecs : numpy.ndarray
        The eigenvectors as a 2d array, where each column is
        an eigenvector.

    A : scipy.linalg.sparse.spmatrix or numpy.ndarray
        A 2d array.

    Returns
    -------
    numpy.ndarray
        The normalized eigenvectors.

    """
    N = vecs.shape[-1]

    def qprod(i):
        return vecs[:, i].T @ A @ vecs[:, i]

    foo, rng = lambda i: vecs[:, i] / qprod(i), range(N)
    return np.stack(list(map(foo, rng))).T


def eig_dense(
    A,
    *args,
    M: np.ndarray = None,
    normalize: bool = False,
    nmode: str = "A",
    return_residuals: bool = False,
    **kwargs
):
    """
    Returns all eigenvectors end eigenvalues for a dense square
    matrix, for the standard eigenvalue problem

    .. math::
        :nowrap:

        \\begin{equation}
            \\mathbf{A} \\mathbf{x} = \\lambda \\mathbf{x},
        \\end{equation}

    or if :math:`\\mathbf{M}` is specified, the general eigenvalue problem

    .. math::
        :nowrap:

        \\begin{equation}
            \\mathbf{A} \\mathbf{x} = \\lambda \\mathbf{M} \\mathbf{x},
        \\end{equation}

    The values are calculated by calling :func:`scipy.linalg.eigs`.
    Extra keyword arguments are forwarded.

    Parameters
    ----------
    A : numpy.ndarray or scipy.linalg.sparse.spmatrix

    M : numpy.ndarray or scipy.linalg.sparse.spmatrix, Optional

    nmode : str, ['A' | 'M'], Optional
        Contrtols the normalization of the eigenvectors. Default is 'A'.

    normalize : bool, Optional
        Controls normalization of the eigenvectors. See the notes below.
        Default is False.

    See also
    --------
    scipy.linalg.eigs

    """
    A_ = A.todense() if isspmatrix_np(A) else A
    if M is None:
        vals, vecs = eigh(A_, **kwargs)
    else:
        M_ = M.todense() if isspmatrix_np(M) else M
        vals, vecs = eigh(A_, b=M_, **kwargs)
    if normalize:
        if nmode == "A":
            vecs = normalize_eigenvectors(vecs, A)
        elif nmode == "M":
            vecs = normalize_eigenvectors(vecs, M)
        else:
            raise NotImplementedError()
    if return_residuals:
        r = calc_eig_res(A, M, vals, vecs)
        return vals, vecs, r
    return vals, vecs


def eig_sparse(
    A,
    *args,
    k: int = 10,
    M: spmatrix = None,
    normalize: bool = False,
    which: str = "SM",
    maxiter: int = None,
    nmode: str = "A",
    return_residuals: bool = False,
    **kwargs
):
    """
    Returns all eigenvectors end eigenvalues for a sparse square matrix,
    for the standard eigenvalue problem

    .. math::
        :nowrap:

        \\begin{equation}
            \\mathbf{A} \\mathbf{x} = \\lambda \\mathbf{x},
        \\end{equation}

    or if :math:`\\mathbf{M}` is specified, the general eigenvalue problem

    .. math::
        :nowrap:

        \\begin{equation}
            \\mathbf{A} \\mathbf{x} = \\lambda \\mathbf{M} \\mathbf{x},
        \\end{equation}

    The values are calculated by calling :func:`scipy.sparse.linalg.eigsh`,
    which uses Arnoldi itrations. See references [1]_ and [2]_ for more details.
    Extra keyword arguments are forwarded.

    Parameters
    ----------
    A : scipy.linalg.sparse.spmatrix

    M : scipy.linalg.sparse.spmatrix, Optional

    k : int or str, Optional
        Number of eigendata to calculate. If `k` is a string, it must be 'all'.
        Default is 10.

    nmode : str, ['A' | 'M'], Optional
        Contrtols the normalization of the eigenvectors. Default is 'A'.

    normalize : bool, Optional
        Controls normalization of the eigenvectors. See the notes below.
        Default is False.

    which : str, ['LM' | 'SM' | 'LR' | 'SR' | 'LI' | 'SI'], Optional
        Which `k` eigenvectors and eigenvalues to find:
            'LM' : largest magnitude
            'SM' : smallest magnitude
            'LR' : largest real part
            'SR' : smallest real part
            'LI' : largest imaginary part
            'SI' : smallest imaginary part
        Default is 'SM'.

    See also
    --------
    :func:`scipy.sparse.linalg.eigsh`

    References
    ----------
    .. [1] ARPACK Software, http://www.caam.rice.edu/software/ARPACK/

    .. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang, ARPACK USERS GUIDE:
           Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
           Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

    """
    vals, vecs = eigsh(A=A, k=k, M=M, which=which, maxiter=maxiter, **kwargs)
    if normalize:
        if nmode == "A":
            vecs = normalize_eigenvectors(vecs, A)
        elif nmode == "M":
            vecs = normalize_eigenvectors(vecs, M)
        else:
            raise NotImplementedError()
    if return_residuals:
        r = calc_eig_res(A, M, vals, vecs)
        return vals, vecs, r
    return vals, vecs
