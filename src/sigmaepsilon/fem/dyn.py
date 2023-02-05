from typing import Union, Tuple
from numpy import ndarray
from scipy.sparse import spmatrix
import numpy as np

from neumann import atleast2d

from .linsolve import solve_standard_form
from .eigsolve import eig_sparse, eig_dense


ArrayLike = Union[ndarray, spmatrix]


def effective_modal_mass(M: ArrayLike, action: ndarray, mode: ndarray) -> float:
    """
    Returns the effective modal mass for a specific mode.

    Assumes that the modal shapes are normalized to the mass matrix.

    Parameters
    ----------
    M : numpy.ndarray
        2d mass matrix as a NumPy or SciPy 2d float array.
    action : Iterable
        1d iterable, with a length matching the dof layout of the structure.
    mode : numpy.ndarray
        1d array representing a modal shape.

    Returns
    -------
    float
        The effective modal mass.

    """
    return (mode.T @ M @ action) ** 2


def effective_modal_masses(M: ArrayLike, action: ndarray, modes: ndarray) -> ndarray:
    """
    Returns the effective modal mass for several modes.

    Assumes that the modal shapes are normalized to the mass matrix.

    Parameters
    ----------
    M : numpy.ndarray
        Mass matrix as a NumPy or SciPy 2d float array.
    action : numpy.ndarray
        1d iterable, with a length matching the dof layout of the structure.
    modes : numpy.ndarray
        A matrix, whose columns are eigenmodes of interest.

    Notes
    -----
    The sum of all effective masses equals the total mass of the structure.

    Returns
    -------
    numpy.ndarray
        1d float array of effective modal masses

    """
    res = []
    for i in range(modes.shape[-1]):
        m_eff = effective_modal_mass(M, action, modes[:, i])
        res.append(m_eff)
    return np.array(res)


def Rayleigh_quotient(
    M: spmatrix, *, K: spmatrix = None, u: ndarray = None, f: ndarray = None, **kw
) -> ndarray:
    """
    Returns Rayleigh's quotient

    .. math::
        :nowrap:

        \\begin{equation}
            \\frac{\\mathbf{v}^T \\mathbf{K} \\mathbf{v}}{\\mathbf{v}^T
            \\mathbf{M} \\mathbf{v}},
        \\end{equation}

    for a prescribed action :math:`\\mathbf{v}`.

    Parameters
    ----------
    M : scipy.linalg.sparse.spmatrix
        The mass matrix.
    u : numpy.ndarray, Optional
        The vector of nodal displacements (1d or 2d).
    K : scipy.linalg.sparse.spmatrix, Optional
        The stiffness matrix. It can be omitted, if 'f' is provided.
    f : numpy.ndarray, Optional
        The vector of nodal forces (1d or 2d). It can be omitted,
        if 'K' is provided.

    Notes
    -----
    The Rayleigh quotient is

    - higher than, or equal to the square of the smallest natural
      circular frequency,

    - smaller than, or equal to the square of the highest natural
      circular frequency.

    Returns
    -------
    numpy.ndarray
        An 1d array of floats.

    """
    if isinstance(f, ndarray) and u is None:
        u = atleast2d(solve_standard_form(K, f), back=True)
    elif isinstance(u, ndarray) and f is None:
        f = K @ u
    res = []
    for i in range(u.shape[-1]):
        nom = u[:, i].T @ f[:, i]
        denom = u[:, i].T @ M @ u[:, i]
        res.append(nom / denom)
    return np.array(res)


def natural_circular_frequencies(
    K: spmatrix,
    M: spmatrix,
    *,
    k: int = 10,
    return_vectors: bool = False,
    maxiter: int = 5000,
    normalize: bool = True,
    as_dense: bool = False,
    around: float = None,
    nmode: str = "M",
    which: str = "SM",
    **kwargs
) -> Tuple[ndarray]:
    """
    Returns the natural circular frequencies :math:`\omega_{0i}` and optionally the
    corresponding eigenvectors as (not trivial) solutions to the eigenproblem

    .. math::
        :nowrap:

        \\begin{equation}
            \\left( \\mathbf{K} - \\omega^2 \\mathbf{M} \\right)
            \\mathbf{v} = \\mathbf{0}.
        \\end{equation}

    Parameters
    ----------
    K : scipy.linalg.sparse.spmatrix
        The stiffness matrix in sparse format.
    M : scipy.linalg.sparse.spmatrix
        The mass matrix in sparse format.
    k : int, Optional
        The number of solutions to return. Only if 'as_dense' is False.
        Default is 10.
    return_vectors : bool, Optional
        To return eigenvectors or not. Default is False.
    maxiter : int, Optional
        Maximum number of iterations, if solved using sparse matrices.
        Only if 'as_dense' is False. Default is 5000.
    normalize : bool, Optional
        If True, the returned eigenvectors are normalized to the mass matrix.
        Only if 'return_vectors' is True. Default is True.
    as_dense : bool, Optional
        If True, the stiffness and mass matrices are handled as dense arrays.
        In this case, if 'return_vectors' is True, all eigenvalues and vectors
        are returned, regardless of other parameters.
    around : float, Optional
        A target (possibly an approximation) value around which eigenvalues
        are searched. Default is None.
    which : str, ['LM' | 'SM' | 'LA' | 'SA' | 'BE' | 'LR' | 'SR'
                 | 'LI' | 'SI'], Optional
        Which `k` eigenvectors and eigenvalues to find:

            'LM' : largest magnitude

            'SM' : smallest magnitude

            'LA' : Largest (algebraic) eigenvalues.

            'SA' : Smallest (algebraic) eigenvalues.

            'BE' : Half (k/2) from each end of the spectrum.

            'LR' : largest real part

            'SR' : smallest real part

            'LI' : largest imaginary part

            'SI' : smallest imaginary part

        Only if 'as_dense' is False. Default is 'LM'.
    **kwargs : dict, Optional
        Keyword arguments are forwarded to the appropriate routine of SciPy.

    Note
    ----
    Calculation is done using the relavant routines of Scipy,
    see its documentation for more control over the parameters.

    See also
    --------
    :func:`scipy.linalg.eigs`
    :func:`scipy.sparse.linalg.eigsh`

    Returns
    -------
    numpy.ndarray
        The natural circular frequencies.
    numpy.ndarray
        The eigenvectors, only if 'return_vectors' is True.

    """
    if around is not None:
        kwargs["sigma"] = around**2

    if as_dense:
        vals, vecs = eig_dense(
            K, M=M, nmode=nmode, normalize=normalize, return_residuals=False, **kwargs
        )
    else:
        vals, vecs = eig_sparse(
            K,
            k=k,
            M=M,
            which=which,
            normalize=normalize,
            maxiter=maxiter,
            return_residuals=False,
            nmode=nmode,
            **kwargs
        )
    cfreqs = np.sqrt(vals)
    if return_vectors:
        return cfreqs, vecs
    return cfreqs


def estimate_smallest_natural_circular_frequency(*args, **kwargs) -> ndarray:
    """
    Returns a lower bound estimation of the smallest natural frequency using
    Rayleigh's quotient. See :func:`Rayleigh_quotient` for the input.

    Notes
    -----
    This relies on the equivalence of elastic internal energy and kinetic
    energy of undamped conservative systems.

    Returns
    -------
    numpy.ndarray
        An 1d array of floats.

    See also
    --------
    :func:`Rayleigh_quotient`

    """
    return np.sqrt(Rayleigh_quotient(*args, **kwargs))
