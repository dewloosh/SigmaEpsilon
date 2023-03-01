from typing import Union, Tuple

from scipy.sparse import coo_matrix
from scipy.sparse import spmatrix
import numpy as np
from numpy import ndarray
from numba import njit, prange

from neumann.logical import isintegerarray, isfloatarray, isboolarray
from neumann import bool_to_float, atleastnd, matrixform
from neumann.linalg.sparse.utils import lower_spdata, upper_spdata

from .fem import (
    nodes2d_to_dofs1d,
    irows_icols_bulk,
    nodal_mass_matrix_data,
    min_stiffness_diagonal,
)
from .imap import index_mappers, box_spmatrix, box_rhs, box_dof_numbering
from ...fem.constants import DEFAULT_DIRICHLET_PENALTY

__cache = True


def fem_coeff_matrix_coo(
    A: ndarray,
    *args,
    inds: ndarray = None,
    rows: ndarray = None,
    cols: ndarray = None,
    N: int = None
):
    """
    Returns the coefficient matrix in sparse 'coo' format.
    Index mapping is either specified with `inds` or with
    both `rows` and `cols`.

    If `lower` or `upper` is provided as a positional argument,
    the result is returned as a lower- or upper-triangular matrix,
    respectively.

    Parameters
    ----------
    A : np.ndarray
        Element coefficient matrices as a 3d array. The lengths if the last
        two axes must be the same.
    inds : np.ndarray, Optional
        Global indices of all quantities of an element as a 2d integer array.
        Length of the second axis must match the lengths of the 2nd and 3rd
        axes of 'A'.
        Default is None.
    rows : np.ndarray, Optional
        Global numbering of the rows as an 1d integer array. Length must
        equal the flattened length of 'A'. Default is None.
    cols : np.ndarray, Optional
        Global numbering of the columns as an 1d integer array. Length must
        equal the flattened length of 'A'. Default is None.
    N : int, Optional
        Total number of coefficients in the system. If not provided,
        it is inferred from index data.

    Returns
    -------
    scipy.sparse.coo_matrix
        Coefficient matrix in sparse coo format (see scipy doc for the details).
    """
    if N is None:
        assert inds is not None, "shape or `inds` must be provided!."
        N = inds.max() + 1
    if rows is None:
        assert inds is not None, (
            "Row and column indices (`rows` and `cols`) "
            "or global index numbering array (inds) must be provided!"
        )
        rows, cols = irows_icols_bulk(inds)

    data, rows, cols = A.flatten(), rows.flatten(), cols.flatten()

    if len(args) > 0:
        if "lower" in args:
            data, rows, cols = lower_spdata(data, rows, cols)
        elif "upper" in args:
            data, rows, cols = upper_spdata(data, rows, cols)

    return coo_matrix((data, (rows, cols)), shape=(N, N))


def _build_nodal_data(values: ndarray, *, inds: ndarray = None, N: int = None):
    """
    Returns nodal data of any sort in standard form.

    The data is given with 'values' and applies to the node indices specified
    by 'inds'. At all times, the function must be informed about the total number of
    nodes in the model, to return an output with a correct shape. If 'inds' is None, it
    is assumed that 'values' contains data for each node in the model, and therefore
    the length of 'values' is the number of nodes in the model. Otherwise, the number of
    nodes is inferred from 'inds' and 'N' (see the doc).

    Parameters
    ----------
    values : numpy.ndarray
        2d or 3d numpy array of floats, that specify some sort of data
        imposed on nodes for one or several datasets. If it is a 3d array, the length
        of the third axis equals the number of datasets.
    inds : numpy.ndarray, Optional
        1d numpy integer array specifying global indices for the rows of 'values'.
    N : int, Optional
        The overall number of nodes in the model. If not provided, we assume that
        it equals the highest index in 'inds' + 1 (this assumes zero-based indexing).

    Notes
    -----
    The function returns the values as a 2d array, even if the input array was only
    2 dimensional, which suggests a single load case.

    Returns
    -------
    indices : numpy.ndarray
        1d numpy integer array of npde indices
    nodal_loads : numpy.ndarray
        2d numpy float array of shape (nN * nDOF, nX), where nN, nDOF and nX
        are the number of nodes, degrees of freedom and datasets.
    N : int
        The overall size of the equation system, as derived from the input.
    """
    assert isfloatarray(values)

    # node based definition with multiple RHS (at least 1)
    values = atleastnd(values, 3, back=True)  # (nP, nDOF, nRHS)
    if inds is None:
        inds = np.arange(len(values))  # node indices
    else:
        assert isintegerarray(inds)
        assert len(values) == len(inds)

    # transform to nodal defintion
    inds, values = nodes2d_to_dofs1d(inds, values)

    N = inds.max() + 1 if N is None else N
    return inds, values, N


def nodal_load_vector(values: ndarray, **kwargs) -> ndarray:
    """
    Assembles the right-hand-side of the global equation system.

    Parameters
    ----------
    values : numpy.ndarray, Optional
        2d or 3d numpy array of floats. If 3d, it is assumed that
        the last axis runs along load cases.

    Returns
    -------
    numpy.ndarray
        The load vector as a 1d or 2d numpy array of floats.
    """
    inds, values, N = _build_nodal_data(values, **kwargs)
    nRHS = values.shape[-1]
    f = np.zeros((N, nRHS))
    f[inds] = values
    return f


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def assemble_load_vector(values: ndarray, gnum: ndarray, N: int = -1):
    """
    Returns global dof numbering based on element topology data.

    Parameters
    ----------
    values : numpy.ndarray
        3d numpy float array of shape (nE, nEVAB, nRHS), representing
        element data. The length of the second axis matches the the number of
        degrees of freedom per cell.
    gnum : int
        Global indices of local degrees of freedoms of elements.
    N : int, Optional
        The number of total unknowns in the system. Must be specified correcly,
        to get a vector the same size of the global system. If not specified, it is
        inherited from 'gnum' (as 'gnum.max() + 1'), but this can lead to a chopped
        array.

    Returns
    -------
    numpy.ndarray
        2d numpy array of integers with a shape of (N, nRHS), where nRHS is the number
        if right hand sizes (load cases).
    """
    nE, nEVAB, nRHS = values.shape
    if N < 0:
        N = gnum.max() + 1
    res = np.zeros((N, nRHS), dtype=values.dtype)
    for i in range(nE):
        for j in range(nEVAB):
            for k in range(nRHS):
                res[gnum[i, j], k] += values[i, j, k]
    return res


def essential_penalty_factor_matrix(
    fixity: ndarray = None,
    *,
    inds: ndarray = None,
    N: int = None,
    eliminate_zeros: bool = True,
    sum_duplicates: bool = True
) -> coo_matrix:
    """
    Returns the penalty factor matrix. This matrix is the basis of several penalty
    matrices used to enforce constraints on the primary variables.

    At all times, the function must be informed about the total number of nodes in the
    model, to return an output with a correct shape. If 'inds' is None, it is assumed that
    'fixity' contains data for each node in the model, and therefore the length of 'fixity'
    is the number of nodes in the model. You can also use 'N' to indicate the number of
    nodes, in which case entries according to 'inds' are marked as constrained.

    Parameters
    ----------
    fixity : numpy.ndarray, Optional
        Fixity information as a 2d boolean array.
    inds : numpy.ndarray, Optional
        1d numpy integer array specifying node indices.
    N : int, Optional
        The overall number of nodes in the model. If not provided, we assume that
        it equals the highest index in 'inds' + 1 (this assumes zero-based indexing).
    """
    if fixity is not None:
        assert isboolarray(fixity)
    else:
        if isintegerarray(inds):
            if not isinstance(N, int):
                N = inds.max() + 1  # assumes zero based indexing
            fixity = np.zeros(N, dtype=bool)
            fixity[inds] = True
    assert isboolarray(fixity), "Invalid input!"
    fixity = bool_to_float(fixity, 1.0, 0.0)
    fixity = atleastnd(fixity, 3, back=True)
    inds, fixity, N = _build_nodal_data(fixity, inds=inds, N=N)
    K = coo_matrix((fixity[:, 0], (inds, inds)), shape=(N, N))
    if eliminate_zeros:
        K.eliminate_zeros()
    if sum_duplicates:
        K.sum_duplicates()
    return K


def essential_penalty_matrix(
    fixity: ndarray = None,
    *,
    inds: ndarray = None,
    pfix: float = DEFAULT_DIRICHLET_PENALTY,
    eliminate_zeros: bool = True,
    sum_duplicates: bool = True
) -> coo_matrix:
    """
    Returns the sparse, COO format penalty matrix, equivalent of
    a Courant-type penalization of the essential boundary conditions.

    Parameters
    ----------
    fixity : numpy.ndarray, Optional
        Fixity information as a 2d float or boolean array.
    inds : numpy.ndarray, Optional
        1d integer array specifying global indices for the rows of 'values'.
    pfix : float, Optional
        Penalty value for fixed dofs. It is used to transform boolean penalty
        data, or to make up for missing values (e.g. only indices are provided).
        Default is `sigmaepsilon.fem.constants.DEFAULT_DIRICHLET_PENALTY`.
    eliminate_zeros : bool, Optional
        Eliminates zeros from the matrix. Default is True.
    eliminate_zeros : bool, Optional
        Summs duplicate entries in the matrix. Default is True.

    Returns
    -------
    scipy.sparse.coo_matrix
        The penalty matrix in sparse COO format.
    """
    if fixity is not None:
        assert isinstance(fixity, np.ndarray)
    else:
        if isintegerarray(inds):
            if not isinstance(N, int):
                N = inds.max() + 1  # assumes zero based indexing
            fixity = np.zeros(N, dtype=bool)
            fixity[inds] = True
    if isboolarray(fixity):
        fixity = bool_to_float(fixity, pfix)
    assert isfloatarray(fixity)
    fixity = atleastnd(fixity, 3, back=True)  # (nP, nDOF, nX)
    inds, fixity, N = _build_nodal_data(fixity, inds=inds)
    K = coo_matrix((fixity[:, 0], (inds, inds)), shape=(N, N))
    if eliminate_zeros:
        K.eliminate_zeros()
    if sum_duplicates:
        K.sum_duplicates()
    return K


def nodal_mass_matrix(
    *args,
    values: ndarray = None,
    eliminate_zeros: bool = True,
    sum_duplicates: bool = True,
    ndof: int = 6,
    **kwargs
) -> coo_matrix:
    """
    Returns the diagonal mass matrix resulting form nodal masses in scipy
    COO format.

    Parameters
    ----------
    See the documentation of 'build_fem_ebc_data' for the discription
    of the possible arguments.

    Returns
    -------
    scipy.sparse.coo_matrix
        The mass matrix in sparse COO format.
    """
    values = nodal_mass_matrix_data(values, ndof)
    inds, vals, N = _build_nodal_data(values, **kwargs)
    M = coo_matrix((vals[:, 0], (inds, inds)), shape=(N, N))
    if eliminate_zeros:
        M.eliminate_zeros()
    if sum_duplicates:
        M.sum_duplicates()
    return M


def estimate_stiffness_penalty(K: Union[ndarray, spmatrix], fixity: ndarray, N: int):
    """
    Returns a sugegsted value for the stiffness penalty, to account
    for essential boundary conditions.

    :math:`\\mathbf{M}`

    .. math::
        :nowrap:

        \\begin{equation}
            \\mathbf{A} \\mathbf{x} = \\lambda \\mathbf{x},
        \\end{equation}

    Parameters
    ----------
    K : Union[ndarray, spmatrix]
        The stiffness matrix as a dense or a sparse array.
    N : int
        The number of degrees of freedom in the model.

    References
    ----------
    .. [1] B. Nour-Omid, P. Wriggers "A two-level iteration method for
       solution of contact problems." Computer Methods in Applied Mechanics
       and Engineering, vol. 54, pp. 131-144, 1986.
    """
    eps = np.finfo(float).eps
    kmin = min_stiffness_diagonal(K)
    return kmin / np.sqrt(N * eps)


@njit(nogil=True, parallel=True, cache=__cache)
def _pull_submatrix(A: ndarray, r: ndarray, c: ndarray) -> ndarray:
    nR = r.shape[0]
    nC = c.shape[0]
    res = np.zeros((nR, nC), dtype=A.dtype)
    for i in prange(nR):
        for j in prange(nC):
            res[i, j] = A[r[i], c[j]]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _pull_subvector(v: ndarray, i: ndarray) -> ndarray:
    nI = i.shape[0]
    res = np.zeros((nI, v.shape[-1]), dtype=v.dtype)
    for j in prange(nI):
        res[j, :] = v[i[j], :]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _push_submatrix(A: ndarray, Asub: ndarray, r: ndarray, c: ndarray):
    nR = r.shape[0]
    nC = c.shape[0]
    for i in prange(nR):
        for j in prange(nC):
            A[r[i], c[j]] = Asub[i, j]


@njit(nogil=True, parallel=True, cache=__cache)
def _push_subvector(v: ndarray, vsub: ndarray, i: ndarray):
    nI = i.shape[0]
    for j in prange(nI):
        v[i[j], :] = vsub[j, :]


@njit(nogil=True, parallel=True, cache=__cache)
def condensate_Kf_bulk(
    K: ndarray, f: ndarray, fixity: ndarray
) -> Tuple[ndarray, ndarray]:
    """
    Returns the condensed coefficient matrices representing constraints
    on the internal forces of the elements (eg. hinges).

    Currently this solution is only able to handle two states, being totally free
    and being fully constrained. The fixity values are expected to be numbers between
    0 and 1, where dofs with a factor > 0.5 are assumed to be the constrained ones.

    Parameters
    ----------
    K : numpy.ndarray
        3d float array, the stiffness matrices of several elements of the same kind.
    factors: numpy.ndarray
        2d boolean array of connectivity factors for each dof of several elements.

    Returns
    -------
    numpy.ndarray
        The condensed stiffness matrices with the same shape as `K`.
    numpy.ndarray
        The condensed load vectors with the same shape as `f`.
    """
    nE = K.shape[0]
    K_out = np.zeros_like(K)
    f_out = np.zeros_like(f)
    for iE in prange(nE):
        b = np.where(fixity[iE])[0]
        i = np.where(~fixity[iE])[0]
        # stiffness matrix
        Kbb = _pull_submatrix(K[iE], b, b)
        Kbi = _pull_submatrix(K[iE], b, i)
        Kib = _pull_submatrix(K[iE], i, b)
        Kii = _pull_submatrix(K[iE], i, i)
        Kii_inv = np.linalg.inv(Kii)
        Kbb -= Kbi @ Kii_inv @ Kib
        _push_submatrix(K_out[iE], Kbb, b, b)
        # load vector
        fb = _pull_subvector(f[iE], b)
        fi = _pull_subvector(f[iE], i)
        fb -= Kbi @ Kii_inv @ fi
        _push_subvector(f_out[iE], fb, b)
    return K_out, f_out


@njit(nogil=True, parallel=True, cache=__cache)
def condensate_M_bulk(M: ndarray, fixity: ndarray) -> ndarray:
    """
    Returns the condensed coefficient matrices representing constraints
    on the internal forces of the elements (eg. hinges).

    Parameters
    ----------
    M : numpy.ndarray
        3d float array, the mass matrices of several elements of the same kind.
    fixity: numpy.ndarray
        2d float boolean of connectivity factors for each dof of several elements.

    Returns
    -------
    numpy.ndarray
        The condensed mass matrices with the same shape as `M`.
    """
    nE = M.shape[0]
    M_out = np.zeros_like(M)
    for iE in prange(nE):
        b = np.where(fixity[iE])[0]
        i = np.where(~fixity[iE])[0]
        Mbb = _pull_submatrix(M[iE], b, b)
        Mbi = _pull_submatrix(M[iE], b, i)
        Mib = _pull_submatrix(M[iE], i, b)
        Mii = _pull_submatrix(M[iE], i, i)
        Mbb -= Mbi @ np.linalg.inv(Mii) @ Mib
        _push_submatrix(M_out[iE], Mbb, b, b)
    return M_out


def assert_min_diagonals_bulk(K: ndarray, minval: float = 1e-12):
    """
    Guarantees, that the values in the diagonal are larger
    a prescribed minimum value.

    Parameters
    ----------
    A : numpy.ndarray
        The coefficient matrix of several elements as a 3d
        array, where elements run along the first axis.
    minval : float, Optional
        The minimum value. Default is 1e-12.

    Returns
    -------
    numpy.ndarray
        The modified coefficient matrix, with the same shape as 'K'.
    """
    inds = np.arange(K.shape[-1])
    d = K[:, inds, inds]
    eid, vid = np.where(d < minval)
    K[eid, vid, vid] = minval
    return K


def box_fem_data_sparse(K_coo: coo_matrix, Kp_coo: coo_matrix, f: ndarray):
    """
    Notes
    -----
    If the load vector 'f' is dense, it must contain values for all
    nodes, even the passive ones.
    """
    # data for boxing and unboxing
    loc_to_glob, glob_to_loc = index_mappers(K_coo, return_inverse=True)
    # boxing
    K = box_spmatrix(K_coo, glob_to_loc) + box_spmatrix(Kp_coo, glob_to_loc)
    f = box_rhs(matrixform(f), loc_to_glob)
    return K, f, loc_to_glob


def box_fem_data_bulk(Kp_coo: coo_matrix, gnum: ndarray, f: ndarray):
    """
    Notes
    -----
    If the load vector 'f' is dense, it must contain values for all
    nodes, even the passive ones.
    """
    # data for boxing and unboxing
    N = f.shape[0]
    loc_to_glob, glob_to_loc = index_mappers(gnum, N=N, return_inverse=True)
    # boxing
    gnum = box_dof_numbering(gnum, glob_to_loc)
    Kp_coo = box_spmatrix(Kp_coo, glob_to_loc)
    f = box_rhs(matrixform(f), loc_to_glob)
    return Kp_coo, gnum, f, loc_to_glob
