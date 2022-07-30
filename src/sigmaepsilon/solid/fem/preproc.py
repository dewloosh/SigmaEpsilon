# -*- coding: utf-8 -*-
from scipy.sparse import coo_matrix
import numpy as np
from numpy import ndarray

from ...math import squeeze
from ...math.array import isintegerarray, isfloatarray, \
    isboolarray, bool_to_float, atleastnd
from ...math.linalg.sparse.utils import lower_spdata, upper_spdata

from .utils import nodes2d_to_dofs1d, irows_icols_bulk, nodal_mass_matrix_data


def fem_coeff_matrix_coo(A: ndarray, *args, inds: ndarray = None,
                         rows: ndarray = None, cols: ndarray = None,
                         N: int = None, **kwargs):
    """
    Returns the coefficient matrix in sparse 'coo' format.
    Index mapping is either specified with `inds` or with
    both `rows` and `cols`.

    If `lower` or `upper` is provided as a positional argument, 
    the result is returned as a lower- or upper-triangular matrix, 
    respectively. 

    Parameters
    ----------
    A : np.ndarray[:, :, :]
        Element coefficient matrices in dense format.

    inds : np.ndarray[:, :], optional
        Global indices. The default is None.

    rows : np.ndarray[:], optional
        Global numbering of the rows. The default is None.

    cols : np.ndarray[:], optional
        Global numbering of the columns. The default is None.

    N : int, optional
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
        assert inds is not None, "Row and column indices (`rows` and `cols`) " \
            "or global index numbering array (inds) must be provided!"
        rows, cols = irows_icols_bulk(inds)

    data, rows, cols = A.flatten(), rows.flatten(), cols.flatten()

    if len(args) > 0:
        if 'lower' in args:
            data, rows, cols = lower_spdata(data, rows, cols)
        elif 'upper' in args:
            data, rows, cols = upper_spdata(data, rows, cols)

    return coo_matrix((data, (rows, cols)), shape=(N, N))


def build_fem_nodal_data(*args, inds: ndarray = None, values: ndarray = None,
                         N: int = None, **kwargs):
    """
    Returns nodal data of any sort in standard form.

    The data is given with 'values' and applies to the node indices specified
    by 'inds'. At all times, the function must be informed about the total number of 
    nodes in the model, to return an output with a correct shape. If 'inds' is None, it 
    is assumed that 'values' contains data for each node in the model, and therefore
    the length of 'values' is the number of nodes in the model. Otherwise, the number of nodes
    is inferred from 'inds' and 'N' (see the doc). 

    Parameters
    ----------

    inds : np.ndarray(int), optional
        1d numpy integer array specifying global indices for the rows of 'values'. 

    values : np.ndarray([float]), optional
        2d or 3d numpy array of floats, that specify some sort of data 
        imposed on nodes.
        If 3d, we assume that last axis stands for the number of load cases.

    N : int, optional
        The overall number of nodes in the model. If not provided, we assume that
        it equals the highest index in 'inds' + 1 (this assumes zero-based indexing).

    Notes
    -----
    The function returns the values as a 2d array, even if the input array was only 
    2 dimensional, which suggests a single load case. 

    Call 'fem_load_vector' with the same arguments to get the vector itself as a numpy array. 

    Returns
    -------
    (indices, nodal_loads, N) or None

        indices : int(:)
            1d numpy integer array of npde indices

        nodal_loads : float(:, :) 
            2d numpy float array of shape (nN * nDOF, nRHS)

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


@squeeze(True)
def fem_load_vector(*args, **kwargs) -> ndarray:
    """
    Assembles the right-hand-side of the global equation system.

    Parameters
    ----------
        See the documentation of 'build_fem_nodal_data' for the discription
        of the possible arguments.

    Returns
    -------
    numpy.ndarray(float)[:]
        The load vector as a numpy array of floats.
    """
    inds, values, N = build_fem_nodal_data(*args, **kwargs)
    nRHS = values.shape[-1]
    f = np.zeros((N, nRHS))
    f[inds] = values
    return f


def build_fem_ebc_data(*args, inds: ndarray = None, values: ndarray = None,
                       pfix: float = 1e12, **kwargs):
    """
    Returns fixity information in standard form. It extends the behaviour of 
    'build_fem_nodal_data' by allowing for boolean input.

    Parameters
    ----------
    inds : see 'build_fem_nodal_data'

    values : see 'build_fem_nodal_data'

    pfix : float, optional
        Penalty value for fixed dofs. It is used to transform boolean penalty
        data, or to make up for missing values (e.g. only indices are provided).
        Default value is 1e+12.

    Notes
    -----
    It is used to create a penalty matrix for a Courant-type penalization of the 
    essential boundary conditions.

    Call 'fem_penalty_matrix_coo' with the same arguments to get the penalty 
    matrix itself as a sparse scipy matrix in 'coo' format. 

    Returns
    -------
    see 'build_fem_nodal_data'

    """
    if values is not None:
        assert isinstance(values, np.ndarray)
        values = atleastnd(values, 3, back=True)  # (nP, nDOF, nLHS)
    else:
        assert isintegerarray(inds)
        raise NotImplementedError

    if isfloatarray(values):
        pass
    elif isboolarray(values):
        values = bool_to_float(values, pfix)
    else:
        raise NotImplementedError

    return build_fem_nodal_data(*args, inds=inds, values=values, **kwargs)


def fem_penalty_matrix_coo(*args, eliminate_zeros=True, sum_duplicates=True,
                           **kwargs) -> coo_matrix:
    """
    Returns the sparse, COO format penalty matrix, equivalent of 
    a Courant-type penalization of the essential boundary conditions.

    Parameters
    ----------
        See the documentation of 'build_fem_ebc_data' for the discription
        of the possible arguments.

    Returns
    -------
    scipy.sparse.coo_matrix
        The penalty matrix in sparse COO format.
    """
    inds, pen, N = build_fem_ebc_data(*args, **kwargs)
    K = coo_matrix((pen[:, 0], (inds, inds)), shape=(N, N))
    if eliminate_zeros:
        K.eliminate_zeros()
    if sum_duplicates:
        K.sum_duplicates()
    return K


def fem_nodal_mass_matrix_coo(*args, values=None, eliminate_zeros=True, 
                              sum_duplicates=True, ndof=6, **kwargs) -> coo_matrix:
    """
    Returns the diagonal mass matrix resulting form nodal masses in scipy COO format.

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
    inds, vals, N = build_fem_nodal_data(*args, values=values, **kwargs)
    M = coo_matrix((vals[:, 0], (inds, inds)), shape=(N, N))
    if eliminate_zeros:
        M.eliminate_zeros()
    if sum_duplicates:
        M.sum_duplicates()
    return M


if __name__ == '__main__':
    inds = np.array([2, 4, 12])
    pen = np.array([1e5, 1e5, 1e12])
    N = 100
    args = build_fem_ebc_data(inds=inds, values=pen, N=N)
