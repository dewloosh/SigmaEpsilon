from typing import Union, Tuple

from numba import njit, prange
import numpy as np
from numpy import ndarray
from scipy.sparse import spmatrix
import awkward as ak

from neumann.linalg.sparse import csr_matrix as csr, JaggedArray
from neumann import find1d, flatten2dC

__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def fixity2d_to_dofs1d(fixity2d: np.ndarray, inds: np.ndarray = None):
    """
    Returns the indices of the degrees of freedoms
    being supressed.

    Optionally, global indices of the rows in 'fixity2d'
    array can be provided by the optional argument 'inds'.

    Parameters
    ----------
    fixity2d : numpy.ndarray
        2d numpy array of booleans. It has as many rows as nodes, and as
        many columns as derees of freedom per node in the model. A True
        value means that the corresponding dof is fully supressed.
    inds : numpy.ndarray, Optional
        1d numpy array of integers, default is None. If provided, it should
        list the global indices of the nodes, for which the data is provided
        by the array 'fixity2d'.

    Returns
    -------
    dofinds : np.ndarray
        1d numpy array of integers.
    """
    ndof = fixity2d.shape[1]
    args = np.argwhere(fixity2d == True)
    N = args.shape[0]
    res = np.zeros(N, dtype=args.dtype)
    for i in prange(N):
        res[i] = args[i, 0] * ndof + args[i, 1]
    if inds is None:
        return res
    else:
        dofinds = np.zeros_like(res)
        for i in prange(len(inds)):
            for j in prange(ndof):
                dofinds[i * ndof + j] = inds[i] * ndof + j
        return dofinds[res]


@njit(nogil=True, parallel=True, cache=__cache)
def nodes2d_to_dofs1d(inds: np.ndarray, values: np.ndarray):
    """
    Returns a tuple of degree of freedom indices and data,
    based on a nodal definition.

    Parameters
    ----------
    inds : numpy.ndarray
        1d numpy array of integers, listing global node indices.
    values : numpy.ndarray
        3d numpy float array of shape (nN, nDOF, nX), listing values
        for each node in 'inds'. nX is the number of datasets.

    Returns
    -------
    dofinds : numpy.ndarray
        1d numpy array of integers, denoting global dof indices.
    dofvals : numpy.ndarray
        2d numpy float array of shape (nN * nDOF, nX), denoting values
        on dofs in 'dofinds'.
    """
    nN, dof_per_node, nRHS = values.shape
    dofinds = np.zeros((nN * dof_per_node), dtype=inds.dtype)
    dofvals = np.zeros((nN * dof_per_node, nRHS), dtype=values.dtype)
    for node in prange(nN):
        for dof in prange(dof_per_node):
            i = node * dof_per_node + dof
            dofinds[i] = inds[node] * dof_per_node + dof
            for rhs in prange(nRHS):
                dofvals[i, rhs] = values[node, dof, rhs]
    return dofinds, dofvals


@njit(nogil=True, cache=__cache, parallel=True)
def weighted_stiffness_bulk(K: np.ndarray, weights: np.ndarray):
    """
    Returns a weighted stiffness matrix.

    Parameters
    ----------
    K : numpy.ndarray
        2d numpy array of floats, where the first axis of K runs
        along the elements.
    weights : numpy.ndarray
        1d numpy array of floats.

    Returns
    -------
    Kw : numpy.ndarray
        2d numpy array of floats with the same shape as 'K'.
    """
    nE = len(weights)
    res = np.zeros_like(K)
    for i in prange(nE):
        res[i, :] = weights[i] * K[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def irows_icols_bulk(edofs: np.ndarray) -> Tuple[ndarray]:
    """
    Returns row and column index data for several finite elements.

    Parameters
    ----------
    edofs : numpy.ndarray
        2d numpy array. Each row has the meaning of global degree of
        freedom numbering for a given finite element.

    Returns
    -------
    irows : numpy.ndarray
        Global indices of the rows of the entries of stiffness matrices
        of the elements.
    icols : numpy.ndarray
        Global indices of the columns of the entries of stiffness matrices
        of the elements.

    Notes
    -----
    The implementation assumes that every cell has the same number of
    degrees of freedom.
    """
    nE, nNE = edofs.shape
    nTOTV = nNE * nNE
    irows = np.zeros((nE, nTOTV), dtype=edofs.dtype)
    icols = np.zeros((nE, nTOTV), dtype=edofs.dtype)
    for iNE in prange(nNE):
        for jNE in prange(nNE):
            i = iNE * nNE + jNE
            for iE in prange(nE):
                irows[iE, i] = edofs[iE, iNE]
                icols[iE, i] = edofs[iE, jNE]
    return irows, icols


@njit(nogil=True, parallel=False, cache=False)
def irows_icols_jagged(edofs: JaggedArray, irows, icols):
    """
    Returns row and column index data for several finite elements of different
    kinds.

    Parameters
    ----------
    edofs : JaggedArray
        2d jagged array. Each row has the meaning of global degree of
        freedom numbering for a given finite element.
    irows, icols
        Awkward array builders.
    """
    for iE in range(len(edofs)):
        dofs = edofs[iE]
        irows.begin_list()
        icols.begin_list()
        nDOF = len(dofs)
        for iDOF in range(nDOF):
            for jDOF in range(nDOF):
                irows.integer(dofs[iDOF])
                icols.integer(dofs[jDOF])
        irows.end_list()
        icols.end_list()


@njit(nogil=True, cache=__cache, parallel=True)
def irows_icols_bulk_filtered(edofs: np.ndarray, inds: np.ndarray) -> Tuple[ndarray]:
    """
    Returns row and column index data for finite elements specified
    by the index array `inds`.

    Parameters
    ----------
    edofs : numpy.ndarray
        2d numpy array. Each row has the meaning of global degree of
        freedom numbering for a given finite element.
    inds: numpy.ndarray
        1d numpy array of integers specifying active elements in an assembly.

    Returns
    -------
    irows : numpy.ndarray
        Global indices of the rows of the entries of stiffness matrices
        of the elements.
    icols : numpy.ndarray
        Global indices of the columns of the entries of stiffness matrices
        of the elements.

    Notes
    -----
    The implementation assumes that every cell has the same number of
    degrees of freedom.
    """
    nI = len(inds)
    nNE = edofs.shape[1]
    nTOTV = nNE * nNE
    irows = np.zeros((nI, nTOTV), dtype=edofs.dtype)
    icols = np.zeros((nI, nTOTV), dtype=edofs.dtype)
    for iNE in prange(nNE):
        for jNE in prange(nNE):
            i = iNE * nNE + jNE
            for iI in prange(nI):
                irows[iI, i] = edofs[inds[iI], iNE]
                icols[iI, i] = edofs[inds[iI], jNE]
    return irows, icols


def irows_icols(
    edofs: Union[JaggedArray, ndarray, ak.Array], inds: ndarray = None
) -> Tuple[Union[JaggedArray, ndarray]]:
    """
    Returns row and column indices for coefficient matrices of several finite
    elements.

    Parameters
    ----------
    edofs : numpy.ndarray or JaggedArray or awkward.Array
        2d numpy array. Each row has the meaning of coefficient numbering for a
        given finite element.
    inds: numpy.ndarray, Optional
        1d numpy array of integers specifying active elements in an assembly.
        Default is None.

    Returns
    -------
    Tuple[Union[JaggedArray, ndarray]]
        Two NumPy arrays or two JaggedArrays.
    """
    if isinstance(edofs, ndarray):
        if inds:
            return irows_icols_bulk_filtered(edofs, inds)
        else:
            return irows_icols_bulk(edofs)
    elif isinstance(edofs, ak.Array):
        if inds:
            raise NotImplementedError
        else:
            irows = ak.ArrayBuilder()
            icols = ak.ArrayBuilder()
            irows_icols_jagged(edofs, irows, icols)
            return JaggedArray(irows.snapshot()), JaggedArray(icols.snapshot())
    elif isinstance(edofs, JaggedArray):
        return irows_icols(edofs.to_ak(), inds)


@njit(nogil=True, cache=__cache, parallel=True)
def topo_to_gnum(topo: np.ndarray, ndofn: int):
    """
    Returns global dof numbering based on element
    topology data.

    Parameters
    ----------
    topo : numpy.ndarray
        2d numpy array of integers. Topology array listing global
        node numbers for several elements.
    ndofn : int
        Number of degrees of freedoms per node.

    Returns
    -------
    gnum : numpy.ndarray
        2d numpy array of integers.
    """
    nE, nNE = topo.shape
    gnum = np.zeros((nE, nNE * ndofn), dtype=topo.dtype)
    for i in prange(nE):
        for j in prange(nNE):
            for k in prange(ndofn):
                gnum[i, j * ndofn + k] = topo[i, j] * ndofn + k
    return gnum


@njit(nogil=True, cache=__cache, parallel=True)
def topo_to_gnum_jagged(topo, gnum, ndofn: int):
    """
    Returns global dof numbering based on element
    topology data.

    Parameters
    ----------
    topo : numpy.ndarray
        2d array of integers. Topology array listing global
        node numbers for several elements.
    gnum : numpy.ndarray or Awkward array
        2d buffer array for the results.
    ndofn : int
        Number of degrees of freedoms per node.

    Returns
    -------
    gnum : numpy.ndarray
        2d numpy array of integers.
    """
    nE, _ = topo.shape
    for i in prange(nE):
        for j in prange(len(topo[i])):
            for k in prange(ndofn):
                gnum[i, j * ndofn + k] = topo[i, j] * ndofn + k
    return gnum


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def penalty_factor_matrix(cellfixity: ndarray, shp: ndarray):
    nE, nNE, nDOF = cellfixity.shape
    N = nDOF * nNE
    res = np.zeros((nE, N, N), dtype=shp.dtype)
    for iE in prange(nE):
        for iNE in prange(nNE):
            for iDOF in prange(nDOF):
                fix = cellfixity[iE, iNE, iDOF]
                for jNE in prange(nNE):
                    i = jNE * nDOF + iDOF
                    for kNE in prange(nNE):
                        j = kNE * nDOF + iDOF
                        res[iE, i, j] += fix * shp[iNE, jNE] * shp[iNE, kNE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def approximation_matrix(ndf: ndarray, NDOFN: int):
    """
    Returns a matrix of approximation coefficients
    for all elements.
    """
    nE, nNE = ndf.shape[:2]
    N = nNE * NDOFN
    nappr = np.eye(N, dtype=ndf.dtype)
    res = np.zeros((nE, N, N), dtype=ndf.dtype)
    for iE in prange(nE):
        for i in prange(nNE):
            for j in prange(nNE):
                for ii in prange(NDOFN):
                    for jj in prange(NDOFN):
                        res[iE, i * 2 + ii, j * 2 + jj] = nappr[i, j] * ndf[iE, i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def nodal_approximation_matrix(ndf: ndarray):
    """
    Returns a matrix of nodal approximation coefficients
    for all elements.
    """
    nE, nNE = ndf.shape[:2]
    nappr = np.eye(nNE, dtype=ndf.dtype)
    res = np.zeros((nE, nNE, nNE), dtype=ndf.dtype)
    for iE in prange(nE):
        for i in prange(nNE):
            for j in prange(nNE):
                res[iE, i, j] = nappr[i, j] * ndf[iE, i]
    return res


@njit(nogil=True, cache=__cache)
def nodal_compatibility_factors(nam_csr_tot: csr, nam: ndarray, topo: ndarray):
    indptr = nam_csr_tot.indptr
    indices = nam_csr_tot.indices
    data = nam_csr_tot.data
    nN = len(indptr) - 1
    widths = np.zeros(nN, dtype=indptr.dtype)
    for iN in range(nN):
        widths[iN] = indptr[iN + 1] - indptr[iN]
    factors = dict()
    nreg = dict()
    for iN in range(nN):
        factors[iN] = np.zeros((widths[iN], widths[iN]), dtype=nam.dtype)
        nreg[iN] = indices[indptr[iN] : indptr[iN + 1]]
    nE, nNE = topo.shape
    for iE in range(nE):
        topoE = topo[iE]
        for jNE in range(nNE):
            nID = topo[iE, jNE]
            dataN = data[indptr[nID] : indptr[nID + 1]]
            topoN = nreg[nID]
            dataNE = np.zeros(widths[nID], dtype=nam.dtype)
            imap = find1d(topoE, topoN)
            dataNE[imap] = nam[iE, jNE]
            rNE = dataN - dataNE  # residual
            factors[nID] += np.outer(rNE, rNE)
    return factors, nreg


@njit(nogil=True, parallel=True, cache=__cache)
def compatibility_factors_to_coo(ncf: dict, nreg: dict):
    """
    ncf : nodal_compatibility_factors
    """
    nN = len(ncf)
    widths = np.zeros(nN, dtype=np.int32)
    for iN in prange(nN):
        widths[iN] = len(nreg[iN])
    shapes = (widths**2).astype(np.int64)
    N = np.sum(shapes)
    data = np.zeros(N, dtype=np.float64)
    rows = np.zeros(N, dtype=np.int32)
    cols = np.zeros(N, dtype=np.int32)
    c = 0
    for iN in range(nN):
        data[c : c + shapes[iN]] = flatten2dC(ncf[iN])
        nNN = widths[iN]
        for jNN in prange(nNN):
            for kNN in prange(nNN):
                rows[c + jNN * nNN + kNN] = nreg[iN][jNN]
                cols[c + jNN * nNN + kNN] = nreg[iN][kNN]
        c += shapes[iN]
    return data, rows, cols


@njit(nogil=True, parallel=True, cache=__cache)
def topo1d_to_gnum1d(topo1d: ndarray, NDOFN: int):
    nN = topo1d.shape[0]
    gnum1d = np.zeros(nN * NDOFN, dtype=topo1d.dtype)
    for i in prange(nN):
        for j in prange(NDOFN):
            gnum1d[i * NDOFN + j] = topo1d[i] * NDOFN + j
    return gnum1d


@njit(nogil=True, parallel=True, cache=__cache)
def ncf_to_cf(ncf: ndarray, NDOFN: int):
    nN = ncf.shape[0]
    cf = np.zeros((nN * NDOFN, nN * NDOFN), dtype=ncf.dtype)
    for i in prange(nN):
        for ii in prange(NDOFN):
            for j in prange(nN):
                cf[i * NDOFN + ii, j * NDOFN + ii] = ncf[i, j]
    return cf


@njit(nogil=True, cache=__cache)
def compatibility_factors(ncf: dict, nreg: dict, NDOFN: int):
    """ncf : nodal_compatibility_factors"""
    nN = len(ncf)
    widths = np.zeros(nN, dtype=np.int32)
    for iN in prange(nN):
        widths[iN] = len(nreg[iN])
    cf = dict()
    reg = dict()
    for iN in range(nN):
        cf[iN] = ncf_to_cf(ncf[iN], NDOFN)
        reg[iN] = topo1d_to_gnum1d(nreg[iN], NDOFN)
    return cf, reg


@njit(nogil=True, parallel=True, cache=__cache)
def element_dofmap_bulk(dofmap: ndarray, nDOF: int, nNODE: int):
    nDOF_ = dofmap.shape[0]
    res = np.zeros(nNODE * nDOF_, dtype=dofmap.dtype)
    for i in prange(nNODE):
        for j in prange(nDOF_):
            res[i * nDOF_ + j] = i * nDOF + dofmap[j]
    return res


@njit(nogil=True, parallel=False, cache=__cache)
def expand_shape_function_matrix_bulk(
    A_in: ndarray, A_out: ndarray, dofmap: ndarray
) -> ndarray:
    """
    Expands the shape function matrices into a standard form.
    """
    # nE, nP, nV, nX_in
    nX = A_in.shape[-1]
    for iX in prange(nX):
        A_out[:, :, :, dofmap[iX]] = A_in[:, :, :, iX]
    return A_out


@njit(nogil=True, parallel=True, cache=__cache)
def expand_coeff_matrix_bulk(A_in: ndarray, A_out: ndarray, dofmap: ndarray) -> ndarray:
    """
    Expands the local coefficient matrices into a standard form. It is used
    for instance when the total matrix is built up of smaller parts.
    """
    nDOF = dofmap.shape[0]
    nV = A_in.shape[0]
    for i in prange(nDOF):
        for j in prange(nDOF):
            for k in prange(nV):
                A_out[k, dofmap[i], dofmap[j]] = A_in[k, i, j]
    return A_out


@njit(nogil=True, parallel=True, cache=__cache)
def expand_load_vector_bulk(v_in: ndarray, v_out: ndarray, dofmap: ndarray) -> ndarray:
    """
    Expands the local load vectors into a standard form.
    """
    nDOF = dofmap.shape[0]
    nV0 = v_out.shape[0]
    nV1 = v_out.shape[2]
    for i in prange(nDOF):
        for j in prange(nV0):
            for k in prange(nV1):
                v_out[j, dofmap[i], k] = v_in[j, i, k]
    return v_out


@njit(nogil=True, parallel=True, cache=__cache)
def nodal_mass_matrix_data(nodal_masses: ndarray, ndof: int = 6):
    N = nodal_masses.shape[0]
    res = np.zeros((N * ndof))
    for i in prange(N):
        res[i * ndof : i * ndof + 3] = nodal_masses[i]
    return res


def min_stiffness_diagonal(K: Union[ndarray, spmatrix]) -> float:
    """
    Returns the minimum diagonal entry of the stiffness matrix.

    Parameters
    ----------
    K : Union[ndarray, spmatrix]
        The stiffness matrix as a dense or a sparse array.
    """
    if isinstance(K, spmatrix):
        return np.min(K.diagonal())
    else:
        i = np.arange(K.shape[-1])
        return np.min(K[:, i, i])
