import numpy as np
from numpy import ndarray
import numpy as np
from scipy.sparse import coo_matrix as coo, spmatrix, isspmatrix as isspmatrix_np
from scipy.sparse.linalg import spsolve, spsolve_triangular, splu
from typing import Union
from time import time

# from concurrent.futures import ThreadPoolExecutor

from linkeddeepdict import LinkedDeepDict
from neumann import matrixform

from .imap import unbox_lhs
from .preproc import fem_coeff_matrix_coo, box_fem_data_sparse
from ...config import __haspardiso__

if __haspardiso__:
    import pypardiso as ppd
    from pypardiso import PyPardisoSolver

    # from pypardiso.scipy_aliases import pypardiso_core

arraylike = Union[ndarray, spmatrix]


def solve_standard_form(
    K: coo,
    f: np.ndarray,
    *args,
    use_umfpack: bool = True,
    summary: bool = False,
    permc_spec: str = "COLAMD",
    solver: str = None,
    mtype: int = 11,
    assume_regular: bool = False,
    **_
):
    """
    Solves the discrete equilibrium equations :math:`\mathbf{K} \mathbf{u} = \mathbf{f}`
    for :math:`\mathbf{u}`.

    Parameters
    ----------
    K : scipy.sparse.spmatrix
        The stiffness matrix in coo format.

    f : numpy.ndarray
        The vector of nodal loads.

    use_umfpack : bool, Optional
        Only if solver is 'scipy'. See the documentation :func:`scipy.sparse.linalg.spsolve`
        for further meaning. Default is True.

    permc_spec : str, Optional
        Only if solver is 'scipy'. See the documentation :func:`scipy.sparse.linalg.spsolve`
        for further meaning.

    solver : str, Optional
        The solver to use. Currently supported options are 'scipy' and 'pardiso'. If nothing is
        specified, we prefer 'pardiso' if it is around, otherwise the solver falls back to SciPy.

    mtype : int, Optional
        Matrix type indicator for the pypardiso solver. The Default value is 11, which denassumes
        a real and nonsymmetric coefficient matrix.

    assume_regular : bool, Optional
        If True, it is assumed that the input is according to the standard form, otherwise
        some preprocessing on the inputs is performed.

    Returns
    -------
    numpy.ndarray
        The solution as a numpy array. The returned array has the same shape as `f`.

    dict, Optional
        A summary including the most important characteristics of the solution process.
        Only if 'summary' is True.

    See also
    --------
    :func:`scipy.sparse.linalg.spsolve`
    :func:`scipy.sparse.linalg.splu`

    """
    solver = "pardiso" if __haspardiso__ else "scipy" if solver is None else solver
    if solver == "pardiso" and not __haspardiso__:
        raise ImportError(
            "You need to install 'pypardiso' for solver type <{}>".format(solver)
        )

    if not assume_regular:
        K.eliminate_zeros()
        K.sum_duplicates()
        f = matrixform(f)

    if solver == "pardiso":
        if mtype == 11:
            t0 = time()
            u = ppd.spsolve(K, f)
            dt = time() - t0
        else:
            pds = PyPardisoSolver(mtype=mtype)
            t0 = time()
            u = pds.solve(K, f)
            dt = time() - t0
    elif solver == "scipy":
        if len(args) > 0:
            if "lower" in args:
                solver = "scipy.sparse.linalg.spsolve_triangular (lower)"
                t0 = time()
                u = spsolve_triangular(K, f, lower=True)
                dt = time() - t0
            elif "upper" in args:
                solver = "scipy.sparse.linalg.spsolve_triangular (upper)"
                t0 = time()
                u = spsolve_triangular(K, f, lower=False)
                dt = time() - t0
            elif "SLU" in args:
                solver = "scipy.sparse.linalg.SuperLU"
                K = K.tocsc()
                t0 = time()
                LU = splu(K)
                u = LU.solve(f)
                dt = time() - t0
        else:
            if use_umfpack:
                use_umfpack = True if f.shape[1] == 1 else False
            solver = "scipy.sparse.linalg.spsolve"
            t0 = time()
            u = spsolve(K, f, permc_spec=permc_spec, use_umfpack=use_umfpack)
            dt = time() - t0
    else:
        raise NotImplementedError(
            "Selected solver '{}' is not supported!".format(solver)
        )

    if isspmatrix_np(u):
        u = u.todense()
    u = u.reshape(f.shape)

    # residual
    if summary:
        d = LinkedDeepDict(
            **{"time": dt, "N": u.shape[0], "solver": solver, "options": {}}
        )
        if solver == "scipy":
            d["options"]["use_umfpack"] = use_umfpack
            d["options"]["permc_spec"] = permc_spec
        elif solver == "pardiso":
            d["options"]["mtype"] = mtype
        return u, d
    else:
        return u


def linsolve_sparse(
    K_coo: coo,
    Kp_coo: coo,
    f: np.ndarray,
    *args,
    use_umfpack=True,
    sparsify=False,
    summary=True,
    **kwargs
):
    """
    Notes:
    ------
    If the load vector 'f' is dense, it must contain values for all
    nodes, even the passive ones.

    """
    # boxing
    N = f.shape[0]
    K, f, loc_to_glob = box_fem_data_sparse(K_coo, Kp_coo, f)
    # solution
    u, d = solve_standard_form(
        K, f, *args, sparsify=sparsify, use_umfpack=use_umfpack, summary=True, **kwargs
    )
    # unboxing
    u = unbox_lhs(u, loc_to_glob, N)

    if summary:
        d["regular"] = loc_to_glob is not None
        return u, d
    return u


def linsolve_bulk(
    K_bulk: np.ndarray, Kp_coo: coo, f: np.ndarray, gnum: np.ndarray, *args, **kwargs
):
    K_coo = fem_coeff_matrix_coo(K_bulk, inds=gnum, N=f.shape[0])
    return linsolve_sparse(K_coo, Kp_coo, f, *args, **kwargs)


def linsolve(K: arraylike, *args, gnum=None, **kwargs):
    if isspmatrix_np(K):
        return linsolve_sparse(K, *args, **kwargs)
    elif isinstance(K, np.ndarray) and gnum is not None:
        return linsolve_bulk(K, *args, gnum, **kwargs)


"""def linsolve_bulk_pop(K_bulk: np.ndarray, Kp_coo: coo, f: np.ndarray,
                      gnum: np.ndarray, fcores: np.ndarray, *args,
                      sparsify=True, use_umfpack=True,
                      summary=False, parallel=True,
                      max_workers=4, permc_spec='COLAMD', **kwargs):
    f = matrixform(f)
    N = f.shape[0]
    loc_to_glob, glob_to_loc = \
        index_mappers(gnum, N=N, return_inverse=True)
    nPop, nE = fcores.shape
    gnum = box_dof_numbering(gnum, glob_to_loc)
    M = gnum.max() + 1
    dtype = K_bulk.dtype
    shape = (M, M)

    # rhs, lhs
    u = np.zeros((nPop,) + f.shape, dtype=dtype)
    f = box_rhs(f.astype(dtype), loc_to_glob)
    Kp_coo = box_spmatrix(Kp_coo, glob_to_loc)

    # stiffness utils
    rows, cols = list(map(lambda x: x.flatten(), irows_icols_bulk(gnum)))

    def K_bulk_w(i): return weighted_stiffness_bulk(
        K_bulk, fcores[i]).flatten()
    def K_coo(i): return coo((K_bulk_w(i), (rows, cols)), shape=shape,
                             dtype=dtype)

    def K(i): return K_coo(i) + Kp_coo

    def solve_and_unbox(i):
        ui = solve_standard_form(K(i), f, *args, sparsify=sparsify,
                                 use_umfpack=use_umfpack, permc_spec=permc_spec,
                                 **kwargs)
        u[i, :, :] = unbox_lhs(ui, loc_to_glob, N)

    # serial and parallel solutions
    if not parallel:
        t0 = time()
        list(map(solve_and_unbox, range(nPop)))
        dt = time() - t0
    else:
        max_workers = min(max_workers, nPop)
        t0 = time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(solve_and_unbox, range(nPop))
        dt = time() - t0

    if summary:
        d = {
            'time [ms]': dt * 1000,
            'avg. time [ms]': dt * 1000 / nPop,
            'N': M,
            'use_umfpack': use_umfpack,
            'permc_spec': permc_spec,
            'sparsify': sparsify,
            'core': 'scipy.sparse.linalg.spsolve',
            'max workers': max_workers if parallel else 1,
            'population size': nPop
        }
        return u, d
    return u
"""
