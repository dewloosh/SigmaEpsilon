from typing import Iterable

import numpy as np
import sympy as sy

from .imap import _imap_2d_to_1d, _imap_4d_to_2d


__all__ = ["smat_sym_ortho_3d", "cmat_sym_ortho_3d", "symbolic_elasticity_tensor"]


def symbolic_elasticity_tensor(
    *args, base: str = "C_", as_matrix: bool = False, imap: dict = None
) -> Iterable:
    """
    Returns a symbolic representation of a 4th order 3x3x3x3 tensor.
    If the argument 'as_matrix' is True, the function returns a 6x6 matrix,
    that unfolds according to the argument 'imap', or if it's not provided,
    the index map of the class the function is called on. If 'imap' is
    provided, it must be a dictionary including exactly 6 keys and
    values. The keys must be integers in the integer range (0, 6), the
    values must be tuples on the integer range (0, 3).
    The default mapping is the Voigt indicial map:
        0 : (0, 0)
        1 : (1, 1)
        2 : (2, 2)
        3 : (1, 2)
        4 : (0, 2)
        5 : (0, 1)
    """
    res = np.zeros((3, 3, 3, 3), dtype=object)
    indices = np.indices((3, 3, 3, 3))
    it = np.nditer([*indices], ["multi_index"])
    for _ in it:
        p, q, r, s = it.multi_index
        if q >= p and s >= r:
            sinds = np.array([p, q, r, s], dtype=np.int16) + 1
            sym = sy.symbols(base + "_".join(sinds.astype(str)))
            res[p, q, r, s] = sym
            res[q, p, r, s] = sym
            res[p, q, s, r] = sym
            res[q, p, s, r] = sym
            res[r, s, p, q] = sym
            res[r, s, q, p] = sym
            res[s, r, p, q] = sym
            res[s, r, q, p] = sym
    if as_matrix:
        mat = np.zeros((6, 6), dtype=object)
        imap = _imap_4d_to_2d if imap is None else imap
        for ij, ijkl in imap.items():
            mat[ij] = res[ijkl]
        if "sympy" in args:
            res = sy.Matrix(mat)
        else:
            res = mat
    return res


def _upper(ij):
    return (ij[1], ij[0]) if ij[0] >= ij[1] else (ij[0], ij[1])


def smat_sym_ortho_3d(imap: dict = None, **subs) -> sy.Matrix:
    """
    Reurns the compliance matrix for a 3d orthotropic
    material in symbolic form.

    Parameters
    ----------
    imap: dict, Optional
        Index map. Default is None.
    subs: dict, Optional
        A dictionary of parameters to substitute.
    simplify: bool, Optional
        Whether to simplify the resulting symbolic matrix. Default is True.

    Notes
    -----
    The resulting symbolic matrix has 12 free symbols,
    but only 9 of these are independent.

    Returns
    -------
    sympy.Matrix
        6x6 symbolic compliance matrix.
    """
    E1, E2, E3 = sy.symbols("E1 E2 E3")
    G23, G12, G13 = sy.symbols("G23 G12 G13")
    NU12, NU21 = sy.symbols("NU12 NU21")
    NU13, NU31 = sy.symbols("NU13 NU31")
    NU23, NU32 = sy.symbols("NU23 NU32")
    S = sy.Matrix(
        [
            [1 / E1, -NU12 / E1, -NU13 / E1, 0, 0, 0],
            [-NU21 / E2, 1 / E2, -NU23 / E2, 0, 0, 0],
            [-NU31 / E3, -NU32 / E3, 1 / E3, 0, 0, 0],
            [0, 0, 0, 1 / G23, 0, 0],
            [0, 0, 0, 0, 1 / G13, 0],
            [0, 0, 0, 0, 0, 1 / G12],
        ]
    )
    if len(subs) > 0:
        S = S.subs([(sym, val) for sym, val in subs.items()])
    if imap is None:
        return S
    assert len(imap) == 6, "Indexmap must contain 6 indices."
    invmap = {_upper(v): k for k, v in imap.items()}
    trmap = {i: invmap[_imap_2d_to_1d[i]] for i in range(6)}
    S_new = sy.zeros(6, 6)
    for inds in trmap.items():
        S_new[inds[1], :] = _swap1d(S[inds[0], :], inds)
    return S_new


def cmat_sym_ortho_3d(imap: dict = None, **subs) -> sy.Matrix:
    """
    Reurns the stiffness matrix for a 3d orthotropic
    material in symbolic form.

    Parameters
    ----------
    imap : dict, optional
        Index map. Default is None.
    subs : dict, optional
        A dictionary of parameters to substitute.
    simplify : bool, Optional
        Whether to simplify the resulting symbolic matrix. Default is True.

    Notes
    -----
    The resulting symbolic matrix has 12 free symbols,
    but only 9 of these are independent.

    Returns
    -------
    sympy.Matrix
        6x6 symbolic stiffness matrix
    """
    res = smat_sym_ortho_3d(imap=imap, simplify=False, **subs).inv()
    res.simplify()
    return res


def _swap1d(a: sy.Matrix, inds: tuple):
    a.col_swap(*inds)
    return a
