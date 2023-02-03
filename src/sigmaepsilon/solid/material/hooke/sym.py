import sympy as sy


__all__ = ["smat_sym_ortho_3d", "cmat_sym_ortho_3d"]


imap_ref_ortho_3D = {0: (0, 0), 1: (1, 1), 2: (2, 2), 3: (1, 2), 4: (0, 2), 5: (0, 1)}


def upper(ij):
    return (ij[1], ij[0]) if ij[0] >= ij[1] else (ij[0], ij[1])


def smat_sym_ortho_3d(imap: dict = None, **subs) -> sy.Matrix:
    """
    Reurns the compliance matrix for a 3d orthotropic
    material in symbolic form.

    Parameters
    ----------
    imap : dict, Optional
        Index map. Default is None.
    subs : dict, Optional
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
    invmap = {upper(v): k for k, v in imap.items()}
    trmap = {i: invmap[imap_ref_ortho_3D[i]] for i in range(6)}
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


if __name__ == "__main__":
    C = cmat_sym_ortho_3d()
