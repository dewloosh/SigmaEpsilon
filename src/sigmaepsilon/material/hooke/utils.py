import numpy as np
from numpy import ndarray
from scipy.optimize import minimize

from dewloosh.core.tools.kwargtools import getasany
from neumann.function import Function

from .sym import smat_sym_ortho_3d


__all__ = ["compliance_matrix", "stiffness_matrix"]


poisson_ratios = ["NU12", "NU13", "NU23", "NU21", "NU32", "NU31"]
elastic_moduli = ["E1", "E2", "E3"]
shear_moduli = ["G12", "G13", "G23"]
standard_keys_ortho = elastic_moduli + poisson_ratios[:3] + shear_moduli
keys_ortho_all = standard_keys_ortho + poisson_ratios[3:]


class HookeError(Exception):
    ...


def _find_all_ortho_params(**params):
    S = smat_sym_ortho_3d()
    S_skew = S - S.T
    subs = [(sym, val) for sym, val in params.items() if val is not None]
    residual = S_skew.subs(subs)
    error = residual.norm()
    f = Function(error)
    v = f.variables
    x0 = np.zeros((len(v),), dtype=float)
    res = minimize(f, x0, method="Nelder-Mead", tol=1e-6)
    params.update({str(sym): val for sym, val in zip(v, res.x)})
    return params


def _group_params(**params) -> tuple:
    """
    Groups and returns all input material parameters as a 3-tuple
    of dictionaries.

    Returns
    -------
    tuple
        3 dictionaries for Young's moduli, Poisson's ratios
        and shear moduli.
    """
    PARAMS = {key.upper(): value for key, value in params.items()}
    keys = list(PARAMS.keys())
    E = {key: PARAMS[key] for key in keys if key[0] == "E"}
    NU = {key: PARAMS[key] for key in keys if key[:2] == "NU"}
    G = {key: PARAMS[key] for key in keys if key[0] == "G"}
    return E, NU, G


def _missing_std_params_ortho(**params):
    """
    Returns the ones missing from the 9 standard orthotropic parameters
    E1, E2, E3, NU12, NU13, NU23, G12, G13, and G23.
    """
    _params = [k for k, v in params.items() if v is not None]
    return [k for k in standard_keys_ortho if k not in _params]


def _missing_params_ortho(**params):
    """
    Returns the ones missing from the 12 orthotropic parameters
    E1, E2, E3, NU12, NU13, NU23, G12, G13, and G23.
    """
    _params = [k for k, v in params.items() if v is not None]
    return [k for k in keys_ortho_all if k not in _params]


def _has_std_params_ortho(**params) -> bool:
    """
    Returns True, if all 9 standard parameters E1, E2, E3, NU12, NU13, NU23,
    G12, G13, and G23 are provided.
    """
    return len(_missing_std_params_ortho(**params)) == 0


def _has_all_params_ortho(**params) -> bool:
    """
    Returns True, if all 12 keys of an orthotropic material are provided.
    """
    return len(_missing_params_ortho(**params)) == 0


def _get_iso_params(**kwargs):
    """
    Returns all 12 orthotropic engineering constants for an isotropic material.

    Requires 2 independent constants from E, NU and G to be provided.
    """
    KWARGS = {key.upper(): value for key, value in kwargs.items()}
    E = KWARGS.get("E", None)
    G = KWARGS.get("G", None)
    NU = KWARGS.get("NU", None)
    try:
        if E is None:
            E = 2 * G * (1 + NU)
        elif G is None:
            G = E / 2 / (1 + NU)
        elif NU is None:
            NU = E / 2 / G - 1
    except TypeError:
        msg = (
            "At least 2 independent constants",
            " must be defined for an isotropic material",
        )
        raise HookeError(msg)
    params = {
        "E1": E,
        "E2": E,
        "E3": E,
        "NU12": NU,
        "NU13": NU,
        "NU23": NU,
        "NU21": NU,
        "NU31": NU,
        "NU32": NU,
        "G12": G,
        "G13": G,
        "G23": G,
    }
    return params


def _finalize_ortho(**params):
    if _has_std_params_ortho(**params):
        params["NU21"] = params["NU12"] * params["E2"] / params["E1"]
        params["NU31"] = params["NU13"] * params["E3"] / params["E1"]
        params["NU32"] = params["NU23"] * params["E3"] / params["E2"]
        return params
    else:
        params = _find_all_ortho_params(**params)
        assert _has_all_params_ortho(**params), "Something went wrong!"


def _get_triso_params_23(**kwargs) -> dict:
    """
    Returns all 12 orthotropic engineering constants for a
    transversely isotropic material.

    From the total of 12 constants, exactly 5 must be provided,
    from which 1 must be the out-of-plane shear moduli. The remaining 4
    necessary constants can be provided in any combination.
    """
    KWARGS = {key.upper(): value for key, value in kwargs.items()}

    # 2 from the 3 shear moduli must be defined
    G23 = getasany(["G23", "G32"], None, **KWARGS)
    G13 = getasany(["G13", "G31", "G12", "G21"], None, **KWARGS)
    G12 = G13

    # three from this 5 must be defined
    NU12 = getasany(["NU12", "NU13"], None, **KWARGS)
    NU21 = getasany(["NU21", "NU31"], None, **KWARGS)
    NU23 = getasany(["NU23", "NU32"], None, **KWARGS)
    E2 = getasany(["E2", "E3"], None, **KWARGS)
    E1 = KWARGS.get("E1", None)

    # from symmetry
    NU32 = NU23
    NU13 = NU12
    NU31 = NU21
    E3 = E2

    params = {
        "E1": E1,
        "E2": E2,
        "E3": E3,
        "NU12": NU12,
        "NU13": NU13,
        "NU23": NU23,
        "NU21": NU21,
        "NU31": NU31,
        "NU32": NU32,
        "G12": G12,
        "G13": G13,
        "G23": G23,
    }
    return _finalize_ortho(**params)


def _get_ortho_params(**kwargs) -> dict:
    """
    Returns all 12 engineering constants for an orthotropic material.

    From the total of 12 constants, 9 must be provided, from which
    3 must be shear moduli. The remaining 6 necessary constants
    can be provided in any combination.
    """
    KWARGS = {key.upper(): value for key, value in kwargs.items()}
    E, NU, G = _group_params(**kwargs)
    nE, nNU, nG = len(E), len(NU), len(G)

    if not nG == 3:
        msg = "The three shear moduli must be provided!"
        raise HookeError(msg)

    if not (nE + nNU) >= 6:
        msg = (
            "6 from the 3 Young's moduli and the 6 Poisson's"
            " ratios must be provided!"
        )
        raise HookeError(msg)

    G12 = getasany(["G12", "G21"], None, **KWARGS)
    if G12 is None:
        msg = "Shear modulus G12 must be provided as either G12 or G21!"
        raise HookeError(msg)

    G23 = getasany(["G23", "G32"], None, **KWARGS)
    if G23 is None:
        msg = "Shear modulus G23 must be provided as either G23 or G32!"
        raise HookeError(msg)

    G13 = getasany(["G13", "G31"], None, **KWARGS)
    if G13 is None:
        msg = "Shear modulus G13 must be provided as either G13 or G31!"
        raise HookeError(msg)

    E1 = KWARGS.get("E1", None)
    E2 = KWARGS.get("E2", None)
    E3 = KWARGS.get("E3", None)
    NU12 = KWARGS.get("NU12", None)
    NU21 = KWARGS.get("NU21", None)
    NU23 = KWARGS.get("NU23", None)
    NU32 = KWARGS.get("NU32", None)
    NU13 = KWARGS.get("NU13", None)
    NU31 = KWARGS.get("NU31", None)

    params = {
        "E1": E1,
        "E2": E2,
        "E3": E3,
        "NU12": NU12,
        "NU13": NU13,
        "NU23": NU23,
        "NU21": NU21,
        "NU31": NU31,
        "NU32": NU32,
        "G12": G12,
        "G13": G13,
        "G23": G23,
    }
    return _finalize_ortho(**params)


def _get_elastic_params(**params):
    E, NU, G = _group_params(**params)
    nE, nNU, nG = len(E), len(NU), len(G)
    nParams = sum([nE, nNU, nG])

    if nParams == 2:
        params = _get_iso_params(**E, **NU, **G)
    elif nParams == 9:
        params = _get_ortho_params(**E, **NU, **G)
    else:
        raise HookeError("Invalid material definition!")

    return params


def compliance_matrix(imap: dict = None, **params) -> ndarray:
    params = _get_elastic_params(**params)
    sympy_matrix = smat_sym_ortho_3d(imap=imap, **params)
    return np.array(sympy_matrix.tolist()).astype(float)


def stiffness_matrix(*args, **kwargs) -> ndarray:
    return np.linalg.inv(compliance_matrix(*args, **kwargs))


if __name__ == "__main__":
    _get_iso_params(E=1, NU=0.2)
