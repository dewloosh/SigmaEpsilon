from typing import Tuple

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize

from dewloosh.core.tools.kwargtools import getasany
from neumann.function import Function
from neumann.linalg import is_pos_def

from .sym import smat_sym_ortho_3d

__all__ = ["elastic_compliance_matrix", "elastic_stiffness_matrix"]


poisson_ratios = ["NU12", "NU13", "NU23", "NU21", "NU32", "NU31"]
elastic_moduli = ["E1", "E2", "E3"]
shear_moduli = ["G12", "G13", "G23"]
standard_keys_ortho = elastic_moduli + poisson_ratios[:3] + shear_moduli
keys_ortho_all = standard_keys_ortho + poisson_ratios[3:]


class HookeError(Exception):
    ...


def _find_all_ortho_params(**params):

    # select 9 parameters (3 shear moduli and 6 from the others)
    _params = dict()
    Young, Poisson, Shear = _group_params(**params)
    _params.update(Shear)
    Young.update(Poisson)
    for i, (k, v) in enumerate(Young.items()):
        if i < 6:
            _params[k] = v

    S = smat_sym_ortho_3d()
    S_skew = S - S.T
    subs = [(sym, val) for sym, val in _params.items() if val is not None]
    residual = S_skew.subs(subs)
    error = residual.norm()
    f = Function(error)
    v = f.variables
    x0 = np.zeros((len(v),), dtype=float)
    res = minimize(f, x0, method="Nelder-Mead", tol=1e-6)
    _params.update({str(sym): val for sym, val in zip(v, res.x)})
    return _params


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


def _finalize_iso(E: float = None, NU: float = None, G: float = None) -> Tuple[float]:
    try:
        if E is None:
            E = 2 * G * (1 + NU)
        elif G is None:
            G = E / 2 / (1 + NU)
        elif NU is None:
            NU = E / 2 / G - 1
        return E, NU, G
    except TypeError:
        msg = (
            "At least 2 independent constants",
            " must be defined for an isotropic material",
        )
        raise HookeError(msg)


def _finalize_ortho(**params):
    if _has_std_params_ortho(**params):
        params["NU21"] = params["NU12"] * params["E2"] / params["E1"]
        params["NU31"] = params["NU13"] * params["E3"] / params["E1"]
        params["NU32"] = params["NU23"] * params["E3"] / params["E2"]
        return params
    else:
        E, NU, G = _group_params(**params)
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

        params = _find_all_ortho_params(**params)
        assert _has_all_params_ortho(**params), "Something went wrong!"


def _get_iso_params(**kwargs):
    """
    Returns all 12 orthotropic engineering constants for an isotropic material.

    Requires 2 independent constants from E, NU and G to be provided.
    """
    KWARGS = {key.upper(): value for key, value in kwargs.items()}
    E = KWARGS.get("E", None)
    G = KWARGS.get("G", None)
    NU = KWARGS.get("NU", None)
    E, NU, G = _finalize_iso(E, NU, G)
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


def _get_triso_params_23(**kwargs) -> dict:
    """
    Returns all 12 orthotropic engineering constants for a
    transversely isotropic material.

    From the total of 12 constants, exactly 5 must be provided,
    from which 1 must be the out-of-plane shear modulus. The remaining 4
    necessary constants can be provided in any possible combination.
    """
    KWARGS = {key.upper(): value for key, value in kwargs.items()}

    G13 = getasany(["G13", "G31", "G12", "G21"], None, **KWARGS)
    assert G13 is not None, "The out-of-plane shear modulus must be provided!"

    NU12 = getasany(["NU12", "NU13"], None, **KWARGS)
    NU21 = getasany(["NU21", "NU31"], None, **KWARGS)
    NU23 = getasany(["NU23", "NU32"], None, **KWARGS)
    E2 = getasany(["E2", "E3"], None, **KWARGS)
    E1 = KWARGS.get("E1", None)
    G23 = getasany(["G23", "G32"], None, **KWARGS)

    E2, NU23, G23 = _finalize_iso(E2, NU23, G23)

    # from material symmetry
    NU32 = NU23
    NU13 = NU12
    NU31 = NU21
    E3 = E2
    G12 = G13

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


def _get_triso_params_13(**kwargs) -> dict:
    """
    Returns all 12 orthotropic engineering constants for a
    transversely isotropic material.

    From the total of 12 constants, exactly 5 must be provided,
    from which 1 must be the out-of-plane shear modulus. The remaining 4
    necessary constants can be provided in any possible combination.
    """
    KWARGS = {key.upper(): value for key, value in kwargs.items()}

    G12 = getasany(["G12", "G21", "G23", "G32"], None, **KWARGS)
    assert G12 is not None, "The out-of-plane shear modulus must be provided!"

    NU12 = getasany(["NU12", "NU32"], None, **KWARGS)
    NU21 = getasany(["NU21", "NU23"], None, **KWARGS)
    NU13 = getasany(["NU13", "NU31"], None, **KWARGS)
    E1 = getasany(["E1", "E3"], None, **KWARGS)
    E2 = KWARGS.get("E2", None)
    G13 = getasany(["G13", "G31"], None, **KWARGS)

    E1, NU13, G13 = _finalize_iso(E1, NU13, G13)

    # from material symmetry
    NU31 = NU13
    NU32 = NU12
    NU23 = NU21
    E3 = E1
    G23 = G12

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


def _get_triso_params_12(**kwargs) -> dict:
    """
    Returns all 12 orthotropic engineering constants for a
    transversely isotropic material.

    From the total of 12 constants, exactly 5 must be provided,
    from which 1 must be the out-of-plane shear modulus. The remaining 4
    necessary constants can be provided in any possible combination.
    """
    KWARGS = {key.upper(): value for key, value in kwargs.items()}

    G13 = getasany(["G13", "G31", "G23", "G32"], None, **KWARGS)
    assert G13 is not None, "The out-of-plane shear modulus must be provided!"

    NU13 = getasany(["NU13", "NU23"], None, **KWARGS)
    NU31 = getasany(["NU31", "NU32"], None, **KWARGS)
    NU12 = getasany(["NU12", "NU21"], None, **KWARGS)
    E1 = getasany(["E1", "E2"], None, **KWARGS)
    E3 = KWARGS.get("E3", None)
    G12 = getasany(["G12", "G21"], None, **KWARGS)

    E1, NU12, G12 = _finalize_iso(E1, NU12, G12)

    # from material symmetry
    NU21 = NU12
    NU23 = NU13
    NU32 = NU31
    E2 = E1
    G23 = G13

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


def _get_elastic_params(isoplane: str = None, **params):
    E, NU, G = _group_params(**params)
    nE, nNU, nG = len(E), len(NU), len(G)
    nParams = sum([nE, nNU, nG])

    if nParams == 2:
        params = _get_iso_params(**E, **NU, **G)
    elif nParams == 5:
        if not isinstance(isoplane, str):
            raise ValueError(
                "The plane of isotropy must be specified. See the docs!")
        if isoplane in ["23", "32"]:
            params = _get_triso_params_23(**E, **NU, **G)
        elif isoplane in ["12", "21"]:
            params = _get_triso_params_12(**E, **NU, **G)
        elif isoplane in ["13", "31"]:
            params = _get_triso_params_13(**E, **NU, **G)
    elif nParams == 9:
        params = _get_ortho_params(**E, **NU, **G)
    else:
        raise HookeError("Invalid material definition!")

    return params


def elastic_compliance_matrix(
    imap: dict = None, verify: bool = False, **params
) -> ndarray:
    """
    Returns the elastic compliance matrix for a Hooke-type material model from
    a set of engineering constants.

    Parameters
    ----------
    imap: dict, Optional
        Index mapping. Default is the Voigt-map.
    verify: bool, Optional
        If True, the resulting matrix is verified before getting returned. Default is False.
    isoplane: str, Optional
        If the number of specified engineering constants is 5, it suggests a transversely
        orthotropic material. In this case, the parameter 'isoplane' defines the plane of
        isotropy. Valid choices are '12', '23' and '13'. Default is None.
        
    Raises
    ------
    ~sigmaepsilon.material.hooke.utils.HookeError

    Returns
    -------
    numpy.ndarray
        A 6x6 NumPy float array.
    """
    params = _get_elastic_params(**params)
    sympy_matrix = smat_sym_ortho_3d(imap=imap, **params)
    S = np.array(sympy_matrix.tolist()).astype(float)
    if verify:
        if not is_pos_def(S):
            raise HookeError(
                "The parameters does not result in a positive definite material!")
    return S


def elastic_stiffness_matrix(*args, verify: bool = False, **kwargs) -> ndarray:
    """
    Returns the elastic stiffness matrix for a Hooke-type material model from
    a set of engineering constants.

    Parameters
    ----------
    imap: dict, Optional
        Index mapping. Default is the Voigt-map.
    verify: bool, Optional
        If True, the resulting matrix is verified before getting returned. Default is False.
    isoplane: str, Optional
        If the number of specified engineering constants is 5, it suggests a transversely
        orthotropic material. In this case, the parameter 'isoplane' defines the plane of
        isotropy. Valid choices are '12', '23' and '13'. Default is None.
        
    Raises
    ------
    ~sigmaepsilon.material.hooke.utils.HookeError

    Returns
    -------
    numpy.ndarray
        A 6x6 NumPy float array.
    """
    C = np.linalg.inv(elastic_compliance_matrix(*args, **kwargs))
    if verify:
        if not is_pos_def(C):
            raise HookeError(
                "The parameters does not result in a positive definite material!")
    return C
