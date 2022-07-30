# -*- coding: utf-8 -*-
from ....core.tools.kwargtools import getasany


standard_keys_ortho = ['E1', 'E2', 'E3',
                       'NU12', 'NU13', 'NU23', 'G12', 'G13', 'G23']
keys_ortho_all = standard_keys_ortho + ['NU21', 'NU32', 'NU31']
bulk_keys_ortho = ['E1', 'E2', 'E3', 'NU12',
                   'NU13', 'NU23', 'NU21', 'NU32', 'NU31']


class HookeError(Exception):
    ...


def group_mat_params(**params) -> tuple:
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
    E = {key: PARAMS[key] for key in keys if key[0] == 'E'}
    NU = {key: PARAMS[key] for key in keys if key[:2] == 'NU'}
    G = {key: PARAMS[key] for key in keys if key[0] == 'G'}
    return E, NU, G


def missing_std_params_ortho(**params):
    """Returns the ones missing from the 9 standard orthotropic parameters."""
    return [k for k in standard_keys_ortho if k not in params]


def missing_params_ortho(**params):
    """Returns the ones missing from the 12 orthotropic parameters."""
    return [k for k in keys_ortho_all if k not in params]


def has_std_params_ortho(**params) -> bool:
    """
    Returns True, if all 9 standard keys are provided.
    Standard keys are the 3 Young's moduli, the 3 shear moduli
    and the minor Poisson's ratios.
    """
    return len(missing_std_params_ortho(**params)) == 0


def has_all_params_ortho(**params) -> bool:
    """ Returns True, if all 12 keys of an orthotropic material are provided."""
    return len(missing_params_ortho(**params)) == 0


def get_iso_params(*args, **kwargs):
    """
    Returns all 12 orthotropic engineering constants for an 
    isotropic material. 
    Requires 2 independent constants to be provided.
    """
    KWARGS = {key.upper(): value for key, value in kwargs.items()}
    E = KWARGS.get('E', None)
    G = KWARGS.get('G', None)
    NU = KWARGS.get('NU', None)
    try:
        if E is None:
            E = 2*G*(1+NU)
        elif G is None:
            G = E/2/(1+NU)
        elif NU is None:
            NU = E/2/G - 1
    except TypeError:
        raise HookeError("At least 2 independent constants"
                         " must be defined for an isotropic material")
    params = \
        {'E1': E, 'E2': E, 'E3': E,
         'NU12': NU, 'NU13': NU, 'NU23': NU,
         'NU21': NU, 'NU31': NU, 'NU32': NU,
         'G12': G, 'G13': G, 'G23': G}
    return params


def get_triso_params(*args, **kwargs) -> dict:
    """
    Returns all 12 orthotropic engineering constants for a 
    transversely isotropic material. 
    From the total of 12 constants, exactly 5 must to be provided, 
    from which 1 must be the out-of-plane shear moduli. 
    The remaining 4 necessary constants can be provided in any combination.
    """
    KWARGS = {key.upper(): value for key, value in kwargs.items()}
    G12 = getasany(['G12', 'G21'], None, **KWARGS)
    G23 = getasany(['G23', 'G32'], None, **KWARGS)
    G13 = getasany(['G13', 'G23'], None, **KWARGS)
    E1 = KWARGS.get('E1', None)
    E2 = KWARGS.get('E2', None)
    E3 = KWARGS.get('E3', None)
    NU12 = KWARGS.get('NU12', None)
    NU21 = KWARGS.get('NU21', None)
    NU23 = KWARGS.get('NU23', None)
    NU32 = KWARGS.get('NU32', None)
    NU13 = KWARGS.get('NU13', None)
    NU31 = KWARGS.get('NU31', None)
    params = \
        {'E1': E1, 'E2': E2, 'E3': E3,
         'NU12': NU12, 'NU13': NU13, 'NU23': NU23,
         'NU21': NU21, 'NU31': NU31, 'NU32': NU32,
         'G12': G12, 'G13': G13, 'G23': G23}
    return params


def get_ortho_params(*args, **kwargs) -> dict:
    """
    Returns all 12 orthotropic engineering constants for an 
    orthotropic material. 
    From the total of 12 constants, 9 must to be provided, from which
    3 must be shear moduli. The remaining 6 necessary constants 
    can be provided in any combination.
    """
    KWARGS = {key.upper(): value for key, value in kwargs.items()}
    G12 = getasany(['G12', 'G21'], None, **KWARGS)
    assert G12 is not None, "Shear modulus G12 \
        must be provided as either G12 or G21!"
    G23 = getasany(['G23', 'G32'], None, **KWARGS)
    assert G23 is not None, "Shear modulus G23 \
        must be provided as either G23 or G32!"
    G13 = getasany(['G13', 'G23'], None, **KWARGS)
    assert G13 is not None, "Shear modulus G13 \
        must be provided as either G13 or G31!"
    E1 = KWARGS.get('E1', None)
    E2 = KWARGS.get('E2', None)
    E3 = KWARGS.get('E3', None)
    NU12 = KWARGS.get('NU12', None)
    NU21 = KWARGS.get('NU21', None)
    NU23 = KWARGS.get('NU23', None)
    NU32 = KWARGS.get('NU32', None)
    NU13 = KWARGS.get('NU13', None)
    NU31 = KWARGS.get('NU31', None)
    params = \
        {'E1': E1, 'E2': E2, 'E3': E3,
         'NU12': NU12, 'NU13': NU13, 'NU23': NU23,
         'NU21': NU21, 'NU31': NU31, 'NU32': NU32,
         'G12': G12, 'G13': G13, 'G23': G23}
    return params


if __name__ == '__main__':
    get_iso_params(E=1, NU=0.2)
