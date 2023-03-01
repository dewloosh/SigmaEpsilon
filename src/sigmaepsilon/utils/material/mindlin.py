import numpy as np
from numpy.linalg import inv
from numba import njit, prange

from .surface import points_of_layers

__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def rotation_matrices(angles: np.ndarray):
    """
    Returns transformation matrices T_126 and T_45 for each angle.
    Angles are expected in radians.
    """
    nL = len(angles)
    T_126 = np.zeros((nL, 3, 3), dtype=angles.dtype)
    T_45 = np.zeros((nL, 2, 2), dtype=angles.dtype)
    for iL in prange(nL):
        a = angles[iL] * np.pi / 180
        T_126[iL, 0, 0] = np.cos(a) ** 2
        T_126[iL, 0, 1] = np.sin(a) ** 2
        T_126[iL, 0, 2] = -np.sin(2 * a)
        T_126[iL, 1, 0] = T_126[iL, 0, 1]
        T_126[iL, 1, 1] = T_126[iL, 0, 0]
        T_126[iL, 1, 2] = -T_126[iL, 0, 2]
        T_126[iL, 2, 0] = np.cos(a) * np.sin(a)
        T_126[iL, 2, 1] = -T_126[iL, 2, 0]
        T_126[iL, 2, 2] = np.cos(a) ** 2 - np.sin(a) ** 2
        T_45[iL, 0, 0] = np.cos(a)
        T_45[iL, 0, 1] = -np.sin(a)
        T_45[iL, 1, 1] = T_45[iL, 0, 0]
        T_45[iL, 1, 0] = -T_45[iL, 0, 1]
    return T_126, T_45


@njit(nogil=True, parallel=True, cache=__cache)
def material_stiffness_matrices(
    C_126: np.ndarray, C_45: np.ndarray, angles: np.ndarray
):
    """
    Returns the components of the material stiffness matrices C_126 and C_45
    in the global system.
    """
    nL = len(C_126)
    C_126_g = np.zeros_like(C_126)
    C_45_g = np.zeros_like(C_45)
    R_126 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=C_126_g.dtype)
    R_126_inv = np.linalg.inv(R_126)
    R_45 = np.array([[2, 0], [0, 2]], dtype=C_126_g.dtype)
    R_45_inv = np.linalg.inv(R_45)
    T_126, T_45 = rotation_matrices(angles)
    for iL in prange(nL):
        C_126_g[iL] = T_126[iL] @ C_126[iL] @ R_126_inv @ T_126[iL].T @ R_126
        C_45_g[iL] = T_45[iL] @ C_45[iL] @ R_45_inv @ T_45[iL].T @ R_45
    return C_126_g, C_45_g


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def shear_factors_MT(ABDS: np.ndarray, C_126: np.ndarray, z: np.ndarray):
    """
    # FIXME Should work with parallel = True, does not. Reason is
    propably a race condition due to using explicit parallel loops.
    """
    A11 = ABDS[0, 0]
    B11 = ABDS[0, 3]
    D11 = ABDS[3, 3]
    A22 = ABDS[1, 1]
    B22 = ABDS[1, 4]
    D22 = ABDS[4, 4]
    nL = C_126.shape[0]  # number of layers

    # calculate shear factors
    eta_x = 1 / (A11 * D11 - B11**2)
    eta_y = 1 / (A22 * D22 - B22**2)
    shear_factors = np.zeros((nL, 2, 3), dtype=ABDS.dtype)

    for iL in prange(nL - 1):
        for iP in prange(2):
            zi0 = z[iL, iP]
            zi1 = z[iL, iP + 1]
            dsfx = (
                -eta_x
                * C_126[iL, 0, 0]
                * (0.5 * (zi1**2 - zi0**2) * A11 - (zi1 - zi0) * B11)
            )
            shear_factors[iL, 0, iP + 1 :] += dsfx
            shear_factors[iL + 1 :, 0, :] += dsfx
            dsfy = (
                -eta_y
                * C_126[iL, 1, 1]
                * (0.5 * (zi1**2 - zi0**2) * A22 - (zi1 - zi0) * B22)
            )
            # these slicings probably cause a race condition
            shear_factors[iL, 1, iP + 1 :] += dsfy
            shear_factors[iL + 1 :, 1, :] += dsfy
    # last layer
    iL = nL - 1
    for iP in prange(2):
        zi0 = z[iL, iP]
        zi1 = z[iL, iP + 1]
        dsfx = (
            -eta_x
            * C_126[iL, 0, 0]
            * (0.5 * (zi1**2 - zi0**2) * A11 - (zi1 - zi0) * B11)
        )
        shear_factors[iL, 0, iP + 1 :] += dsfx
        dsfy = (
            -eta_y
            * C_126[iL, 1, 1]
            * (0.5 * (zi1**2 - zi0**2) * A22 - (zi1 - zi0) * B22)
        )
        shear_factors[iL, 1, iP + 1 :] += dsfy
    shear_factors[iL, :, 2] = 0.0
    return shear_factors


@njit(nogil=True, cache=__cache)
def shear_factors_ST(ABDS: np.ndarray, C_126: np.ndarray, z: np.ndarray):
    """
    Single-thread implementation of calculation of shear factors for
    multi-layer Mindlin shells.
    """
    A11 = ABDS[0, 0]
    B11 = ABDS[0, 3]
    D11 = ABDS[3, 3]
    A22 = ABDS[1, 1]
    B22 = ABDS[1, 4]
    D22 = ABDS[4, 4]
    nL = z.shape[0]  # number of layers

    # calculate shear factors
    eta_x = 1 / (A11 * D11 - B11**2)
    eta_y = 1 / (A22 * D22 - B22**2)
    shear_factors = np.zeros((nL, 2, 3), dtype=ABDS.dtype)

    for iL in range(nL):
        zi = z[iL]
        # first point through the thickness
        shear_factors[iL, 0, 0] = shear_factors[iL - 1, 0, 2]
        shear_factors[iL, 1, 0] = shear_factors[iL - 1, 1, 2]
        # second point through the thickness
        shear_factors[iL, 0, 1] = shear_factors[iL, 0, 0] - eta_x * C_126[iL, 0, 0] * (
            0.5 * (zi[1] ** 2 - zi[0] ** 2) * A11 - (zi[1] - zi[0]) * B11
        )
        shear_factors[iL, 1, 1] = shear_factors[iL, 1, 0] - eta_y * C_126[iL, 1, 1] * (
            0.5 * (zi[1] ** 2 - zi[0] ** 2) * A22 - (zi[1] - zi[0]) * B22
        )
        # third point through the thickness
        shear_factors[iL, 0, 2] = shear_factors[iL, 0, 0] - eta_x * C_126[iL, 0, 0] * (
            0.5 * (zi[2] ** 2 - zi[0] ** 2) * A11 - (zi[2] - zi[0]) * B11
        )
        shear_factors[iL, 1, 2] = shear_factors[iL, 1, 0] - eta_y * C_126[iL, 1, 1] * (
            0.5 * (zi[2] ** 2 - zi[0] ** 2) * A22 - (zi[2] - zi[0]) * B22
        )
    shear_factors[nL - 1, :, 2] = 0.0
    return shear_factors


@njit(nogil=True, parallel=True, cache=__cache)
def shear_correction_data(
    ABDS: np.ndarray, C_126: np.ndarray, C_45: np.ndarray, bounds: np.ndarray
):
    """
    FIXME : Results are OK, but a bit slower than expected when measured
    against the pure python implementation.
    """

    nL = bounds.shape[0]  # number of layers

    # z coordinate of 3 points per each layer
    z = points_of_layers(bounds, 3)

    # calculate shear factors
    shear_factors = shear_factors_ST(ABDS, C_126, z)

    # compile shear factors
    sf = np.zeros((nL, 2, 3), dtype=ABDS.dtype)
    for iL in prange(nL):
        monoms_inv = inv(np.array([[1, z, z**2] for z in z[iL]], dtype=ABDS.dtype))
        sf[iL, 0] = monoms_inv @ shear_factors[iL, 0]
        sf[iL, 1] = monoms_inv @ shear_factors[iL, 1]

    # potential energy using constant stress distribution
    # and unit shear force
    pot_c_x = 0.5 / ABDS[6, 6]
    pot_c_y = 0.5 / ABDS[7, 7]

    # positions and weights of Gauss-points
    gP = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)], dtype=ABDS.dtype)
    gW = np.array([5 / 9, 8 / 9, 5 / 9], dtype=ABDS.dtype)

    # potential energy using parabolic stress distribution
    # and unit shear force
    pot_p_x, pot_p_y = 0.0, 0.0
    for iL in prange(nL):
        dJ = 0.5 * (bounds[iL, 1] - bounds[iL, 0])
        Gxi = C_45[iL, 0, 0]
        Gyi = C_45[iL, 1, 1]
        for iG in prange(3):
            ziG = 0.5 * (
                (bounds[iL, 1] + bounds[iL, 0])
                + gP[iG] * (bounds[iL, 1] - bounds[iL, 0])
            )
            monoms = np.array([1, ziG, ziG**2], dtype=ABDS.dtype)
            sfx = np.dot(monoms, sf[iL, 0])
            sfy = np.dot(monoms, sf[iL, 1])
            pot_p_x += 0.5 * (sfx**2) * dJ * gW[iG] / Gxi
            pot_p_y += 0.5 * (sfy**2) * dJ * gW[iG] / Gyi
    kx = pot_c_x / pot_p_x
    ky = pot_c_y / pot_p_y

    return np.array([[kx, 0], [0, ky]]), shear_factors


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def stiffness_data(
    C_126: np.ndarray, C_45: np.ndarray, angles: np.ndarray, bounds: np.ndarray
):
    """
    FIXME Call
        Ks, sf = shear_correction_data(ABDS, C_126_g, C_45_g, bounds)
    is a bit slow for some reason.
    """
    ABDS = np.zeros((8, 8), dtype=C_126.dtype)
    nL = C_126.shape[0]
    C_126_g, C_45_g = material_stiffness_matrices(C_126, C_45, angles)
    for iL in prange(nL):
        ABDS[0:3, 0:3] += C_126_g[iL] * (bounds[iL, 1] - bounds[iL, 0])
        ABDS[0:3, 3:6] += (
            (1 / 2) * C_126_g[iL] * (bounds[iL, 1] ** 2 - bounds[iL, 0] ** 2)
        )
        ABDS[3:6, 3:6] += (
            (1 / 3) * C_126_g[iL] * (bounds[iL, 1] ** 3 - bounds[iL, 0] ** 3)
        )
        ABDS[6:8, 6:8] += C_45_g[iL] * (bounds[iL, 1] - bounds[iL, 0])
    ABDS[3:6, 0:3] = ABDS[0:3, 3:6]
    Ks, sf = shear_correction_data(ABDS, C_126_g, C_45_g, bounds)
    return ABDS, Ks, sf


@njit(nogil=True, cache=__cache)
def z_to_shear_factors(z, sfx, sfy):
    monoms = np.array([1, z, z**2], dtype=sfx.dtype)
    return np.dot(monoms, sfx), np.dot(monoms, sfy)
