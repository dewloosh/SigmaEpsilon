from typing import Tuple, Union, Callable, Iterable

import numpy as np
from numpy import ndarray
from numpy.linalg import inv
import sympy as sy

from ...utils.material.hooke import _get_elastic_params
from ...utils.material.mindlin import (
    _get_shell_material_stiffness_matrix,
    shell_rotation_matrix,
)
from ...utils.material.symbolic import smat_sym_ortho_3d
from .meta import Surface, SurfaceLayer


__all__ = ["MindlinShell", "MindlinPlate", "MindlinShellLayer", "MindlinPlateLayer"]


class MindlinShellLayer(SurfaceLayer):
    """
    Helper object for the stress analysis of a layer of a shell.
    """

    __loc__ = [-1.0, 0.0, 1.0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # locations of discrete points along the thickness
        self.zi = [self.loc_to_z(loc) for loc in MindlinShellLayer.__loc__]

        # Shear factors have to be multiplied by shear force to obtain shear
        # stress. They are determined externaly at discrete points, which
        # are later used for interpolation.
        self.shear_factors_x = np.array([0.0, 0.0, 0.0])
        self.shear_factors_y = np.array([0.0, 0.0, 0.0])

        # polinomial coefficients for shear factor interpoaltion
        self.sfx = None
        self.sfy = None

    def material_elastic_stiffness_matrix(self) -> Tuple[ndarray, ndarray]:
        """
        Returns the transformed material stiffness matrix.
        """
        Cm = _get_shell_material_stiffness_matrix(self.hooke)
        T = self.rotation_matrix()
        R = np.diag([1.0, 1.0, 2.0, 2.0, 2.0])
        self._hooke = inv(T) @ Cm @ R @ T @ inv(R)
        # self._hooke = T @ Cm @ inv(R) @ T.T @ R
        return self._hooke

    def rotation_matrix(self) -> Tuple[ndarray, ndarray]:
        """
        Returns the rotation matrix.
        """
        return shell_rotation_matrix(self.angle, rad=False)

    def elastic_stiffness_matrix(self) -> ndarray:
        """
        Returns the uncorrected stiffness contribution to the layer.
        """
        C = self.material_elastic_stiffness_matrix()
        tmin = self.tmin
        tmax = self.tmax
        A = C[:3, :3] * (tmax - tmin)
        B = (1 / 2) * C[:3, :3] * (tmax**2 - tmin**2)
        D = (1 / 3) * C[:3, :3] * (tmax**3 - tmin**3)
        S = C[3:, 3:] * (tmax - tmin)
        ABDS = np.zeros([8, 8], dtype=float)
        ABDS[0:3, 0:3] = A
        ABDS[0:3, 3:6] = B
        ABDS[3:6, 0:3] = B
        ABDS[3:6, 3:6] = D
        ABDS[6:8, 6:8] = S
        return ABDS

    def compile_shear_factors(self):
        """
        Prepares data for continuous interpolation of shear factors. Should
        be called if shear factors are already set.
        """
        coeff_inv = inv(np.array([[1, z, z**2] for z in self.zi]))
        self.sfx = np.matmul(coeff_inv, self.shear_factors_x)
        self.sfy = np.matmul(coeff_inv, self.shear_factors_y)

    def loc_to_shear_factors(self, loc: float) -> Tuple[float, float]:
        """
        Returns shear factor for local z direction by quadratic interpolation.
        Local coordinate is expected between -1 and 1.
        """
        z = self.loc_to_z(loc)
        monoms = np.array([1, z, z**2])
        return np.dot(monoms, self.sfx), np.dot(monoms, self.sfy)

    def approxfunc(self, data) -> Callable:
        z0, z1, z2 = self.zi
        z = np.array([[1, z0, z0**2], [1, z1, z1**2], [1, z2, z2**2]])
        a, b, c = inv(z) @ np.array(data)
        return lambda z: a + b * z + c * z**2


class MindlinPlateLayer(MindlinShellLayer):
    """
    Helper object for the stress analysis of a layer of a plate.
    """

    def elastic_stiffness_matrix(self) -> ndarray:
        """
        Returns the uncorrected stiffness contribution to the layer.
        """
        C = self.material_elastic_stiffness_matrix()
        tmin = self.tmin
        tmax = self.tmax
        D = (1 / 3) * C[:3, :3] * (tmax**3 - tmin**3)
        S = C[3:, 3:] * (tmax - tmin)
        ABDS = np.zeros([5, 5])
        ABDS[:3, :3] = D
        ABDS[3:, 3:] = S
        return ABDS


class MindlinShell(Surface):
    """
    Helper object for the stress analysis of a shell.

    Example
    -------
    >>> from sigmaepsilon.material import MindlinShell
    >>> model = {
    >>>     '0' : {
    >>>         'hooke' : MindlinShell.Hooke(E=2100000, NU=0.3),
    >>>         'angle' : 0.,
    >>>         'thickness' : 0.1
    >>>         },
    >>>     }
    >>> C = MindlinShell.from_dict(model).elastic_stiffness_matrix()
    """

    __layerclass__ = MindlinShellLayer

    @classmethod
    def Hooke(cls, symbolic: bool = False, **kwargs) -> Union[sy.Matrix, np.ndarray]:
        """
        Returns a Hooke matrix appropriate for shells.

        Parameters
        ----------
        symbolic: bool, Optional
            If True, a symbolic matrix is returned.

        Returns
        -------
        Union[sympy.Matrix, numpy.ndarray]
            The matrix in symbolic or numeric form.

        Examples
        --------
        >>> from sigmaepsilon.material import MindlinShell
        >>> MindlinShell.Hooke(E=2100000, NU=0.3)
        >>> MindlinShell.Hooke(symbolic=True)
        """
        S = smat_sym_ortho_3d()
        S.row_del(2)
        S.col_del(2)

        if symbolic:
            return S.inv()

        constants = _get_elastic_params(**kwargs)
        subs = {s: constants[str(s)] for s in S.free_symbols}
        S = S.subs([(sym, val) for sym, val in subs.items()])
        S = np.array(S, dtype=float)
        return inv(S)

    def elastic_stiffness_matrix(self) -> ndarray:
        """
        Returns the stiffness matrix of the shell.

        Returns
        -------
        numpy.ndarray
            The ABDS matrix of the shell.
        """
        ABDS = super().elastic_stiffness_matrix()
        layers: Iterable[MindlinShellLayer] = self.layers()
        A11 = ABDS[0, 0]
        B11 = ABDS[0, 3]
        D11 = ABDS[3, 3]
        S55 = ABDS[6, 6]
        A22 = ABDS[1, 1]
        B22 = ABDS[1, 4]
        D22 = ABDS[4, 4]
        S44 = ABDS[7, 7]

        ABDS_inv = np.linalg.inv(ABDS[:6, :6])
        alpha_x = ABDS_inv[0, 3]
        beta_x = ABDS_inv[3, 3]
        alpha_y = ABDS_inv[1, 4]
        beta_y = ABDS_inv[4, 4]

        eta_x = 1 / (A11 * D11 - B11**2)
        eta_y = 1 / (A22 * D22 - B22**2)
        alpha_x = -B11 * eta_x
        beta_x = A11 * eta_x
        alpha_y = -B22 * eta_y
        beta_y = A22 * eta_y

        # Create shear factors. These need to be multiplied with the shear
        # force in order to obtain shear stress at a given height. Since the
        # shear stress distribution is of 2nd order, the factors are
        # determined at 3 locations per layer.
        for i, layer in enumerate(layers):
            zi = layer.zi
            Exi = layer._hooke[0, 0]
            Eyi = layer._hooke[1, 1]

            # first point through the thickness
            layer.shear_factors_x[0] = layers[i - 1].shear_factors_x[-1]
            layer.shear_factors_y[0] = layers[i - 1].shear_factors_y[-1]

            # second point through the thickness
            layer.shear_factors_x[1] = layer.shear_factors_x[0] - Exi * (
                0.5 * (zi[1] ** 2 - zi[0] ** 2) * beta_x + (zi[1] - zi[0]) * alpha_x
            )
            layer.shear_factors_y[1] = layer.shear_factors_y[0] - Eyi * (
                0.5 * (zi[1] ** 2 - zi[0] ** 2) * beta_y + (zi[1] - zi[0]) * alpha_y
            )

            # third point through the thickness
            layer.shear_factors_x[2] = layer.shear_factors_x[0] - Exi * (
                0.5 * (zi[2] ** 2 - zi[0] ** 2) * beta_x + (zi[2] - zi[0]) * alpha_x
            )
            layer.shear_factors_y[2] = layer.shear_factors_y[0] - Eyi * (
                0.5 * (zi[2] ** 2 - zi[0] ** 2) * beta_y + (zi[2] - zi[0]) * alpha_y
            )

        # remove numerical junk from the end
        layers[-1].shear_factors_x[-1] = 0.0
        layers[-1].shear_factors_y[-1] = 0.0

        # prepare data for interpolation of shear stresses in a layer
        for layer in layers:
            layer.compile_shear_factors()

        # potential energy using constant stress distribution
        # and unit shear force
        pot_c_x = 0.5 / S55
        pot_c_y = 0.5 / S44

        # positions and weights of Gauss-points
        gP = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])
        gW = np.array([5 / 9, 8 / 9, 5 / 9])

        # potential energy using parabolic stress distribution
        # and unit shear force
        pot_p_x, pot_p_y = 0.0, 0.0
        for layer in layers:
            dJ = 0.5 * (layer.tmax - layer.tmin)
            Gxi = layer._hooke[-2, -2]
            Gyi = layer._hooke[-1, -1]
            for loc, weight in zip(gP, gW):
                sfx, sfy = layer.loc_to_shear_factors(loc)
                pot_p_x += 0.5 * (sfx**2) * dJ * weight / Gxi
                pot_p_y += 0.5 * (sfy**2) * dJ * weight / Gyi
        kx = pot_c_x / pot_p_x
        ky = pot_c_y / pot_p_y

        ABDS[6, 6] = kx * S55
        ABDS[7, 7] = ky * S44
        self.kx = kx
        self.ky = ky
        self.ABDS = ABDS
        self.SDBA = inv(ABDS)
        return ABDS


class MindlinPlate(MindlinShell):
    """
    Helper object for the stress analysis of a membrane.

    Example
    -------
    >>> from sigmaepsilon.model import MindlinPlate
    >>> model = {
    >>>     '0' : {
    >>>         'hooke' : MindlinPlate.Hooke(E=2100000, NU=0.3),
    >>>         'angle' : 0.,
    >>>         'thickness' : 0.1
    >>>         },
    >>>     }
    >>> C = MindlinPlate.from_dict(model).elastic_stiffness_matrix()
    """

    def elastic_stiffness_matrix(self) -> ndarray:
        """
        Returns the stiffness matrix of the plate.

        Returns
        -------
        numpy.ndarray
            The ABDS matrix of the plate.
        """
        ABDS_ = super().elastic_stiffness_matrix()
        ABDS = np.zeros([5, 5], dtype=ABDS_.dtype)
        ABDS[:3, :3] = ABDS_[3:6, 3:6]
        ABDS[3:, 3:] = ABDS_[6:, 6:]
        return ABDS
