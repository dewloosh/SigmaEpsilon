# -*- coding: utf-8 -*-
from typing import Union
import numpy as np
import sympy as sy
from numpy import ndarray
from numpy.linalg import inv

from ..material.hooke.utils import group_mat_params, get_iso_params
from ..material.hooke.sym import smat_sym_ortho_3d

from .metashell import Surface, Layer


__all__ = ['Membrane', 'MembraneLayer']


class MembraneLayer(Layer):
    """
    Helper object for the stress analysis of a layer of a membrane.
    """

    __loc__ = [-1., 1.]  # locations of approximation points
    __shape__ = (3, 3)  # shape of the ABDS matrix

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # locations of discrete points along the thickness
        self.zi = [self.loc_to_z(loc) for loc in self.__loc__]

    def material_stiffness_matrix(self) -> ndarray:
        """
        Returns and stores the transformed material stiffness matrix.
        """
        Cm_126 = self.hooke
        T_126 = self.rotation_matrix()
        R_126 = np.diag([1, 1, 2])
        C_126 = T_126 @ Cm_126 @ inv(R_126) @ T_126.T @ R_126
        C_126[np.abs(C_126) < 1e-12] = 0.0
        self.C_126 = C_126
        return C_126

    def rotation_matrix(self) -> ndarray:
        """
        Returns the transformation matrix.
        """
        angle = self.angle * np.pi / 180
        T_126 = np.zeros([3, 3])
        T_126[0, 0] = np.cos(angle)**2
        T_126[0, 1] = np.sin(angle)**2
        T_126[0, 2] = -np.sin(2 * angle)
        T_126[1, 0] = T_126[0, 1]
        T_126[1, 1] = T_126[0, 0]
        T_126[1, 2] = -T_126[0, 2]
        T_126[2, 0] = np.cos(angle) * np.sin(angle)
        T_126[2, 1] = -T_126[2, 0]
        T_126[2, 2] = np.cos(angle)**2 - np.sin(angle)**2
        return T_126

    def stiffness_matrix(self) -> ndarray:
        """
        Returns the stiffness contribution of the layer.
        """
        return self.material_stiffness_matrix() * (self.tmax - self.tmin)

    def approxfunc(self, data):
        """
        Returns a function that can be used for approximations thorugh
        the thickness.
        """
        z0, z1 = self.zi
        z = np.array([[1, z0], [1, z1]])
        a, b = np.linalg.inv(z) @ np.array(data)
        return lambda z: a + b*z


class Membrane(Surface):
    """
    Helper object for the stress analysis of a membrane.
    
    Example
    -------
    >>> from sigmaepsilon.solid.model import Membrane as Model
    >>> model = {
    >>>     '0' : {
    >>>         'hooke' : Model.Hooke(E=2100000, nu=0.3),
    >>>         'angle' : 0., 
    >>>         'thickness' : 0.1
    >>>         },
    >>>     }
    >>> C = Model.from_dict(model).stiffness_matrix()
    
    """
    
    __layerclass__ = MembraneLayer
    
    __imap__ = {0 : (0, 0), 1 : (1, 1), 2 : (2, 2),
                5 : (1, 2), 4 : (0, 2), 3 : (0, 1)}
        
    @classmethod    
    def Hooke(cls, *args, symbolic=False, **kwargs) -> Union[sy.Matrix, np.ndarray]:
        """
        Returns a Hooke matrix appropriate for membranes.
        
        Parameters
        ----------
        symbolic : bool, Optional
            If Truem a symbolic matrix is returned.
        
        Returns
        -------
        Union[sy.Matrix, np.ndarray]
            The matrix in symbolic or numeric form.
            
        Examples
        --------
        >>> from sigmaepsilon.solid.model import Membrane as Model
        >>> Model.Hooke(E=2100000, nu=0.3)
        
        >>> Model.Hooke(symbolic=True)
            
        """
        if symbolic:
            S = smat_sym_ortho_3d()
            S.row_del(2)
            S.row_del(2)
            S.row_del(2)
            S.col_del(2)
            S.col_del(2)
            S.col_del(2)  
            return S
        else:
            E, NU, G = group_mat_params(**kwargs)
            nE, nNU, nG = len(E), len(NU), len(G)
            nParams = sum([nE, nNU, nG])
            if nParams == 2:
                constants = get_iso_params(**E, **NU, **G)
            S = cls.Hooke(symbolic=True)
            subs = {s : constants[str(s)] for s in S.free_symbols}
            S = S.subs([(sym, val) for sym, val in subs.items()])
            S = np.array(S, dtype=float)
            return np.linalg.inv(S)
            
    def stiffness_matrix(self) -> ndarray:
        """
        Returns the stiffness matrix of the membrane.
                
        Returns
        -------
        ndarray
            The ABDS matrix of the membrane.
            
        """
        self.ABDS = super().stiffness_matrix()
        self.SDBA = np.linalg.inv(self.ABDS)
        return self.ABDS
