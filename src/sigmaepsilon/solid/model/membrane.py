# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import inv

from ..material.hooke.utils import group_mat_params, get_iso_params
from ..material.hooke.sym import smat_sym_ortho_3d

from .metashell import Surface, Layer


__all__ = ['Membrane']


class MembraneLayer(Layer):

    __loc__ = [-1., 1.]
    __shape__ = (3, 3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # locations of discrete points along the thickness
        self.zi = [self.loc_to_z(loc) for loc in self.__loc__]

    def material_stiffness_matrix(self):
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

    def rotation_matrix(self):
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

    def stiffness_matrix(self):
        """
        Returns the stiffness contribution to the layer.
        """
        return self.material_stiffness_matrix() * (self.tmax - self.tmin)

    def approxfunc(self, data):
        z0, z1 = self.zi
        z = np.array([[1, z0], [1, z1]])
        a, b = np.linalg.inv(z) @ np.array(data)
        return lambda z: a + b*z


class Membrane(Surface):
    
    __layerclass__ = MembraneLayer
    
    __imap__ = {0 : (0, 0), 1 : (1, 1), 2 : (2, 2),
                5 : (1, 2), 4 : (0, 2), 3 : (0, 1)}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @classmethod    
    def Hooke(cls, *args, symbolic=False, **kwargs):
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
            
    def stiffness_matrix(self):
        self.ABDS = super().stiffness_matrix()
        self.SDBA = np.linalg.inv(self.ABDS)
        return self.ABDS
