# -*- coding: utf-8 -*-
import numpy as np

from dewloosh.core.wrapping import Wrapper
from dewloosh.mesh.utils import centralize


class BeamSection(Wrapper):

    def __init__(self, *args, wrap=None, **kwargs):
        if len(args) > 0:
            try:
                from sectionproperties.analysis.section import Section
                if isinstance(args[0], Section):
                    wrap = args[0]
            except:
                raise TypeError("Invalid type for first argument.")
        super().__init__(*args, wrap=wrap, **kwargs)
        self.strength_props = {}

    def coords(self):
        """
        Returns vertex coordinates of the supporting point cloud.
        """
        return centralize(np.array(self.mesh['vertices']))

    def topology(self):
        """
        Returns vertex indices of T6 triangles.
        """
        return np.array(self.mesh['triangles'].tolist())
    
    def TriMesh(self):
        raise NotImplementedError

    def calculate_geometric_properties(self, *args, **kwargs):
        return self._wrapped.calculate_geometric_properties(*args, **kwargs)
    
    def calculate_warping_properties(self, *args, **kwargs):
        return self._wrapped.calculate_warping_properties(*args, **kwargs)

    @property
    def A(self):
        return self.section_props.area
    
    @property
    def Ix(self):
        return self.Iy + self.Iz
    
    @property
    def Iy(self):
        return self.section_props.ixx_c
    
    @property
    def Iz(self):
        return self.section_props.iyy_c
    
    @property
    def geometric_properties(self):
        return {'A': self.A, 'Ix': self.Ix, 'Iy': self.Iy, 'Iz': self.Iz}
    
    def get_section_properties(self, separate=False):

        if not separate:
            # 0-xx 1-yy 2-zz 3-yz 4-xz 5-xy
            strs = np.zeros((6, 6, len(self.mesh['vertices'])))
            """strs = {'xx': {}, 'yy': {}, 'zz': {}, 'xy': {}, 'xz': {}, 'yz': {}}"""
        else:
            raise NotImplementedError

        self.calculate_geometric_properties()
        section_properties = self.geometric_properties

        self.calculate_warping_properties()
        _strs = np.eye(6)
        for i in range(6):
            N, Vy, Vx, Mzz, Mxx, Myy = _strs[i]
            stress_post = self.calculate_stress(
                N=N, Vy=Vy, Vx=Vx, Mzz=Mzz, Mxx=Mxx, Myy=Myy
            )
            stresses = stress_post.get_stress()
            if not separate:
                # this assumes that the section is homogeneous, hence the 0 index
                strs[0, i, :] = stresses[0]['sig_zz']  # xx
                strs[5, i, :] = stresses[0]['sig_zx']  # xy
                strs[4, i, :] = stresses[0]['sig_zy']  # xz
            else:
                raise NotImplementedError

        section_properties['stress-factors'] = strs
        return section_properties
