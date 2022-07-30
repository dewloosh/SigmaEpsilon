# -*- coding: utf-8 -*-
from ....mesh.topo import detach_mesh_bulk
import numpy as np
from typing import Tuple

from sectionproperties.pre.pre import DEFAULT_MATERIAL
from sectionproperties.pre.geometry import Geometry
from sectionproperties.pre.library.steel_sections import circular_hollow_section as CHS
from sectionproperties.pre.library.steel_sections import rectangular_hollow_section as RHS
from sectionproperties.pre.library.steel_sections import i_section
from sectionproperties.pre.library.steel_sections import tapered_flange_i_section as TFI
from sectionproperties.pre.library.steel_sections import channel_section as PFC
from sectionproperties.pre.library.steel_sections import tapered_flange_channel as TFC
from sectionproperties.pre.library.steel_sections import tee_section
from sectionproperties.analysis.section import Section

from ....core.wrapping import Wrapper
from ....core.tools.kwargtools import getallfromkwargs

from ....mesh.utils import centralize
from ....mesh.tri.trimesh import TriMesh


def get_section_metadata(shape: str, material=None, **section: dict) -> Tuple[Geometry, float]:
    """
    Returns the metadata of a section necessary to create a section object instance.

    The parameters in `section` are forwarded to the appropriate constructor of the 
    `sectionproperties` library. For the available parameters, see their documentation:

        https://sectionproperties.readthedocs.io/en/latest/rst/section_library.html

    Parameters
    ----------
    shape : str
        Describes the shape of the section, i.e. 'I', 'H', 'RHS', etc.

    **section : dict
        Parameters required for a given section. See the documentation of
        the `sectionproperties` library for more details.

    Returns
    -------
    sectionproperties.pre.geometry.Geometry
        A geometry object.

    float
        The mesh size required by the section.

    Examples
    --------
    >>> geom, mesh_size = get_section_metadata('CHS', D=1.0, t=0.1, n=64)

    """
    material = DEFAULT_MATERIAL if material is None else material
    geom, mesh_size = None, None
    if shape == 'CHS':
        geom = CHS(d=section['D'], t=section['t'], n=section.get('n', 64), 
                   material=material)
        mesh_size = section['t']
    elif shape == 'RHS':
        keys = ['d', 'b', 't', 'n_out', 'n_r']
        params = getallfromkwargs(keys, **section)
        geom = RHS(material=material, **params)
        mesh_size = section['t']
    elif shape == 'I':
        keys = ['d', 'b', 't_f', 't_w', 'r', 'n_r']
        params = getallfromkwargs(keys, **section)
        geom = i_section(material=material, **params)
        mesh_size = min(section['t_f'], section['t_w'])
    elif shape == 'TFI':
        keys = ['d', 'b', 't_f', 't_w', 'r_r', 'r_f', 'alpha', 'n_r']
        params = getallfromkwargs(keys, **section)
        geom = TFI(material=material, **params)
        mesh_size = min(section['t_f'], section['t_w'])
    elif shape == 'PFC':
        keys = ['d', 'b', 't_f', 't_w', 'r', 'n_r']
        params = getallfromkwargs(keys, **section)
        geom = PFC(material=material, **params)
        mesh_size = min(section['t_f'], section['t_w'])
    elif shape == 'TFC':
        keys = ['d', 'b', 't_f', 't_w', 'r_r', 'r_f', 'alpha', 'n_r']
        params = getallfromkwargs(keys, **section)
        geom = TFC(material=material, **params)
        mesh_size = min(section['t_f'], section['t_w'])
    elif shape == 'T':
        keys = ['d', 'b', 't_f', 't_w', 'r', 'n_r']
        params = getallfromkwargs(keys, **section)
        geom = tee_section(material=material, **params)
        mesh_size = min(section['t_f'], section['t_w'])
    else:
        raise NotImplementedError(
            "Section type <{}> is not yet implemented :(".format(shape))
    return geom, mesh_size


def get_section(shape, *args, **kwargs) -> Section:
    """
    Returns a `sectionproperties.analysis.section.Section` instance.

    The parameters in `kwargs` are forwarded to the appropriate constructor of the 
    `sectionproperties` library. For the available parameters, see their documentation:

        https://sectionproperties.readthedocs.io/en/latest/rst/section_library.html

    Parameters
    ----------
    shape : str
        Describes the shape of the section, i.e. 'I', 'H', 'RHS', etc.

    **kwargs : dict
        Parameters required for a given section. See the documentation of
        the `sectionproperties` library for more details.

    Returns
    -------
    sectionproperties.analysis.section.Section

    Examples
    --------
    >>> section = get_section('CHS', D=1.0, t=0.1, n=64)

    """
    geom, mesh_size = get_section_metadata(shape, **kwargs)
    if geom is not None:
        assert isinstance(mesh_size, float)
        geom.create_mesh(mesh_sizes=[mesh_size])
        return Section(geom)
    raise RuntimeError("Unable to get section.")


class BeamSection(Wrapper):

    def __init__(self, *args, wrap=None, shape=None, **kwargs):
        if len(args) > 0:
            try:
                if shape is None:
                    if isinstance(args[0], Section):
                        wrap = args[0]
                else:
                    wrap = get_section(shape, **kwargs)
            except:
                raise RuntimeError("Invalid input.")
        super().__init__(*args, wrap=wrap, **kwargs)

    def coords(self):
        """
        Returns centralized vertex coordinates of the supporting 
        point cloud.
        """
        return centralize(np.array(self.mesh['vertices']))

    def topology(self):
        """Returns vertex indices of T6 triangles."""
        return np.array(self.mesh['triangles'].tolist())

    def trimesh(self, **kwargs) -> TriMesh:
        """
        Returns the mesh of the section as a collection of T3 triangles.
        Keyword arguments are forwarded to the constructor of 
        :class:`dewloosh.mesh.tri.trimesh.TriMesh`.
        
        See Also
        --------
        :class:`dewloosh.mesh.tri.trimesh.TriMesh`
        
        """
        points, triangles = self.coords(), self.topology() 
        points, triangles = detach_mesh_bulk(points, triangles[:, :3])
        return TriMesh(points=points, triangles=triangles, **kwargs)
    
    def extrude(self, *args, length=None, frame=None, N=None, **kwargs):
        return self.trimesh(frame=frame).extrude(h=length, N=N)

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
    
    def model_stiffness_matrix(self, E=None, nu=None):
        if E is None:
            E = self.geometry.material.elastic_modulus
            nu = self.geometry.material.poissons_ratio
        G = E / (2 * (1 + nu))
        return np.array([
            [E*self.A, 0, 0, 0],
            [0, G*self.Ix, 0, 0],
            [0, 0, E*self.Iy, 0],
            [0, 0, 0, E*self.Iz]
        ])
