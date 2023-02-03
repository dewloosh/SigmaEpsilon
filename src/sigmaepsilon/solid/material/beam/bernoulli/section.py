import numpy as np

from sectionproperties.pre.pre import DEFAULT_MATERIAL, Material
from sectionproperties.pre.geometry import Geometry
from sectionproperties.pre.library.steel_sections import circular_hollow_section as CHS
from sectionproperties.pre.library.steel_sections import (
    rectangular_hollow_section as RHS,
)
from sectionproperties.pre.library.primitive_sections import rectangular_section as RS
from sectionproperties.pre.library.steel_sections import i_section
from sectionproperties.pre.library.steel_sections import tapered_flange_i_section as TFI
from sectionproperties.pre.library.steel_sections import channel_section as PFC
from sectionproperties.pre.library.steel_sections import tapered_flange_channel as TFC
from sectionproperties.pre.library.steel_sections import tee_section
from sectionproperties.analysis.section import Section

from dewloosh.core.wrapping import Wrapper
from linkeddeepdict.tools.kwargtools import getallfromkwargs
from neumann import atleastnd, ascont
from polymesh.utils import centralize
from polymesh.trimesh import TriMesh
from polymesh.tetmesh import TetMesh
from polymesh.utils.topology import T6_to_T3, detach_mesh_bulk
from .utils import calc_beam_stresses_2d, calc_beam_stresses_4d
from ...meta import MaterialModel


def generate_mesh(
    geometry: Geometry, *, l_max: float = None, a_max: float = None, n_max: int = None
) -> Geometry:
    """
    Calculates a float describing the maximum mesh element area to be used
    for the finite-element mesh of the section.

    Parameters
    ----------
    geometry : :class:`sectionproperties.pre.geometry.Geometry`
        Describes the shape of the section, i.e. 'I', 'H', 'RHS', etc.
    l_max : float, Optional
        Maximum element edge length. Default is None.
    a_max : float, Optional
        Maximum element area. Default is None.
    n_max : int, Optional
        Maximum number of elements. Default is None.

    Note
    ----
    The value of the mesh size is derived from the assumption of a perfectly
    regular mesh of equilateral triangles.

    Returns
    -------
    :class:`sectionproperties.pre.geometry.Geometry`
        The geometry object provided with a finite element mesh.
    """
    area = geometry.calculate_area()
    mesh_sizes_max = []
    if isinstance(l_max, float):
        mesh_sizes_max.append(l_max**2 * np.sqrt(3) / 4)
    if isinstance(a_max, float):
        mesh_sizes_max.append(a_max)
    if isinstance(n_max, int):
        mesh_sizes_max.append(area / n_max)

    mesh_size_max = None
    if len(mesh_sizes_max) > 0:
        mesh_size_max = min(mesh_sizes_max)

    geometry.create_mesh(mesh_sizes=[mesh_size_max])
    return geometry


def get_section(
    shape, *args, mesh_params: dict = None, material: Material = None, **section_params
) -> Section:
    """
    Returns a :class:`sectionproperties.analysis.section.Section` instance.

    The parameters in `section_params` are forwarded to the appropriate constructor of the
    `sectionproperties` library. For the available parameters, see their documentation:

        https://sectionproperties.readthedocs.io/en/latest/rst/section_library.html

    Parameters
    ----------
    shape : str
        Describes the shape of the section. The currently supported section types
        are 'I', 'CHS', 'RHS', 'TFI', 'PFC', 'TFC', 'T'.
    mesh_params : dict, Optional
        A dictionary of parameters controlling mesh generation. Default is None.
        For the possible keys and values see :func:`generate_mesh`, to which
        these parameters are forwarded to. Default is None.
    material : :class:`sectionproperties.pre.pre.Material`, Optional
        The material of the section. If not specified, a default material is
        used. Default is None.
    **section_params : dict
        Parameters required for a given section. See the documentation of
        the `sectionproperties` library for more details. The parameters
        are only required, if the first argument is a string.

    Note
    ----
    Specification of a material is only necessary, if stresses are to be
    calculated. If the reason of creating the section is the calculation of
    geometrical properties of a section, the choice of material is irrelevant.

    Returns
    -------
    :class:`sectionproperties.analysis.section.Section`
        An object representing a cross-section of a beam.

    Examples
    --------
    >>> from sigmaepsilon.solid.model.bernoulli import get_section
    >>> mesh_params = dict(n_min=100, n_max=500)
    >>> section = get_section('CHS', d=1.0, t=0.1, n=64, mesh_params=mesh_params)
    """

    material = DEFAULT_MATERIAL if material is None else material
    geom, ms = None, None
    if shape == "CHS":
        geom = CHS(
            d=section_params["d"],
            t=section_params["t"],
            n=section_params.get("n", 64),
            material=material,
        )
        ms = section_params["t"]
    elif shape == "RS":
        keys = ["d", "b"]
        params = getallfromkwargs(keys, **section_params)
        geom = RS(material=material, **params)
        ms = section_params["t"]
    elif shape == "RHS":
        keys = ["d", "b", "t", "n_out", "n_r"]
        params = getallfromkwargs(keys, **section_params)
        geom = RHS(material=material, **params)
        ms = section_params["t"]
    elif shape == "I":
        keys = ["d", "b", "t_f", "t_w", "r", "n_r"]
        params = getallfromkwargs(keys, **section_params)
        geom = i_section(material=material, **params)
        ms = min(section_params["t_f"], section_params["t_w"])
    elif shape == "TFI":
        keys = ["d", "b", "t_f", "t_w", "r_r", "r_f", "alpha", "n_r"]
        params = getallfromkwargs(keys, **section_params)
        geom = TFI(material=material, **params)
        ms = min(section_params["t_f"], section_params["t_w"])
    elif shape == "PFC":
        keys = ["d", "b", "t_f", "t_w", "r", "n_r"]
        params = getallfromkwargs(keys, **section_params)
        geom = PFC(material=material, **params)
        ms = min(section_params["t_f"], section_params["t_w"])
    elif shape == "TFC":
        keys = ["d", "b", "t_f", "t_w", "r_r", "r_f", "alpha", "n_r"]
        params = getallfromkwargs(keys, **section_params)
        geom = TFC(material=material, **params)
        ms = min(section_params["t_f"], section_params["t_w"])
    elif shape == "T":
        keys = ["d", "b", "t_f", "t_w", "r", "n_r"]
        params = getallfromkwargs(keys, **section_params)
        geom = tee_section(material=material, **params)
        ms = min(section_params["t_f"], section_params["t_w"])
    else:
        raise NotImplementedError(
            "Section type <{}> is not yet implemented :(".format(shape)
        )

    if geom is not None:
        if mesh_params is None:
            assert ms is not None, "Invalid input!"
            mesh_params = dict(l_max=ms)
        assert isinstance(mesh_params, dict)
        geom = generate_mesh(geom, **mesh_params)
        return Section(geom)
    raise RuntimeError("Unable to get section.")


class BeamSection(Wrapper, MaterialModel):
    """
    Wraps an instance of `sectionproperties.analysis.section.Section` and
    adds a little here and there to make some of the functionality more
    accessible.

    Parameters
    ----------
    *args : tuple, Optional
        The first parameter can be a string referring to a section type, or an
        instance of :class:`sectionproperties.analysis.section.Section`. In the
        former case, parameters of the section can be provided as keyword arguments,
        which are then forwarded to :func:`get_section`, see its documentation
        for further details.
    **kwargs : dict, Optional
        The parameters of a section as keyword arguments, only if the first
        positional argument is a string (see above).
    mesh_params : dict, Optional
        A dictionary controlling the density of the mesh of the section.
        Default is None.
    material : :class:`sectionproperties.pre.pre.Material`, Optional
        The material of the section. If not specified, a default material is
        used. Default is None.
    wrap : :class:`sectionproperties.analysis.section.Section`
        A section object to be wrapped. It can also be provided as the
        first positional argument.
    Notes
    -----
    The implementation here only covers homogeneous cross sections. If you want
    to define an inhomogeneous section, it must be explicity given either as
    the first position argument, or with the keyword argument `wrap`.

    Examples
    --------
    >>> from sigmaepsilon.solid import BeamSection
    >>> section = BeamSection(get_section('CHS', d=1.0, t=0.1, n=64))

    or simply provide the shape as the first argument and everything
    else with keyword arguments:

    >>> section = BeamSection('CHS', d=1.0, t=0.1, n=64)

    Plot a section with Matplotlib using 6-noded triangles:

    >>> import matplotlib.pyplot as plt
    >>> from dewloosh.mpl import triplot
    >>> section = BeamSection('CHS', d=1.0, t=0.3, n=32,
    >>>                       mesh_params=dict(n_max=20))
    >>> triobj = section.trimesh(T6=True).to_triobj()
    >>> fig, ax = plt.subplots(figsize=(4, 2))
    >>> triplot(triobj, fig=fig, ax=ax, lw=0.1)
    """

    def __init__(
        self,
        *args,
        wrap=None,
        shape=None,
        mesh_params=None,
        material: Material = None,
        **kwargs
    ):
        if len(args) > 0:
            try:
                if isinstance(args[0], str):
                    wrap = get_section(
                        args[0], mesh_params=mesh_params, material=material, **kwargs
                    )
                else:
                    if shape is None:
                        if isinstance(args[0], Section):
                            wrap = args[0]
                    else:
                        wrap = get_section(
                            shape, mesh_params=mesh_params, material=material, **kwargs
                        )
            except Exception:
                raise RuntimeError("Invalid input.")
        super().__init__(*args, wrap=wrap, **kwargs)
        self.props = None

    def coords(self) -> np.ndarray:
        """
        Returns centralized vertex coordinates of the supporting
        point cloud as a numpy array.
        """
        return centralize(np.array(self.mesh["vertices"]))

    def topology(self) -> np.ndarray:
        """
        Returns vertex indices of T6 triangles as a numpy array.
        """
        return np.array(self.mesh["triangles"].tolist())

    def trimesh(self, subdivide=False, T6=False, **kwargs) -> TriMesh:
        """
        Returns the mesh of the section as a collection of T3 triangles.
        Keyword arguments are forwarded to the constructor of
        :class:`polymesh.tri.trimesh.TriMesh`.

        Parameters
        ----------
        T6 : boolean, Optional
            If False, the original T6 trinagles are transformed into T3
            triangles. Default is False.
        subdivide : boolean, Optional
            Controls how the T6 triangles are transformed into T3 triangles,
            if the argument 'T6' is False. If True, the T6 triangles
            are subdivided into 4 T3 triangles. If False, T3 triangles are
            formed by the corners of T6 triangles only, and all the remaining
            nodes are neglected.

        See Also
        --------
        :class:`polymesh.tri.trimesh.TriMesh`

        Examples
        --------
        >>> from sigmaepsilon.solid import BeamSection
        >>> section = BeamSection(get_section('CHS', d=1.0, t=0.1, n=64))
        >>> trimesh = section.trimesh()
        """
        points, triangles = self.coords(), self.topology()
        if not T6:
            if subdivide:
                path = np.array([[0, 5, 4], [5, 1, 3], [3, 2, 4], [5, 3, 4]], dtype=int)
                points, triangles = T6_to_T3(points, triangles, path=path)
            else:
                points, triangles = detach_mesh_bulk(points, triangles[:, :3])
        return TriMesh(points=points, triangles=triangles, **kwargs)

    def extrude(self, *args, length=None, frame=None, N=None, **kwargs) -> TetMesh:
        """
        Creates a 3d tetragonal mesh from the section.

        Parameters
        ----------
        length : float
            Length of the beam.
        N : int
            Number of subdivisions along the length of the beam.
        frame : numpy.ndarray
            A 3x3 matrix representing an orthonormal coordinate frame.
        """
        return self.trimesh(frame=frame).extrude(h=length, N=N)

    def calculate_geometric_properties(self, *args, **kwargs):
        return self._wrapped.calculate_geometric_properties(*args, **kwargs)

    def calculate_warping_properties(self, *args, **kwargs):
        return self._wrapped.calculate_warping_properties(*args, **kwargs)

    @property
    def A(self) -> float:
        """
        Returns the cross-sectional area.
        """
        return self.section_props.area

    @property
    def Ix(self) -> float:
        """
        Returns the second moment of inertia around 'x'.
        """
        return self.Iy + self.Iz

    @property
    def Iy(self) -> float:
        """
        Returns the second moment of inertia around 'y'.
        """
        return self.section_props.ixx_c

    @property
    def Iz(self) -> float:
        """
        Returns the second moment of inertia around 'z'.
        """
        return self.section_props.iyy_c

    @property
    def geometric_properties(self) -> dict:
        """
        Returns the geometric properties of the section.
        """
        return {"A": self.A, "Ix": self.Ix, "Iy": self.Iy, "Iz": self.Iz}

    @property
    def section_properties(self) -> dict:
        """
        Returns all properties of the section.
        """
        return self.props

    def calculate_section_properties(self, separate=False) -> dict:
        """
        Retruns a dictionary containing the properties of the section.
        """
        if not separate:
            # 0-xx 1-yy 2-zz 3-yz 4-xz 5-xy
            strs = np.zeros((6, 6, len(self.mesh["vertices"])))
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
                strs[0, i, :] = stresses[0]["sig_zz"]  # xx
                strs[5, i, :] = stresses[0]["sig_zx"]  # xy
                strs[4, i, :] = stresses[0]["sig_zy"]  # xz
            else:
                raise NotImplementedError

        section_properties["stress-factors"] = strs
        self.props = section_properties
        return section_properties

    def get_section_properties(self) -> dict:
        """
        Returns all properties of the section.
        """
        if self.props is not None:
            return self.props
        else:
            self.calculate_section_properties()

    def elastic_stiffness_matrix(self, E: float = None, nu: float = None) -> np.ndarray:
        """
        Returns the elastic stiffness matrix of the section.

        Parameters
        ----------
        E : float, Optional
            Young's modulus. If not provided, the value is obtained
            from the associated material of the section, if there is one.
            Default is None.
        nu : float, Optional
            Poisson's ratio. If not provided, the value is obtained
            from the associated material of the section, if there is one.
            Default is None.

        Returns
        -------
        numpy.ndarray
            The 4x4 elastic ABDS matrix of the section.
        """
        if E is None:
            E = self.geometry.material.elastic_modulus
            nu = self.geometry.material.poissons_ratio
        G = E / (2 * (1 + nu))
        return np.array(
            [
                [E * self.A, 0, 0, 0],
                [0, G * self.Ix, 0, 0],
                [0, 0, E * self.Iy, 0],
                [0, 0, 0, E * self.Iz],
            ]
        )

    def calculate_stresses(self, forces: np.ndarray) -> np.ndarray:
        """
        Returns stresses for all points of the mesh. The internal forces
        are expected in the order Fx, Fy, Fz, Mx, My, Mz, where x, y and z
        denote the local axes of the section, the stress components are
        returned in the order s11, s22, s33, s12, s13, s23.

        Parameters
        ----------
        forces : numpy.ndarray
            It can be either

                A 1d float array representing all internal forces of a beam.

                A 2d float array. In this case it is assumed, that the provided
                data has the shape (nX, nSTRE), where 'nX' is the number of records
                in the data.

                A 3d float array. In this case it is assumed, that the provided
                data has the shape (nE, nP, nSTRE), representing
                all internal forces (nSTRE) for multiple elements(nE) and
                evaluation points (nP).

                A 4d float array of shape (nE, nP, nSTRE, nRHS), representing
                all internal forces (nSTRE) for multiple elements(nE), load cases(nRHS)
                and evaluation points (nP).

        Returns
        -------
        numpy.ndarray
            A numpy float array that can be

                A 2d array of shape (nPS, nS),

                A 5d array of shape (nE, nP, nPS, nS, nRHS),

            where

                nPS : the number of points of the mesh of the section

                nS : the number of stress components

                nE : the number of finite elements

                nP : the number of evaluation points for an element

                nRHS : the number of right hand sides (possibly load cases)

        Notes
        -----
        This only covers homogeneous cross sections.
        """
        assert isinstance(forces, np.ndarray)
        factors = self.props["stress-factors"]
        nD = len(forces.shape)
        if nD == 1:
            f_ = ascont(atleastnd(forces, n=2, front=True))
            return calc_beam_stresses_2d(f_, factors)[0, ...]
        elif nD == 2:
            return calc_beam_stresses_2d(forces, factors)
        elif nD == 3:
            f_ = ascont(atleastnd(forces, n=4, back=True))
            return calc_beam_stresses_4d(f_, factors)[..., 0]
        elif nD == 4:
            return calc_beam_stresses_4d(forces, factors)
        else:
            raise RuntimeError("Unrecognizable shape of internal forces.")
