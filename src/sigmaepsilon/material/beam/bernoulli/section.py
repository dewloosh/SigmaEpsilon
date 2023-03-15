import numpy as np

from neumann import atleastnd, ascont
from polymesh.section import LineSection

from ....utils.material.bernoulli import calc_beam_stresses_2d, calc_beam_stresses_4d
from ....core.material import MaterialLike


class BeamSection(LineSection, MaterialLike):
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
    >>> from sigmaepsilon import BeamSection
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

    def calculate_section_properties(self, separate: bool = False) -> dict:
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
