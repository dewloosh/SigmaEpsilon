import numpy as np
from numpy import ndarray
from typing import Union, Callable, Iterable

from scipy import interpolate

from neumann import atleast1d, atleastnd, ascont
from neumann.utils import to_range_1d

from ...utils.fem.cells.bernoulli import (
    shape_function_matrix_Bernoulli_bulk,
    body_load_vector_Bernoulli,
    global_shape_function_derivatives_Bernoulli_bulk as gdshpB,
    lumped_mass_matrices_direct_Bernoulli as dlump,
)

from ...utils.fem.postproc import (
    approx_element_solution_bulk,
    calculate_internal_forces_bulk,
)
from ..material.beam import (
    BernoulliBeam,
    _postproc_bernoulli_internal_forces_H_,
    _postproc_bernoulli_internal_forces_L_,
)

__all__ = ["BernoulliBase"]


class BernoulliBase(BernoulliBeam):
    """
    Skeleton class for 1d finite elements, whose bending behaviour
    is governed by the Bernoulli theory. The implementations covered
    by this class are independent of the number of nodes of the cells.
    Specification of an actual finite element consists of specifying class
    level attributes only.

    See Also
    --------
    :class:`~sigmaepsilon.fem.cells.bernoulli2.Bernoulli2`
    :class:`~sigmaepsilon.fem.cells.bernoulli3.Bernoulli3`
    """

    qrule: str = None
    quadrature: dict = None
    shpfnc: Callable = None
    dshpfnc: Callable = None
    dshpfnc_geom: Callable = None

    def shape_function_values(
        self,
        pcoords: Union[float, Iterable[float]],
        *_,
        rng: Iterable = None,
        lengths: Iterable[float] = None,
        **__,
    ) -> ndarray:
        """
        Evaluates the shape functions at the points specified by 'pcoords'.

        Parameters
        ----------
        pcoords : float or Iterable[float]
            Locations of the evaluation points.
        rng : Iterable, Optional
            The range in which the locations ought to be understood,
            typically [0, 1] or [-1, 1]. Default is [0, 1].
        lengths : Iterable, Optional
            The lengths of the beams in the block. Default is None.

        Returns
        -------
        numpy.ndarray
            The returned array has a shape of (nP, nNE=2, nDOF=6), where nP,
            nNE and nDOF stand for the number of evaluation points, nodes
            per element and number of degrees of freedom, respectively.
        """
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        pcoords = atleast1d(np.array(pcoords))
        pcoords = to_range_1d(pcoords, source=rng, target=[-1, 1])
        lengths = self.lengths() if lengths is None else lengths
        return self.__class__.shpfnc(pcoords, lengths).astype(float)

    def shape_function_derivatives(
        self,
        pcoords: Union[float, Iterable[float]] = None,
        *_,
        rng: Iterable = None,
        jac: ndarray = None,
        dshp: ndarray = None,
        lengths: Iterable[float] = None,
        **__,
    ) -> ndarray:
        """
        Evaluates the shape function derivatives (up to third) at the points specified
        by 'pcoords'. The function either returns the derivatives on the master or the actual
        element, depending on the inputs.

        Valid combination of inputs are:

            * 'pcoords' and optionally 'jac' : this can be used to calculate the derivatives
              in both the global ('jac' is provided) and the master frame ('jac' is not provided).

            * 'dshp' and 'jac' : this combination can only be used to return the derivatives
              wrt. to the global frame.

        Parameters
        ----------
        pcoords : float or Iterable, Optional
            Locations of the evaluation points. Default is None.
        rng : Iterable, Optional
            The range in which the locations ought to be understood,
            typically [0, 1] or [-1, 1]. Only if 'pcoords' is provided.
            Default is [0, 1].
        lengths : Iterable, Optional
            The lengths of the beams in the block. Default is None.
        jac : Iterable, Optional
            The jacobian matrix as a float array of shape (nE, nP, 1, 1), evaluated for
            each point in each cell. Default is None.
        dshp : Iterable, Optional
            The shape function derivatives of the master element as a float array of
            shape (nP, nNE, nDOF=6, 3), evaluated at a 'nP' number of points.
            Default is None.

        Returns
        -------
        numpy.ndarray
            The returned array has a shape of (nP, nNE=2, nDOF=6, 3), where nP, nNE and
            nDOF stand for the number of evaluation points, nodes per element and number of
            degrees of freedom, respectively. Number 3 refers to first, second and third
            derivatives.

        Notes
        -----
        The returned array is always 4 dimensional, even if there is only one
        evaluation point.
        """
        if pcoords is not None:
            lengths = self.lengths() if lengths is None else lengths
            # calculate derivatives wrt. the parametric coordinates in the range [-1, 1]
            pcoords = atleast1d(np.array(pcoords))
            rng = np.array([-1.0, 1.0]) if rng is None else np.array(rng)
            pcoords = to_range_1d(pcoords, source=rng, target=[-1.0, 1.0])
            dshp = self.__class__.dshpfnc(pcoords, lengths)
            if jac is None:
                # return derivatives wrt. the master frame
                return dshp.astype(float)
            else:
                # return derivatives wrt. the global frame
                return gdshpB(dshp, jac).astype(float)
        elif dshp is not None and jac is not None:
            # return derivatives wrt. the local frame
            return gdshpB(dshp, jac).astype(float)

    def shape_function_matrix(
        self,
        pcoords: Union[float, Iterable[float]] = None,
        *_,
        rng: Iterable = None,
        lengths: Iterable[float] = None,
        **__,
    ) -> ndarray:
        """
        Evaluates the shape function matrix at the points specified by 'pcoords'.

        Parameters
        ----------
        pcoords : float or Iterable
            Locations of the evaluation points.
        rng : Iterable, Optional
            The range in which the locations ought to be understood,
            typically [0, 1] or [-1, 1]. Default is [0, 1].

        Other Parameters
        ----------------
        These parameters are for advanced users and can be omitted.
        They are here only to avoid repeated evaulation of common quantities.

        lengths : Iterable, Optional
            The lengths of the beams in the block. Default is None.

        Returns
        -------
        numpy.ndarray
            The returned array has a shape of (nE, nP, nDOF, nDOF * nNE), where nE, nP,
            nNE and nDOF stand for the number of elements, evaluation points, nodes
            per element and number of degrees of freedom respectively.

        Notes
        -----
        The returned array is always 4 dimensional, even if there is only one
        evaluation point.
        """
        pcoords = atleast1d(np.array(pcoords))
        rng = np.array([-1.0, 1.0]) if rng is None else np.array(rng)
        pcoords = to_range_1d(pcoords, source=rng, target=[-1, 1])
        lengths = self.lengths() if lengths is None else lengths
        dshp = self.__class__.dshpfnc(pcoords, lengths)
        ecoords = self.local_coordinates()
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)
        shp = self.shape_function_values(pcoords, rng=rng)
        gdshp = self.shape_function_derivatives(jac=jac, dshp=dshp)
        return shape_function_matrix_Bernoulli_bulk(shp, gdshp).astype(float)

    def integrate_body_loads(self, values: ndarray) -> ndarray:
        """
        Returns nodal representation of body loads.

        Parameters
        ----------
        values : numpy.ndarray
            2d or 3d numpy float array of material densities of shape
            (nE, nNE * nDOF, nRHS) or (nE, nNE * nDOF), where nE, nNE, nDOF and nRHS
            stand for the number of elements, nodes per element, number of degrees
            of freedom and number of load cases respectively.

        Returns
        -------
        numpy.ndarray
            An array of shape (nE, nNE * 6, nRHS).

        Notes
        -----
        The returned array is always 3 dimensional, even if there is only one
        load case.

        See Also
        --------
        :func:`~body_load_vector_bulk`
        """
        values = atleastnd(values, 3, back=True).astype(float)
        # (nE, nNE * nDOF, nRHS) -> (nE, nRHS, nNE * nDOF)
        values = np.swapaxes(values, 1, 2)
        values = ascont(values)
        qpos, qweights = self.quadrature["full"]
        rng = np.array([-1.0, 1.0])
        shp = self.shape_function_values(qpos, rng=rng)
        # (nP, nNE=2, nDOF=6)
        dshp = self.shape_function_derivatives(qpos, rng=rng)
        # (nP, nNE=2, nDOF=6, 3)
        ecoords = self.local_coordinates()
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)
        # (nE, nP, 1, 1)
        djac = self.jacobian(jac=jac)  # (nE, nG)
        gdshp = self.shape_function_derivatives(jac=jac, dshp=dshp)
        # (nE, nP, nNE=2, nDOF=6, 3)
        return body_load_vector_Bernoulli(values, shp, gdshp, djac, qweights)

    def _internal_forces_H_(
        self,
        *_,
        cells: Union[int, Iterable[int]] = None,
        points: Union[float, Iterable] = None,  # [-1, 1]
    ) -> ndarray:
        shp = self.shape_function_values(points, rng=[-1, 1])[cells]
        dshp = self.shape_function_derivatives(points, rng=[-1, 1])[cells]
        ecoords = self.local_coordinates()[cells]
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)
        # jac -> (nE, nP, 1, 1)
        B = self.strain_displacement_matrix(shp=shp, dshp=dshp, jac=jac)
        # B -> (nE, nP, nSTRE, nNODE * 6)

        dofsol = self.dof_solution(flatten=True, cells=cells)
        # dofsol -> (nE, nNE * nDOF, nRHS)
        dofsol = ascont(np.swapaxes(dofsol, 1, 2))
        # dofsol -> (nE, nRHS, nEVAB)
        strains = approx_element_solution_bulk(dofsol, B)
        # strains -> (nE, nRHS, nP, nSTRE)
        strains -= self.kinetic_strains(points=points)[cells]
        D = self.elastic_material_stiffness_matrix()[cells]
        forces = calculate_internal_forces_bulk(strains, D)
        # forces -> (nE, nRHS, nP, nSTRE)
        forces = ascont(np.moveaxis(forces, 1, -1))
        dofsol = ascont(np.swapaxes(dofsol, 1, 2))
        # dofsol -> (nE, nEVAB, nRHS)
        # forces -> (nE, nP, nSTRE, nRHS)
        gdshp = self.shape_function_derivatives(jac=jac, dshp=dshp)
        nE, _, nRHS = dofsol.shape
        nNE, nDOF = self.__class__.NNODE, self.__class__.NDOFN
        dofsol = dofsol.reshape(nE, nNE, nDOF, nRHS)
        D = self.elastic_material_stiffness_matrix()[cells]
        forces = _postproc_bernoulli_internal_forces_H_(dofsol, forces, D, gdshp)
        return forces  # (nE, nP, 6, nRHS)

    def _internal_forces_L_(
        self,
        *_,
        cells: Union[int, Iterable[int]] = None,
        points: Union[float, Iterable] = None,  # [-1, 1]
    ) -> ndarray:
        local_points = np.array(self.lcoords()).flatten()

        shp = self.shape_function_values(local_points, rng=[-1, 1])[cells]
        dshp = self.shape_function_derivatives(local_points, rng=[-1, 1])[cells]
        ecoords = self.local_coordinates()[cells]
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)
        # jac -> (nE, nNE, 1, 1)
        B = self.strain_displacement_matrix(shp=shp, dshp=dshp, jac=jac)
        # B -> (nE, nNE, nSTRE, nNODE * 6)

        dofsol = self.dof_solution(flatten=True, cells=cells)
        # dofsol -> (nE, nNE * nDOF, nRHS)
        dofsol = ascont(np.swapaxes(dofsol, 1, 2))
        # dofsol -> (nE, nRHS, nEVAB)
        strains = approx_element_solution_bulk(dofsol, B)
        # strains -> (nE, nRHS, nNE, nSTRE)
        strains -= self.kinetic_strains(points=local_points)[cells]
        D = self.elastic_material_stiffness_matrix()[cells]

        nE, nRHS, nP = strains.shape[:3]
        forces = np.zeros((nE, nRHS, nP, 6), dtype=float)
        inds = np.array([0, 3, 4, 5], dtype=int)
        calculate_internal_forces_bulk(strains, D, forces, inds)
        # forces -> (nE, nRHS, nNE, nSTRE)
        forces = np.moveaxis(forces, 1, -1)
        # forces -> (nE, nNE, nSTRE, nRHS)

        *_, dshpf = self.Geometry.generate_class_functions(update=False, return_symbolic=False)
        dshp_geom = np.squeeze(dshpf([[i] for i in local_points]))
        # dshp_geom -> (nNE, nNE)
        _postproc_bernoulli_internal_forces_L_(forces, dshp_geom, jac)
        # forces -> (nE, nNE, nSTRE, nRHS)

        if isinstance(points, Iterable):
            approx = interpolate.interp1d(
                local_points, forces, axis=1, assume_sorted=True
            )
            forces = approx(points)

        return ascont(forces)

    def _internal_forces_(self, *args, **kwargs):
        return self._internal_forces_L_(self, *args, **kwargs)
        # return self._internal_forces_H_(self, *args, **kwargs)

    def lumped_mass_matrix(
        self,
        *_,
        lumping: str = "direct",
        alpha: float = 1 / 50,
        frmt: str = "full",
        **__,
    ) -> ndarray:
        """
        Returns the lumped mass matrix of the block.

        Parameters
        ----------
        alpha: float, Optional
            A nonnegative parameter, typically between 0 and 1/50 (see notes).
            Default is 1/20.
        lumping: str, Optional
            Controls lumping. Currently only direct lumping is available.
            Default is 'direct'.
        frmt: str, Optional
            Possible values are 'full' or 'diag'. If 'diag', only the diagonal
            entries are returned, if 'full' a full matrix is returned.
            Default is 'full'.
        """
        dbkey = self.__class__._attr_map_["M"]
        if lumping == "direct":
            dens = self.db.density
            try:
                areas = self.areas()
            except Exception:
                areas = np.ones_like(dens)
            lengths = self.lengths()
            topo = self.topology()
            ediags = dlump(dens, lengths, areas, topo, alpha)
            if frmt == "full":
                N = ediags.shape[-1]
                M = np.zeros((ediags.shape[0], N, N))
                inds = np.arange(N)
                M[:, inds, inds] = ediags
                self.db[dbkey] = M
                return M
            elif frmt == "diag":
                self.db[dbkey] = ediags
                return ediags
        raise RuntimeError("Lumping mode not recognized :(")
