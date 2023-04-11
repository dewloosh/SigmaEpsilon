from typing import Iterable, Union

import numpy as np
from numpy import ndarray

from .ebc import _link_opposite_sides_sym, _link_opposite_sides
from .utils import (
    _strain_field_3d_bulk,
    _postproc_3d_gauss_stresses,
    _calc_avg_hooke_3d_to_shell,
)
from sigmaepsilon.fem import Structure
from sigmaepsilon.fem.ebc import NodeToNode


class RepresentativeVolumeElement(Structure):
    
    def __init__(self, *args, symmetric:bool=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.symmetric = symmetric
    
    def _periodic_essential_ebc(
        self, axis: Union[int, Iterable[int]] = 0,
    ) -> NodeToNode:
        if self.symmetric:
            return _link_opposite_sides_sym(self.mesh.points(), axis)
        else:
            return _link_opposite_sides(self.mesh, axis)
        
    def ABD(self) -> ndarray:
        """
        Returns the elasticity matrix for a Kirchhoff-Love plate.
        
        Returns
        -------
        numpy.ndarray
            The 6x6 elastic stiffness matrix.
        """
        return self.homogenize(target="KL")

    def ABDS(self) -> ndarray:
        """
        Returns the elasticity matrix for a Mindlin-Reissner plate.
        
        Returns
        -------
        numpy.ndarray
            The 8x8 elastic stiffness matrix.
        """
        return self.homogenize(target="MR")

    def homogenize(self, target: str = "MR") -> ndarray:
        """
        Returns the homogenized elasticity matrix for a Mindlin-Reissner or a
        Kirchhoff-Love plate.
        
        Parameters
        ----------
        target: str, Optional
            The target model. Accepted values are 'MR' for Mindlin-Reissner
            shells and 'KL' for Kirchhoff-Love shells.
        
        Returns
        -------
        numpy.ndarray
            The elastic stiffness matrix.
        """
        target = target.lower()
        if target in ["mr", "kl"]:
            return self._to_shell(target)
    
    def _to_shell(self, target: str = "MR") -> ndarray:
        mesh = self.mesh
        NDOFN = mesh.NDOFN

        if target.lower() == "mr":
            NSTRE = 8
        elif target.lower() == "kl":
            NSTRE = 6

        # mesh data
        points = mesh.points()
        [(xmin, xmax), (ymin, ymax), (_, zmax)] = points.bounds()
        area = (xmax - xmin) * (ymax - ymin)

        # nodal loads
        nodal_loads = np.zeros((len(points), NDOFN, NSTRE), dtype=float)
        mesh.pd.loads = nodal_loads

        # nodal supports against rigid body motion
        nodal_fixity = np.zeros((len(points), NDOFN), dtype=bool)
        i = points.index_of_closest([xmin, ymax, zmax])
        nodal_fixity[i, :3] = True
        i = points.index_of_closest([xmax, ymax, zmax])
        nodal_fixity[i, 2] = True
        i = points.index_of_closest([xmin, ymin, zmax])
        nodal_fixity[i, 2] = True
        i = points.index_of_closest([xmax, ymin, zmax])
        nodal_fixity[i, 1] = True
        mesh.pd.fixity = nodal_fixity

        # nodal supports to guarantee peridic displacement solution
        periodicity_constraints = self._periodic_essential_ebc(axis=[0, 1])

        # initial strain loads
        blocks = list(mesh.cellblocks(inclusive=True))
        for block in blocks:
            centers = block.cd.centers()
            strain_loads = np.zeros((centers.shape[0], 6, NSTRE))
            _strain_field_3d_bulk(centers, out=strain_loads, NSTRE=NSTRE)
            if block.cd.NDIM == 3:
                block.cd.strain_loads = strain_loads
            elif block.cd.NDIM == 2:
                raise NotImplementedError
            else:
                raise NotImplementedError

        # solve BVP
        self.constraints.append(periodicity_constraints)
        self.linear_static_analysis()

        # postproc
        hooke = np.zeros((NSTRE, NSTRE), dtype=float)
        hooke_avg = np.zeros_like(hooke)
        for block in blocks:
            block.cd.strain_loads = None
            hooke_block = block.cd.elastic_material_stiffness_matrix()
            gp, gw = block.cd.quadrature["full"]
            if block.cd.NDIM == 3:
                cell_forces = block.cd.internal_forces(points=gp, flatten=False)
                # cell_forces = block.internal_forces(points=gp, flatten=False, target='global')
                ec = block.cd.coords()
                dshp = block.cd.shape_function_derivatives(gp)
                jac = block.cd.jacobian_matrix(dshp=dshp, ecoords=ec)
                djac = block.cd.jacobian(jac=jac)
                ec = block.cd.coords(points=gp)
                _postproc_3d_gauss_stresses(cell_forces, ec, gw, djac, hooke)
                _calc_avg_hooke_3d_to_shell(hooke_block, ec, gw, djac, hooke_avg)
            elif block.cd.NDIM == 2:
                raise NotImplementedError
            else:
                raise NotImplementedError

        self.constraints.pop(-1)

        hooke /= area
        hooke_avg /= area
        result = hooke_avg - hooke
        if NSTRE == 8:
            result[6:8, 6:8] = hooke[6:8, 6:8] * 5 / 6
        result = (result + result.T) / 2
        return result
