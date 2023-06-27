import numpy as np
from numpy import ndarray
from typing import Union, Iterable

from neumann.utils import to_range_1d

from .solid import Solid
from ...utils.fem.bernoulli import (
    strain_displacement_matrix_bernoulli,
)


class TimoshenkoBeam(Solid):
    dofs = ("UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ")

    NDOFN = 6
    NSTRE = 6

    def strain_displacement_matrix(
        self,
        pcoords: Union[float, Iterable[float]] = None,
        *,
        rng: Iterable = None,
    ) -> ndarray:
        """
        Calculates the strain displacement matrix.

        Parameters
        ----------
        pcoords: float or Iterable, Optional
            Locations of the evaluation points. Default is None.
        rng: Iterable, Optional
            The range in which the locations ought to be understood,
            typically [0, 1] or [-1, 1]. Only if 'pcoords' is provided.
            Default is [0, 1].
        """
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        pcoords = to_range_1d(pcoords, source=rng, target=[-1, 1])
        dshp = self.shape_function_derivatives(pcoords, rng=rng)
        ecoords = self.local_coordinates()
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)
        return strain_displacement_matrix_bernoulli(dshp, jac)

    def masses(self, *_, density: ndarray = None, **__) -> ndarray:
        """
        Returns the masses of the cells in the block.
        """
        if isinstance(density, ndarray):
            dens = density
        else:
            dens = self.density
        lengths = self.lengths()
        return dens * lengths

    def mass(self, *args, **kwargs) -> float:
        """
        Returns the total mass of the block.
        """
        return np.sum(self.masses(*args, **kwargs))
