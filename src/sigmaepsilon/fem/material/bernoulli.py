import numpy as np
from numpy import ndarray
from typing import Union, Iterable

from .solid import Solid
from ...utils.fem.bernoulli import (
    strain_displacement_matrix_bernoulli,
)


class BernoulliBeam(Solid):
    dofs = ("UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ")

    NDOFN = 6
    NSTRE = 4

    def strain_displacement_matrix(
        self,
        pcoords: Union[float, Iterable[float]] = None,
        *_,
        rng: Iterable = None,
        jac: ndarray = None,
        dshp: ndarray = None,
        **__
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
        jac: Iterable, Optional
            The jacobian matrix as a float array of shape (nE, nP, 1, 1), evaluated for
            each point in each cell. Default is None.
        dshp: Iterable, Optional
            The shape function derivatives of the master element as a float array of
            shape (nP, nNE, nDOF=6, 3), evaluated at a 'nP' number of points.
            Default is None.
        """
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        gdshp = self.shape_function_derivatives(pcoords, rng=rng, jac=jac, dshp=dshp)
        return strain_displacement_matrix_bernoulli(gdshp)

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
