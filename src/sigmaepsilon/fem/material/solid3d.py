from typing import Union

from numpy import ndarray

from neumann.linalg import ReferenceFrame
from polymesh import PolyData

from .solid import Solid
from ...material import ElasticityTensor
from ...utils.fem.solid3d import (
    strain_displacement_matrix_solid3d,
    HMH_3d_bulk_multi_solid3d,
)


__all__ = ["Solid3d"]


class Solid3d(Solid):
    """
    A model for 3d solids.

    See Also
    --------
    :class:`~sigmaepsilon.fem.model.solid.Solid`
    """

    dofs = ("UX", "UY", "UZ")

    NDOFN = 3
    NSTRE = 6

    def strain_displacement_matrix(
        self, pcoords: ndarray = None, *, dshp: ndarray = None, jac: ndarray = None, **_
    ) -> ndarray:
        """
        Calculates the strain displacement matrix for solids.

        Parameters
        ----------
        pcoords: numpy.ndarray, Optional
            Locations of evaluation points. Either 'pcoords' or
            both 'dshp' and 'jac' must be provided.
        dshp: numpy.ndarray, Optional
            Shape function derivatives evaluated at some points.
            Only required if 'pcoords' is not provided.
        jac: numpy.ndarray
            Jacobian matrices evaluated at some points.
            Only required if 'pcoords' is not provided.

        Returns
        -------
        numpy.ndarray
            An array of shape (nE, nP, nSTRE, nTOTV), where nE, nP
            nSTRE and nTOTV are the number of elements, evaulation
            points, stress components and total DOFs per element.
        """
        if dshp is None and pcoords is not None:
            dshp = self.shape_function_derivatives(pcoords)
        if jac is None:
            jac = self.jacobian_matrix(dshp=dshp)
        return strain_displacement_matrix_solid3d(dshp, jac)

    @classmethod
    def HMH(cls, stresses: ndarray) -> ndarray:
        """
        Evaluates the Huber-Mises-Hencky stress at multiple points
        of multiple cells.

        Parameters
        ----------
        stresses: numpy.ndarray, Optional
            Array of shape (nE, nP, nSTRE), where nE, nP and nSTRE
            are the number of elements, evaulation points and stress
            components. The stresses are expected in the order
            s11, s22, s33, s23, s13, s12.

        Returns
        -------
        numpy.ndarray
            An array of shape (nE, nP), where nE and nP are the
            number of elements and evaulation points.
        """
        return HMH_3d_bulk_multi_solid3d(stresses)

    def elastic_material_stiffness_matrix(
        self,
        target: Union[str, ReferenceFrame] = "local",
    ) -> ndarray:
        if isinstance(target, str) and target == "local":
            return self.material_stiffness

        if isinstance(target, str):
            assert target == "global"
            container: PolyData = self.container
            target = container.source().frame
        else:
            if not isinstance(target, ReferenceFrame):
                raise TypeError("'target' should be an instance of ReferenceFrame")

        source = ReferenceFrame(self.frames)
        tensor = ElasticityTensor(
            self.material_stiffness, frame=source, tensorial=False
        )
        return tensor.contracted_components(target=target)
