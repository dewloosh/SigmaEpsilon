from numpy import ndarray

from .surface import Surface
from ...utils.fem.mindlinplate import (
    strain_displacement_matrix_mindlin_plate,
    HMH_mindlin_plate,
    material_strains_mindlin_plate,
    model_stiffness_iso_homg_mindlin_plate,
)


__all__ = ["MindlinPlate"]


class MindlinPlate(Surface):
    dofs = ("UZ", "ROTX", "ROTY")
    strn = ("kxx", "kyy", "kxy" "exz", "eyz")

    NDOFN = 3
    NSTRE = 5

    @classmethod
    def model_stiffness_matrix_iso_homg(cls, C, t, *_, **__) -> ndarray:
        return model_stiffness_iso_homg_mindlin_plate(C, t)

    @classmethod
    def strain_displacement_matrix(
        cls, *_, shp=None, dshp=None, jac=None, **__
    ) -> ndarray:
        return strain_displacement_matrix_mindlin_plate(shp, dshp, jac)

    @classmethod
    def HMH(cls, data, *_, **__) -> ndarray:
        return HMH_mindlin_plate(data)

    @classmethod
    def material_strains(cls, model_strains, z, t, *_, **__) -> ndarray:
        return material_strains_mindlin_plate(model_strains, z, t)
