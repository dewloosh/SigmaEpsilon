from numpy import ndarray

from .surface import Surface
from ...utils.fem.mindlinshell import (
    strain_displacement_matrix_mindlin_shell,
    HMH_mindlin_shell,
    material_strains_mindlin_shell,
    model_stiffness_iso_homg_mindlin_shell,
)


__all__ = ["MindlinShell"]


class MindlinShell(Surface):
    dofs = ("UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ")

    NDOFN = 6
    NSTRE = 8

    @classmethod
    def model_stiffness_matrix_iso_homg(cls, C, t, *_, **__) -> ndarray:
        return model_stiffness_iso_homg_mindlin_shell(C, t)

    @classmethod
    def strain_displacement_matrix(
        cls, *_, shp=None, dshp=None, jac=None, **__
    ) -> ndarray:
        return strain_displacement_matrix_mindlin_shell(shp, dshp, jac)

    @classmethod
    def HMH(cls, data, *_, **__) -> ndarray:
        return HMH_mindlin_shell(data)

    @classmethod
    def material_strains(cls, model_strains, z, t, *_, **__) -> ndarray:
        return material_strains_mindlin_shell(model_strains, z, t)
