from .surface import Surface
from ...utils.fem.membrane import (
    strain_displacement_matrix_membrane,
    HMH_membrane,
    material_strains_membrane,
    model_stiffness_iso_homg_membrane,
)


__all__ = ["Membrane", "DrillingMembrane"]


class Membrane(Surface):
    dofs = ("UX", "UY")
    strn = ("exx", "eyy", "exy")

    NDOFN = 2
    NSTRE = 3

    @classmethod
    def model_stiffness_matrix_iso_homg(cls, C, t, *args, **kwargs):
        return model_stiffness_iso_homg_membrane(C, t)

    @classmethod
    def strain_displacement_matrix(cls, *args, dshp=None, jac=None, **kwargs):
        return strain_displacement_matrix_membrane(dshp, jac)

    @classmethod
    def HMH(cls, data, *args, **kwargs):
        return HMH_membrane(data)

    @classmethod
    def material_strains(cls, model_strains, z, t, *args, **kwargs):
        return material_strains_membrane(model_strains, z)


class DrillingMembrane(Membrane):
    dofs = ("UX", "UY", "ROTZ")

    NDOFN = 3
    NSTRE = 3
