# -*- coding: utf-8 -*-
from abc import abstractmethod

from dewloosh.math.array import atleast2d

from .utils import model_strains, stresses_from_strains
from ..utils import topo_to_gnum
from ..meta import FemModel


class Solid(FemModel):

    NDOFN = 3
    NSTRE = 6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def material_stiffness_matrix(self, *args, **kwargs):
        return self._wrapped.mat.to_numpy()

    def model_stiffness_matrices(self, *args, **kwargs):
        return self.material_stiffness_matrices()

    @classmethod
    def strains_at(cls, *args,  **kwargs):
        raise NotImplementedError

    @classmethod
    def HMH(cls, *args,  **kwargs):
        raise NotImplementedError

    @classmethod
    def model_strains(cls, dofsol1d, gnum, B, *args, **kwargs):
        return model_strains(dofsol1d, gnum, B)

    def strains_at(self, lcoords, *args,  z=None, topo=None, **kwargs):
        topo = self.topology() if topo is None else topo
        lcoords = atleast2d(lcoords)
        dshp = self.shape_function_derivatives(lcoords)
        ecoords = self.local_coordinates(topo=topo)
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)
        gnum = topo_to_gnum(topo, self.NDOFN)
        dofsol1d = self.pointdata.dofsol.flatten()
        B = self.strain_displacement_matrix(dshp=dshp, jac=jac)
        return self.model_strains(dofsol1d, gnum, B)

    def stresses_at(self, *args, z=None, topo=None, **kwargs):
        """
        Returns stresses for every node of every element.
        """
        # The next code line returns either generalized strains
        # or material strains, depending on the value of z
        strns = self.strains_at(*args, z=z, topo=topo, **kwargs)
        C = self.model_stiffness_matrix()
        if 'HMH' in args:
            return self.HMH(stresses_from_strains(C, strns))
        return stresses_from_strains(C, strns)

    def stresses_at_nodes(self, *args, **kwargs):
        """
        Returns stresses for every node of every element.
        """
        return self.stresses_at(self.lcoords(), *args, **kwargs)

    def stresses_at_centers(self, *args, **kwargs):
        """
        Returns stresses at the centre of every element.
        """
        return self.stresses_at(self.lcenter(), *args, **kwargs)
