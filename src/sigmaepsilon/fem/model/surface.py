# -*- coding: utf-8 -*-
from dewloosh.math.array import atleast2d

from ..utils import topo_to_gnum
from .solid import Solid
from .utils import stresses_from_strains


class Surface(Solid):

    def model_stiffness_matrix(self, *args, **kwargs):
        C = self.material_stiffness_matrix()
        t = self.thickness()
        return self.model_stiffness_matrix_iso_homg(C, t)

    @classmethod
    def model_stiffness_matrix_iso_homg(cls, *args,  **kwargs):
        raise NotImplementedError

    @classmethod
    def material_strains(cls, *args, **kwargs):
        raise NotImplementedError

    def ABDS(self, *args, **kwargs):
        return self.model_stiffness_matrix(*args, **kwargs)

    def strains_at(self, lcoords, *args,  z=None, topo=None, **kwargs):
        if topo is None:
            topo = self.nodes.to_numpy()
        lcoords = atleast2d(lcoords)
        dshp = self.shape_function_derivatives(lcoords)
        ecoords = self.local_coordinates(topo=topo)
        jac = self.jacobian_matrix(dshp, ecoords)
        gnum = topo_to_gnum(topo, self.NDOFN)
        dofsol1d = self.pointdata.dofsol.to_numpy().flatten()
        B = self.strain_displacement_matrix(dshp, jac)
        if z is None:
            # return generalized model strains
            return self.model_strains(dofsol1d, gnum, B)
        else:
            # returns material strains
            t = self.thickness()
            model_strains = self.model_strains(dofsol1d, gnum, B)
            return self.material_strains(model_strains, z, t)

    def stresses_at(self, *args, z=None, topo=None, **kwargs):
        """
        Returns stresses for every node of every element.
        """
        # The next code line returns either generalized strains
        # or material strains, depending on the value of z
        strns = self.strains_at(*args, z=z, topo=topo, **kwargs)
        if z is None:
            C = self.model_stiffness_matrix()
        else:
            C = self.material_stiffness_matrix()
        if 'HMH' in args:
            return self.HMH(stresses_from_strains(C, strns))
        return stresses_from_strains(C, strns)
