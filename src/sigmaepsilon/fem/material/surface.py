from neumann import atleast2d, atleastnd

from ...utils.material import stresses_from_strains
from ...utils.fem.fem import topo_to_gnum
from .solid import Solid


class Surface(Solid):
    strn = ("exx", "eyy", "exy", "kxx", "kyy", "kxy", "exz", "eyz")

    def model_stiffness_matrix(self, *args, **kwargs):
        C = self.material_stiffness_matrix()
        t = self.thickness()
        return self.model_stiffness_matrix_iso_homg(C, t)

    @classmethod
    def model_stiffness_matrix_iso_homg(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def material_strains(cls, *args, **kwargs):
        raise NotImplementedError

    def ABDS(self, *args, **kwargs):
        return self.model_stiffness_matrix(*args, **kwargs)

    def strains_at(self, lcoords, *args, z=None, topo=None, **kwargs):
        if topo is None:
            topo = self.topology().to_numpy()
        lcoords = atleast2d(lcoords)
        dshp = self.shape_function_derivatives(lcoords)
        ecoords = self.local_coordinates(topo=topo)
        jac = self.jacobian_matrix(dshp, ecoords)
        gnum = topo_to_gnum(topo, self.NDOFN)

        dofsol1d = self.pointdata.dofsol.to_numpy().flatten()

        dofsol = self.pointdata.dofsol
        dofsol = atleastnd(dofsol, 3, back=True)
        nP, nDOF, nRHS = dofsol.shape
        dofsol = dofsol.reshape(nP * nDOF, nRHS)

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
        if "HMH" in args:
            return self.HMH(stresses_from_strains(C, strns))
        return stresses_from_strains(C, strns)
