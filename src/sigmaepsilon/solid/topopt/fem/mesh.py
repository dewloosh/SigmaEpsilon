import numpy as np

from neumann import squeeze
from neumann.linalg.sparse.csr import csr_matrix as csr

from sigmaepsilon.solid.fem.mesh import FemMesh as Mesh

__all__ = ["FemMesh"]


class FemMesh(Mesh):
    ...


class FuckedUpFemMesh(Mesh):
    ...

    @squeeze(True)
    def load_vector(self, *args, **kwargs):
        raise NotImplementedError
        # pointdata
        nodal_loads = self.root().pointdata.loads.to_numpy()
        nodal_loads = atleast3d(nodal_loads, back=True)  # (nP, nDOF, nRHS)
        f_p = fem_load_vector(values=nodal_loads, squeeze=False)

        # celldata
        # blocks = self.cellblocks(inclusive=True)
        # def foo(b): return b.celldata.body_load_vector(squeeze=False, shape=shp)
        # f_c = np.sum(list(map(foo, blocks)))
        """
        if not self.is_compatible():
            # distribute nodal loads
            blocks = list(self.cellblocks(inclusive=True))
            def foo(b): return b.celldata.distribute_nodal_data(loads, 'loads')
            list(map(foo, blocks))
            # collect nodal loads
            N = len(loads)
            def foo(b): return b.celldata.collect_nodal_data('loads', N=N)
            loads = np.sum(list(map(foo, blocks)), axis=0)
        """
        return f_p

    def penalty_matrix_coo(
        self,
        *args,
        eliminate_zeros=True,
        sum_duplicates=True,
        ensure_comp=False,
        distribute=False,
        **kwargs
    ):
        raise NotImplementedError
        """A penalty matrix that enforces essential(Dirichlet) 
        boundary conditions. Returns a scipy sparse matrix in coo format."""

        # essential boundary conditions
        fixity = self.root().pointdata.fixity.to_numpy()
        K_coo = fem_penalty_matrix_coo(
            values=fixity,
            eliminate_zeros=eliminate_zeros,
            sum_duplicates=sum_duplicates,
        )

        if distribute:
            # distribute nodal fixity
            fixity = self.root().pointdata.fixity.to_numpy().astype(int)
            blocks = list(self.cellblocks(inclusive=True))

            def foo(b):
                return b.celldata.distribute_nodal_data(fixity, "fixity")

            list(map(foo, blocks))

            # assemble
            def foo(b):
                return b.celldata.penalty_matrix_coo()

            K_coo = np.sum(list(map(foo, blocks))).tocoo()

        if ensure_comp and not self.is_compatible():
            # penalty arising from incompatibility
            p = kwargs.get("compatibility_penalty", None)
            if p is not None:
                K_coo += self.compatibility_penalty_matrix_coo(
                    eliminate_zeros=False, p=p
                )

        return K_coo.tocoo()

    def compatibility_penalty_matrix_coo(
        self, *args, eliminate_zeros=True, p=1e12, **kwargs
    ):
        blocks = self.cellblocks(inclusive=True)
        nam_csr_tot = csr(self.nodal_approximation_matrix_coo(eliminate_zeros=False))

        def foo(b):
            return b.celldata.compatibility_penalty_matrix_coo(
                nam_csr_tot=nam_csr_tot, p=p
            )

        res = np.sum(list(map(foo, blocks))).tocoo()
        if eliminate_zeros:
            res.eliminate_zeros()
        return res

    def is_compatible(self):
        blocks = self.cellblocks(inclusive=True)

        def fltr(b):
            return not b.celldata.compatible

        return len(list(filter(fltr, blocks))) == 0


if __name__ == "__main__":
    pass
