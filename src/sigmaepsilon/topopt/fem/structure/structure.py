# -*- coding: utf-8 -*-
from copy import deepcopy

from sigmaepsilon.solid.fem.structure import Structure as SolidStructure

from ...oc.SIMP_OC_FEM import OC_SIMP_COMP as OC

__all__ = ['Structure']


class Structure(SolidStructure):
    
    def minimize_volume(self, inplace=True):
        pass
    
    def optimize_topology(self, *args, maxiter=300, **kwargs):
        oc = self.optimizer(*args, maxiter=maxiter,**kwargs)
        _solution = next(oc)
        yield _solution
        solution = deepcopy(_solution)
        try:
            for _ in range(maxiter):
                _solution = next(oc)
                yield _solution
                solution = deepcopy(_solution)
        except:
            return solution
        return solution
    
    def optimizer(self, *args, r_max=None, neighbours=None, **kwargs):
        if r_max is not None and neighbours is None:
            neighbours = self.mesh.cells_around_cells(r_max, 'dict')
        return OC(self, *args, neighbours=neighbours, **kwargs)