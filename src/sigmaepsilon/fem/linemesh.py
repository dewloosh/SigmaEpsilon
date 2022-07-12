# -*- coding: utf-8 -*-
import numpy as np

from dewloosh.mesh.config import __hasplotly__, __hasmatplotlib__

from dewloosh.solid.fem import FemMesh

from .cells.bernoulli2 import Bernoulli2
from .cells.bernoulli3 import Bernoulli3


class LineMesh(FemMesh):
    
    _cell_classes_ = {
        2: Bernoulli2,
        3: Bernoulli3,
    }

    def __init__(self, *args, areas=None, connectivity=None, **kwargs):
        
        if connectivity is not None:
            if isinstance(connectivity, np.ndarray):
                assert len(connectivity.shape) == 3
                assert connectivity.shape[0] == nE
                assert connectivity.shape[1] == 2
                assert connectivity.shape[2] == self.__class__.NDOFN          
                
        super().__init__(*args, connectivity=connectivity, **kwargs)
        
        if self.celldata is not None:
            nE = len(self.celldata)
            if areas is None:
                areas = np.ones(nE)
            else:
                assert len(areas.shape) == 1, \
                    "'areas' must be a 1d float or integer numpy array!"
            dbkey = self.celldata.__class__._attr_map_['areas']
            self.celldata.db[dbkey] = areas
                
    def masses(self, *args, **kwargs):
        blocks = self.cellblocks(*args, inclusive=True, **kwargs)
        vmap = map(lambda b: b.celldata.masses(), blocks)
        return np.concatenate(list(vmap))
    
    def mass(self, *args, **kwargs):
        return np.sum(self.masses(*args, **kwargs))
               
    def to_plotly(self):
        assert __hasplotly__
        raise NotImplementedError

    def to_mpl(self):
        """
        Returns the mesh as a `matplotlib` figure.
        """
        assert __hasmatplotlib__
        raise NotImplementedError


class BernoulliFrame(LineMesh):
    
    NDOFN = 6
    