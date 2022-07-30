# -*- coding: utf-8 -*-
import numpy as np

from .cell import PolyCell3d
from .tet.tetutils import tet_vol_bulk
from .utils import cells_coords
from .topo.tr import H8_to_TET4


class PolyHedron(PolyCell3d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_tetrahedra(self, coords, topo):
        raise NotImplementedError

    def volume(self, *args, coords=None, topo=None, **kwargs):
        if coords is None:
            coords = self.container.root().coords()
        topo = self.nodes if topo is None else topo
        return np.sum(self.volumes(coords, topo))

    def volumes(self, *args, coords=None, topo=None, **kwargs):
        if coords is None:
            coords = self.container.root().coords()
        topo = self.nodes if topo is None else topo
        volumes = tet_vol_bulk(cells_coords(*self.to_tetrahedra(coords, topo)))
        res = np.sum(volumes.reshape(topo.shape[0], int(
            len(volumes) / topo.shape[0])), axis=1)
        return np.squeeze(res)


class TetraHedron(PolyHedron):

    NNODE = 4
    vtkCellType = 10
    __label__ = 'TET4'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_tetrahedra(self, coords, topo):
        return coords, topo


class QuadraticTetraHedron(PolyHedron):

    NNODE = 10
    vtkCellType = 24
    __label__ = 'TET10'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_tetrahedra(self, coords, topo):
        raise NotImplementedError


class HexaHedron(PolyHedron):

    NNODE = 8
    vtkCellType = 12
    __label__ = 'H8'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_tetrahedra(self, coords, topo, data=None):
        return H8_to_TET4(coords, topo, data)


class TriquadraticHexaHedron(PolyHedron):

    NNODE = 27
    vtkCellType = 29
    __label__ = 'H27'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Wedge(PolyHedron):

    NNODE = 6
    vtkCellType = 13
    __label__ = 'W6'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BiquadraticWedge(PolyHedron):

    NNODE = 18
    vtkCellType = 32
    __label__ = 'W18'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)