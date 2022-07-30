# -*- coding: utf-8 -*-
import numpy as np

from .utils import cells_coords
from .tri.triutils import area_tri_bulk
from .topo.tr import T6_to_T3, Q4_to_T3, Q9_to_Q4, T3_to_T6
from .cell import PolyCell2d


class PolyGon(PolyCell2d):
    """
    Base class for polygons.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_TriMesh(cls, *args, coords=None, topo=None, **kwargs):
        raise NotImplementedError

    def to_triangles(self):
        raise NotImplementedError

    def area(self, *args, coords=None, topo=None, **kwargs):
        if coords is None:
            coords = self.container.root().coords()
        topo = self.nodes if topo is None else topo
        return np.sum(self.areas(coords=coords, topo=topo))

    def areas(self, *args, coords=None, topo=None, **kwargs):
        if coords is None:
            coords = self.container.root().coords()
        topo = self.nodes if topo is None else topo
        topo_tri = self.to_triangles()
        areas = area_tri_bulk(cells_coords(coords, topo_tri))
        res = np.sum(areas.reshape(topo.shape[0], int(
            len(areas)/topo.shape[0])), axis=1)
        return np.squeeze(res)

    def volumes(self, *args, **kwargs):
        dbkey = self.__class__._attr_map_['t']
        areas = self.areas(*args, **kwargs)
        if dbkey in self.fields:
            t = self.db[dbkey].to_numpy()
            return areas * t
        else:
            return areas


class Triangle(PolyGon):
    """
    Base class for triangles.

    """

    NNODE = 3
    vtkCellType = 5
    __label__ = 'T3'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_triangles(self):
        return self.nodes

    @classmethod
    def from_TriMesh(cls, *args, coords=None, topo=None, **kwargs):
        from dewloosh.mesh.tri.trimesh import TriMesh
        if len(args) > 0 and isinstance(args[0], TriMesh):
            return TriMesh.coords(), TriMesh.topology()
        elif coords is not None and topo is not None:
            return coords, topo
        else:
            raise NotImplementedError

    def areas(self, *args, coords=None, topo=None, **kwargs):
        if coords is None:
            coords = self.container.root().coords()
        topo = self.nodes if topo is None else topo
        return area_tri_bulk(cells_coords(coords, topo))


class QuadraticTriangle(PolyGon):

    NNODE = 6
    vtkCellType = 22
    __label__ = 'T6'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_triangles(self):
        return T6_to_T3(None, self.nodes)[1]

    @classmethod
    def from_TriMesh(cls, *args, coords=None, topo=None, **kwargs):
        from dewloosh.mesh.tri.trimesh import TriMesh
        if len(args) > 0 and isinstance(args[0], TriMesh):
            return T3_to_T6(TriMesh.coords(), TriMesh.topology())
        elif coords is not None and topo is not None:
            return T3_to_T6(coords, topo)
        else:
            raise NotImplementedError


class Quadrilateral(PolyGon):

    NNODE = 4
    vtkCellType = 9
    __label__ = 'Q4'

    def to_triangles(self):
        return Q4_to_T3(None, self.nodes)[1]


class BiQuadraticQuadrilateral(PolyGon):

    NNODE = 9
    vtkCellType = 28
    __label__ = 'Q9'

    def to_triangles(self):
        return Q4_to_T3(*Q9_to_Q4(None, self.nodes))[1]
