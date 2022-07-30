# -*- coding: utf-8 -*-
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from typing import Union, MutableMapping

import numpy as np
from numpy import ndarray

from ..math.array import atleast1d, ascont
from ..math.utils import to_range

from .celldata import CellData
from .utils import jacobian_matrix_bulk, points_of_cells, pcoords_to_coords_1d
from .topo import rewire, TopologyArray


MapLike = Union[ndarray, MutableMapping]


class PolyCell(CellData):

    NNODE = None
    NDIM = None
    vtkCellType = None

    def __init__(self, *args, i: ndarray=None, **kwargs):
        if isinstance(i, ndarray):
            key = self.__class__._attr_map_['id']
            kwargs[key] = i
        super().__init__(*args, **kwargs)

    def measures(self, *args, **kwargs):
        raise NotImplementedError

    def measure(self, *args, **kwargs):
        return np.sum(self.measure(*args, **kwargs))

    def volume(self, *args, **kwargs):
        return np.sum(self.volumes(*args, **kwargs))

    def volumes(self, *args, **kwargs):
        raise NotImplementedError

    def jacobian_matrix(self, *args, dshp=None, ecoords=None, topo=None, **kwargs):
        """
        Returns the jacobian matrix.
        """
        ecoords = self.local_coordinates(
            topo=topo) if ecoords is None else ecoords
        return jacobian_matrix_bulk(dshp, ecoords)

    def jacobian(self, *args, jac=None, **kwargs):
        """
        Returns the jacobian determinant
        """
        return np.linalg.det(jac)

    def points_of_cells(self, *args, target=None, **kwargs):
        """
        Returns the points of the cells.
        """
        assert target is None
        coords = kwargs.get('coords', None)
        if coords is None:
            if self.pointdata is not None:
                coords = self.pointdata.x
            else:
                coords = self.container.source().coords()
        topo = self.topology().to_numpy()
        return points_of_cells(coords, topo)

    def local_coordinates(self, *args, **kwargs):
        """
        Returns local coordinates of the selection.
        """
        frames = kwargs.get('frames', self.frames)
        topo = self.topology().to_numpy()
        coords = kwargs.get('coords', None)
        if coords is None:
            if self.pointdata is not None:
                coords = self.pointdata.x
            else:
                coords = self.container.source().coords()
        return points_of_cells(coords, topo, local_axes=frames)

    def coords(self, *args, **kwargs):
        """
        Returns the coordinates of the cells in the selection.

        """
        return self.points_of_cells(*args, **kwargs)

    def topology(self) -> TopologyArray:
        """
        Returns the numerical representation of the topology of the selection.

        """
        key = self.__class__._attr_map_['nodes']
        if key in self.fields:
            return TopologyArray(self.nodes)
        else:
            return None

    def rewire(self, imap: MapLike = None):
        """
        Rewires the topology of the block according to the mapping
        described by the argument `imap`. The mapping happens the
        following way:

        topology_new[old_index] = imap[topology_old[old_index]] 

        Parameters
        ----------
        imap : MapLike
            Mapping from old to new node indices.

        """
        topo = rewire(self.topology().to_array(), imap)
        self.nodes = topo


class PolyCell1d(PolyCell):

    NDIM = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def lenth(self, *args, **kwargs):
        return np.sum(self.lengths(*args, **kwargs))

    def lengths(self, *args, **kwargs):
        raise NotImplementedError

    def areas(self, *args, **kwargs):
        raise NotImplementedError

    def area(self, *args, **kwargs):
        return np.sum(self.areas(*args, **kwargs))

    def measures(self, *args, **kwargs):
        return self.lengths(*args, **kwargs)

    # NOTE The functionality of `pcoords_to_coords_1d` needs to be generalized
    # for higher order cells.
    def points_of_cells(self, *args, points=None, cells=None, target='global',
                        rng=None, flatten=False, **kwargs):
        if isinstance(target, str):
            assert target.lower() in ['global', 'g']
        else:
            raise NotImplementedError
        topo = kwargs.get('topo', self.topology().to_numpy())
        coords = kwargs.get('coords', None)
        if coords is None:
            if self.pointdata is not None:
                coords = self.pointdata.x
            else:
                coords = self.container.source().coords()
        ecoords = points_of_cells(coords, topo)
        if points is None and cells is None:
            return ecoords

        # points or cells is not None
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id)
            cells = atleast1d(cells[conds])
            if len(cells) == 0:
                return {}
            ecoords = ecoords[cells]
            topo = topo[cells]
        else:
            cells = np.s_[:]

        if points is None:
            points = np.array(self.lcoords()).flatten()
            rng = [-1, 1]
        else:
            rng = np.array([0, 1]) if rng is None else np.array(rng)

        points, rng = to_range(points, source=rng, target=[
                               0, 1]).flatten(), [0, 1]
        datacoords = pcoords_to_coords_1d(points, ecoords)  # (nE * nP, nD)

        if not flatten:
            nE = ecoords.shape[0]
            nP = points.shape[0]
            datacoords = datacoords.reshape(
                nE, nP, datacoords.shape[-1])  # (nE, nP, nD)

        # values : (nE, nP, nDOF, nRHS) or (nE, nP * nDOF, nRHS)
        if isinstance(cells, slice):
            # results are requested on all elements
            data = datacoords
        elif isinstance(cells, Iterable):
            data = {c: datacoords[i] for i, c in enumerate(cells)}
        else:
            raise TypeError(
                "Invalid data type <> for cells.".format(type(cells)))

        return data


class PolyCell2d(PolyCell):

    NDIM = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def area(self, *args, **kwargs):
        return np.sum(self.areas(*args, **kwargs))

    def areas(self, *args, **kwargs):
        raise NotImplementedError

    def measures(self, *args, **kwargs):
        return self.areas(*args, **kwargs)

    def local_coordinates(self, *args, **kwargs):
        ecoords = super(PolyCell2d, self).local_coordinates(*args, **kwargs)
        return ascont(ecoords[:, :, :2])

    def thickness(self, *args, **kwargs) -> ndarray:
        return self._wrapped[self.__class__._attr_map_['t']].to_numpy()


class PolyCell3d(PolyCell):

    NDIM = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def measures(self, *args, **kwargs):
        return self.volumes(*args, **kwargs)
