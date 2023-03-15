from typing import Callable

import numpy as np
from numpy import ndarray

from neumann import atleastnd
from polymesh.pointdata import PointData as MeshPointData


class PointData(MeshPointData):
    """
    A subclass of :class:`polymesh.pointdata.PointData` to handle data related
    to the pointcloud of a finite element mesh.

    Parameters
    ----------
    fixity : numpy.ndarray, Optional
        A 2d boolean or float array describing essential boundary conditions.
        Default is None.
    loads : numpy.ndarray, Optional
        A 2d or 3d float array of nodal loads.
        Default is None.
    mass : numpy.ndarray, Optional
        2d array of nodal masses for each degree of freedom of a node.
        Default is None.
    fields : dict, Optional
        Every value of this dictionary is added to the dataset. Default is `None`.
    **kwargs : dict, Optional
        For every key and value pair where the value is a numpy array
        with a matching shape (has entries for all points), the key
        is considered as a field and the value is added to the database.

    Examples
    --------
    We will define the point-related data of a simple linear console. At the
    moment this means the specification of cordinates, nodal loads, and boundary
    conditions. The common properties of these attributes is that each of these
    can be best described by arrays having the same length as the pointcloud itself.
    Import the necessary stuff:

    >>> from sigmaepsilon import Structure, LineMesh, PointData
    >>> from neumann.linalg import linspace, Vector
    >>> from polymesh.space import StandardFrame, PointCloud, frames_of_lines
    >>> import numpy as np

    Define a coordinate frame, and a coordinate array,

    >>> GlobalFrame = StandardFrame(dim=3)
    >>> nElem = 20  # number of finite elements to use
    >>> p0 = np.array([0., 0., 0.])
    >>> p1 = np.array([L, 0., 0.])
    >>> coords = linspace(p0, p1, nElem+1)
    >>> coords = PointCloud(coords, frame=GlobalFrame).show()

    two numpy arrays resembling the boundary conditions,

    >>> # support at the leftmost, load at the rightmost node
    >>> loads = np.zeros((coords.shape[0], 6))
    >>> fixity = np.zeros((coords.shape[0], 6)).astype(bool)
    >>> global_load_vector = Vector([0., 0, F], frame=GlobalFrame).show()
    >>> loads[-1, :3] = global_load_vector
    >>> fixity[0, :] = True

    and finally we can define our dataset.

    >>> pd = PointData(coords=coords, frame=GlobalFrame,
    >>>                loads=loads, fixity=fixity)

    See also
    --------
    :class:`sigmaepsilon.fem.cells.celldata.CellData`
    :class:`polymesh.PolyData`
    :class:`polymesh.CellData`
    :class:`awkward.Record`
    """

    _attr_map_ = {
        "loads": "loads",
        "mass": "mass",
        "fixity": "fixity",
        "dofsol": "dofsol",
        "shapes": "_shapes_",
        "reactions": "reactions",
        "forces": "forces",
    }
    NDOFN = 6

    def __init__(
        self,
        *args,
        fixity: ndarray = None,
        loads: ndarray = None,
        mass: ndarray = None,
        fields: dict = None,
        **kwargs
    ):
        amap = self.__class__._attr_map_

        if fields is not None:
            fixity = fields.pop(amap["fixity"], fixity)
            loads = fields.pop(amap["loads"], loads)
            mass = fields.pop(amap["mass"], mass)

        super().__init__(*args, fields=fields, **kwargs)

        if self.db is not None:
            nP = len(self)
            NDOFN = self.__class__.NDOFN

            if fixity is None:
                fixity = np.zeros((nP, NDOFN), dtype=bool)
            else:
                assert isinstance(fixity, np.ndarray) and len(fixity.shape) == 2
            self.fixity = fixity

            if loads is None:
                loads = np.zeros((nP, NDOFN, 1))
            if loads is not None:
                assert isinstance(loads, np.ndarray)
                if loads.shape[0] == nP:
                    self.loads = loads
                elif loads.shape[0] == nP * NDOFN:
                    loads = atleastnd(loads, 2, back=True)
                    self.loads = loads.reshape(nP, NDOFN, loads.shape[-1])

            if mass is not None:
                if isinstance(mass, float) or isinstance(mass, int):
                    raise NotImplementedError
                elif isinstance(mass, np.ndarray):
                    self.mass = mass

    @property
    def _dbkey_fixity_(self) -> str:
        return self.__class__._attr_map_["fixity"]

    @property
    def has_fixity(self):
        return self._dbkey_fixity_ in self._wrapped.fields

    @property
    def fixity(self) -> ndarray:
        """Returns fixity information as a 2d numpy array."""
        return self._wrapped[self.__class__._attr_map_["fixity"]].to_numpy()

    @fixity.setter
    def fixity(self, value: ndarray):
        """Sets essential boundary conditions."""
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_["fixity"]] = value

    @property
    def loads(self) -> ndarray:
        """Returns the nodal load vector."""
        return self._wrapped[self.__class__._attr_map_["loads"]].to_numpy()

    @loads.setter
    def loads(self, value: ndarray):
        """Sets the nodal load vector."""
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_["loads"]] = value

    @property
    def mass(self) -> ndarray:
        """Retruns nodal masses."""
        return self._wrapped[self.__class__._attr_map_["mass"]].to_numpy()

    @mass.setter
    def mass(self, value: ndarray):
        """Sets nodal masses."""
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_["mass"]] = value

    @property
    def dofsol(self) -> ndarray:
        """Returns the displacement vector."""
        return self._wrapped[self.__class__._attr_map_["dofsol"]].to_numpy()

    @dofsol.setter
    def dofsol(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_["dofsol"]] = value

    @property
    def vshapes(self) -> ndarray:
        """Returns the modal shapes."""
        return self._wrapped[self.__class__._attr_map_["shapes"]].to_numpy()

    @vshapes.setter
    def vshapes(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_["shapes"]] = value

    @property
    def reactions(self) -> ndarray:
        """Returns the vector of reaction forces."""
        return self._wrapped[self.__class__._attr_map_["reactions"]].to_numpy()

    @reactions.setter
    def reactions(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_["reactions"]] = value

    @property
    def forces(self) -> ndarray:
        return self._wrapped[self.__class__._attr_map_["forces"]].to_numpy()

    @forces.setter
    def forces(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_["forces"]] = value


def flatten_pd(default=True):
    def decorator(fnc: Callable):
        def inner(*args, flatten: bool = default, **kwargs):
            if flatten:
                x = fnc(*args, **kwargs)
                if len(x.shape) == 2:
                    return x.flatten()
                else:
                    nN, nDOFN, nRHS = x.shape
                    return x.reshape((nN * nDOFN, nRHS))
            else:
                return fnc(*args, **kwargs)

        inner.__doc__ = fnc.__doc__
        return inner

    return decorator
