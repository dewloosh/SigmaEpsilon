# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray

from neumann.array import atleastnd

from polymesh.pointdata import PointData as MeshPointData


class PointData(MeshPointData):
    """
    A class to handle data related to the pointcloud of a finite element mesh.

    Technicall this is a wrapper around an :class:`awkward.Record` instance.

    Parameters
    ----------
    fields : `dict`, Optional
        Every value of this dictionary is added to the dataset. Default is `None`.
        
    Examples
    --------
    
    We will define the point-related data of a simple linear console. At the
    moment this means the specification of cordinates, nodal loads, and boundary 
    conditions. The common properties of these attributes is that each of these
    can be best described by arrays having the same length as the pointcloud itself.
    Import the necessary stuff:
    
    >>> from sigmaepsilon.solid import Structure, LineMesh, PointData
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
    :class:`polymesh.PolyData`
    :class:`polymesh.CellData`
    :class:`awkward.Record`
    
    """

    _attr_map_ = {
        'loads': 'loads',
        'mass': 'mass',
        'fixity': 'fixity',
        'dofsol' : 'dofsol',
        'reactions' : 'reactions',
        'forces' : 'forces',
    }
    NDOFN = 6

    def __init__(self, *args, fixity=None, loads=None, mass=None,
                 fields=None, **kwargs):
        amap = self.__class__._attr_map_

        if fields is not None:
            fixity = fields.pop(amap['fixity'], fixity)
            loads = fields.pop(amap['loads'], loads)
            mass = fields.pop(amap['mass'], mass)

        super().__init__(*args, fields=fields, **kwargs)

        if self.db is not None:
            nP = len(self)
            NDOFN = self.__class__.NDOFN

            if fixity is None:
                fixity = np.zeros((nP, NDOFN), dtype=bool)
            else:
                assert isinstance(fixity, np.ndarray) and len(
                    fixity.shape) == 2
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
    def fixity(self) -> ndarray:
        return self._wrapped[self.__class__._attr_map_['fixity']].to_numpy()

    @fixity.setter
    def fixity(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['fixity']] = value

    @property
    def loads(self) -> ndarray:
        return self._wrapped[self.__class__._attr_map_['loads']].to_numpy()

    @loads.setter
    def loads(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['loads']] = value
        
    @property
    def mass(self) -> ndarray:
        return self._wrapped[self.__class__._attr_map_['mass']].to_numpy()

    @mass.setter
    def mass(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['mass']] = value
        
    @property
    def dofsol(self) -> ndarray:
        return self._wrapped[self.__class__._attr_map_['dofsol']].to_numpy()

    @dofsol.setter
    def dofsol(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['dofsol']] = value 
        
    @property
    def reactions(self) -> ndarray:
        return self._wrapped[self.__class__._attr_map_['reactions']].to_numpy()

    @reactions.setter
    def reactions(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['reactions']] = value   
        
    @property
    def forces(self) -> ndarray:
        return self._wrapped[self.__class__._attr_map_['forces']].to_numpy()

    @forces.setter
    def forces(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['forces']] = value    
    