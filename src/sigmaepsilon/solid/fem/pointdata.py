# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray

from ...math.array import atleastnd

from ...mesh.pointdata import PointData as MeshPointData


class PointData(MeshPointData):

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
    