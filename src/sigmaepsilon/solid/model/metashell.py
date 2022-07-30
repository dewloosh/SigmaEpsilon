# -*- coding: utf-8 -*-
import numpy as np
from abc import abstractmethod

from ...core import DeepDict
from ...core.tools.kwargtools import getasany, allinkwargs, anyinkwargs
from ...core.tools.dtk import parsedicts_addr


class MetaSurface(DeepDict):
    """
    Base object implementing methods that both a folder (a shell) and a
    file (a layer) can posess.
    """
    
    __layerclass__ = None
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def angle(self):
        _angle = self.get('angle', None)
        if _angle is None:
            return None if self.is_root() else self.parent.angle
        else:
            return _angle

    @angle.setter
    def angle(self, value):
        if self.__layerclass__ is None:
            self['angle'] = value
        else:
            for layer in self.containers(dtype=self.__layerclass__):
                layer['angle'] = value
            self['angle'] = 0
            del self['angle']
            
    @property
    def hooke(self):
        _hooke = self.get('hooke', None)
        if _hooke is None:
            return None if self.is_root() else self.parent.hooke
        else:
            return _hooke

    @hooke.setter
    def hooke(self, value):
        if self.__layerclass__ is None:
            self['hooke'] = value
        else:
            for layer in self.containers(dtype=self.__layerclass__):
                layer['hooke'] = value
            self['hooke'] = 0
            del self['hooke']
    
    @property
    def t(self):
        return self.get('thickness', None)
    
    @t.setter
    def t(self, value):
        if self.__layerclass__ is None:
            self['thickness'] = value
        else:
            raise RuntimeError
        

class Layer(MetaSurface):
    """
    Helper class for early binding.
    """

    __loc__ = [-1., 0., 1.]
    __shape__ = (8, 8)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hooke = getasany(['material', 'm', 'hooke'], **kwargs)
        # set thickness
        self.tmin = None
        self.tmax = None
        self.t = None
        if allinkwargs(['tmin', 'tmax'], **kwargs):
            self.tmin = kwargs.get('tmin', None)
            self.tmax = kwargs.get('tmax', None)
            self.t = self.tmax-self.tmin
        elif anyinkwargs(['t', 'thickness'], **kwargs):
            self.t = getasany(['t', 'thickness'], **kwargs)
            if 'tmin' in kwargs:
                self.tmin = kwargs['tmin']
                self.tmax = self.tmin + self.t
            elif 'tmax' in kwargs:
                self.tmax = kwargs['tmax']
                self.tmin = self.tmax - self.t
            else:
                self.tmin = (-1) * self.t / 2
                self.tmax = self.t / 2
                
    def loc_to_z(self, loc):
        """
        Returns height of a local point by linear interpolation.
        Local coordinate is expected between -1 and 1.
        """
        return 0.5 * ((self.tmax + self.tmin) + loc * (self.tmax - self.tmin))

    @abstractmethod
    def stiffness_matrix(self):
        raise NotImplementedError
    
    
class Surface(MetaSurface):

    __layerclass__ = Layer

    def Layer(self, *args, **kwargs):
        return self.__layerclass__(*args, **kwargs)
    
    def Hooke(self):
        raise NotImplementedError

    def layers(self):
        return [layer for layer in self.containers(dtype=self.__layerclass__)]

    def iterlayers(self):
        return self.containers(dtype=self.__layerclass__)

    def stiffness_matrix(self):
        self._set_layers()
        res = np.zeros(self.__layerclass__.__shape__)
        for layer in self.iterlayers():
            res += layer.stiffness_matrix()
        return res

    def _set_layers(self):
        """
        Sets thickness ranges for the layers.
        """
        layers = self.layers()
        t = sum([layer.t for layer in layers])
        layers[0].tmin = -t/2
        nLayers = len(layers)
        for i in range(nLayers-1):
            layers[i].tmax = layers[i].tmin + layers[i].t
            layers[i+1].tmin = layers[i].tmax
        layers[-1].tmax = t/2
        for layer in layers:
            layer.zi = [layer.loc_to_z(l_) for l_ in layer.__loc__]
        return True
    
    @classmethod
    def from_dict(cls, d: dict = None, **kwargs) -> 'Surface':  
        res = cls(**d)
        for addr, value in parsedicts_addr(d, inclusive=True):
            if len(addr) == 0:
                continue
            if 'hooke' in value:
                subcls = cls.__layerclass__
            else: 
                continue
            value['key'] = addr[-1]
            res[addr] = subcls(**value)
        return res
