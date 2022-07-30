# -*- coding: utf-8 -*-
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from ...core import Wrapper
from ...core.abc import ABC_Safe

from ..array import ascont


__all__ = ['ArrayBase', 'Array']


class ArrayBase(ABC_Safe, np.ndarray):
    """
    Base backend class for array-like classes. Use it like if it was
    a ``numpy.ndarray`` instance.
    
    """

    def __new__(subtype, shape=None, dtype=float, buffer=None,
                offset=0, strides=None, order=None, frame=None,
                inds=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = super().__new__(subtype, shape, dtype,
                              buffer, offset, strides, order)
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return 0to
        #    InfoArray.__new__)
        if obj is None:
            return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).


class Array(NDArrayOperatorsMixin, Wrapper):
    """
    Base frontend class for array-like classes. Use it like if it 
    was a ``numpy.ndarray`` instance.
    
    """

    _array_cls_ = ArrayBase

    def __init__(self, *args, cls_params=None, contiguous=True, **kwargs):
        if len(args) > 0 and isinstance(args[0], np.ndarray):
            buf = ascont(args[0]) if contiguous else args[0]
        else:
            buf = np.array(*args, **kwargs)
        cls_params = dict() if cls_params is None else cls_params
        self._array = self._array_cls_(shape=buf.shape, buffer=buf,
                                       dtype=buf.dtype, **cls_params)
        super(Array, self).__init__(wrap=self._array)

    @property
    def dim(self):
        return len(self._array.shape)

    def __repr__(self):
        return f"{self.__class__.__name__}\n({self._array})"

    def __array__(self, dtype=None):
        if dtype is not None:
            return self._array.astype(dtype)
        return self._array

    def __getitem__(self, key):
        return self._array.__getitem__(key)

    def __setitem__(self, key):
        return self._array.__setitem__(key)

    def __len__(self):
        return self._array.shape[0]

    def to_numpy(self):
        return self.__array__()