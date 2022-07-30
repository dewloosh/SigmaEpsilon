# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray

from .utils import show_vector, show_vectors
from .frame import ReferenceFrame as Frame
from .array import ArrayBase, Array


__all__ = ['Vector']


class VectorBase(ArrayBase):
    """Base class for vector backends. Use it like if it 
    was a ``numpy.ndarray`` instance."""

    def __new__(subtype, shape=None, dtype=float, buffer=None,
                offset=0, strides=None, order=None, frame=None):
        obj = super().__new__(subtype, shape, dtype, buffer,
                              offset, strides, order)
        obj._frame = frame
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._frame = getattr(obj, '_frame', None)

    @property
    def frame(self) -> Frame:
        return self._frame

    @frame.setter
    def frame(self, value: Frame):
        if isinstance(value, Frame):
            self._frame = value
        else:
            raise TypeError('Value must be a {} instance'.format(Frame))


class Vector(Array):
    """
    Extends `NumPy`'s ``ndarray`` class to handle arrays with associated
    reference frames. The class also provides a mechanism to transform
    vectors between different frames. Use it like if it was a ``numpy.ndarray`` 
    instance.
    
    All parameters are identical to those of ``numpy.ndarray``, except that
    this class allows to specify an embedding frame.
    
    Parameters
    ----------
    args : Tuple, Optional.
        Positional arguments forwarded to `numpy.ndarray`.
    
    frame : ndarray, Optional.
        The reference frame the vector is represented by its coordinates.
    
    kwargs : Dict, Optional.
        Keyword arguments forwarded to `numpy.ndarray`.
    
    Examples
    --------
    Import the necessary classes:
    
    >>> from dewloosh.math import Vector, ReferenceFrame as Frame
    
    Create a default frame in 3d space, and create 2 others, each
    being rotated with 30 degrees around the third axis.
    
    >>> A = Frame(dim=3)
    >>> B = A.orient_new('Body', [0, 0, 30*np.pi/180], 'XYZ')
    >>> C = B.orient_new('Body', [0, 0, 30*np.pi/180], 'XYZ')

    To create a vector in a frame:
    
    >>> vA = Vector([1.0, 1.0, 0.0], frame=A)
    
    To create a vector with a relative transformation to another one:
    
    >>> vB = vA.orient_new('Body', [0, 0, -30*np.pi/180], 'XYZ')
    
    Use the `array` property to query the componets of a `Vector`:
    
    >>> vB.array
    
    If you want to obtain the components of a vector in a specific
    target frame (C), do this:
    
    >>> vB.show(C)
    
    Hence, to create a vector in a target frame (C), knowing the components in a 
    source frame (A):
    
    >>> vC = Vector(vA.show(C), frame=C)
    
    See Also
    --------
    :class:`dewloosh.math.linalg.vector.VectorBase`
    :class:`dewloosh.math.linalg.array.Array`
    :class:`dewloosh.math.linalg.frame.frame.ReferenceFrame`
    
    """

    _array_cls_ = VectorBase
    _frame_cls_ = Frame

    def __init__(self, *args, frame=None, **kwargs):
        cls_params = kwargs.get('cls_params', dict())
        if frame is not None:
            cls_params['frame'] = frame
        kwargs['cls_params'] = cls_params
        super().__init__(*args, **kwargs)
        if self._array._frame is None:
            self._array._frame = self._frame_cls_(dim=self._array.shape)

    @property
    def array(self) -> VectorBase:
        """Returns the coordinates of the vector."""
        return self._array

    @array.setter
    def array(self, value):
        """Sets the coordinates of the vector."""
        buf = np.array(value)
        assert buf.shape == self._array.shape
        self._array = self._array_cls_(shape=buf.shape, buffer=buf,
                                       dtype=buf.dtype)

    def show(self, target: Frame = None, *args, dcm=None, **kwargs) -> ndarray:
        """
        Returns the components in a target frame. If the target is 
        `None`, the components are returned in the global frame.
        
        The transformation can also be specified with a proper DCM matrix.
        
        Parameters
        ----------
        target : ndarray, Optional.
            Target frame.
        
        dcm : ndarray, Optional.
            The dcm matrix of the transformation.

        Returns
        -------      
        numpy.ndarray
            The components of the vector in a specified frame, or
            the global frame, depending on the arguments.

        """
        if not isinstance(dcm, ndarray):
            if target is None:
                target = Frame(dim=self._array.shape[-1])
            dcm = self.frame.dcm(target=target)
        if len(self.array.shape) == 1:
            return show_vector(dcm, self.array)  # dcm @ arr  
        else:
            return show_vectors(dcm, self.array)  # dcm @ arr 
        
    def orient(self, *args, **kwargs):
        dcm = Frame.eye(dim=len(self)).orient_new(*args, **kwargs).dcm()
        self.array = dcm.T @ self._array
        return self

    def orient_new(self, *args, keep_frame=True, **kwargs):
        dcm = Frame.eye(dim=len(self)).orient_new(*args, **kwargs).dcm()
        if keep_frame:
            array = dcm.T @ self._array
            return Vector(array, frame=self.frame)
        else:
            raise NotImplementedError

    def __repr__(self):
        return np.ndarray.__repr__(self._array)

    def __str__(self):
        return np.ndarray.__str__(self._array)


if __name__ == '__main__':

    A = Frame(dim=3)
    B = A.orient_new('Body', [0, 0, 30*np.pi/180], 'XYZ')
    C = B.orient_new('Body', [0, 0, 30*np.pi/180], 'XYZ')

    vA = Vector([1.0, 1.0, 0.0], frame=A)
    vB = vA.orient_new('Body', [0, 0, -30*np.pi/180], 'XYZ')
    vC = Vector(vA.show(C), frame=C)
