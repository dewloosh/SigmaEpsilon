# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from sympy.physics.vector import ReferenceFrame as SymPyFrame
from typing import Iterable


from .utils import transpose_dcm_multi
from ...array import repeat
from ..array import Array


class ReferenceFrame(Array):
    """
    A base reference-frame for orthogonal vector spaces. 
    It facilitates tramsformation of tensor-like quantities across 
    different coordinate frames.

    The class is basically an interface on the `ReferenceFrame` class 
    in `sympy.physics.vector`, with a similarly working `orient_new` function.

    Parameters
    ----------
    axes : ndarray, Optional.
        2d numpy array of floats specifying cartesian reference frames.
        Dafault is None.
        
    parent : ReferenceFrame, Optional.
        A parent frame in which this the current frame is embedded in.
        Default is False.
    
    dim : int, Optional
        Dimension of the mesh. Deafult is 3.
            
    Examples
    --------
    Define a standard Cartesian frame and rotate it around axis 'Z'
    with an amount of 180 degrees:

    >>> A = ReferenceFrame(dim=3)
    >>> B = A.orient_new('Body', [0, 0, np.pi], 'XYZ')

    To create a third frame that rotates from B the way B rotates from A, we
    can do

    >>> A = ReferenceFrame(dim=3)
    >>> C = A.orient_new('Body', [0, 0, 2*np.pi], 'XYZ')

    or we can define it relative to B (this literally makes C to looke 
    in B like B looks in A)

    >>> C = ReferenceFrame(B.axes, parent=B)

    Notes
    -----
    The `dewloosh.geom.CartesianFrame` class takes the idea of the reference 
    frame a step further by introducing the idea of the 'origo'. 

    """

    def __init__(self, axes: ndarray = None, parent=None, *args,
                 order: str = 'row', name: str = None, dim: int = None, 
                 **kwargs):
        order = 'C' if order in ['row', 'C'] else 'F'
        try:
            if not isinstance(axes, ndarray):
                if isinstance(dim, Iterable):
                    if len(dim) == 1:
                        axes = np.eye(dim[0])
                    elif len(dim) == 2:
                        axes = repeat(np.eye(dim[-1]), dim[0])
                    else:
                        raise NotImplementedError
                elif isinstance(dim, int):
                    axes = np.eye(dim)
        except Exception as e:
            raise e
        super().__init__(axes, *args, order=order, **kwargs)
        self.name = name
        self.parent = parent
        self._order = 0 if order == 'C' else 1

    @classmethod
    def eye(cls, *args, dim=3, **kwargs):
        if len(args) > 0 and isinstance(args[0], int):
            dim = args[0]
        return cls(np.eye(dim), *args, **kwargs)

    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.root()

    @property
    def order(self) -> str:
        return 'row' if self._order == 0 else 'col'

    @property
    def axes(self) -> ndarray:
        """
        Returns a matrix, where each row (or column) is the component array
        of a basis vector with respect to the parent frame, or ambient
        space if there is none.
        """
        return self._array

    @axes.setter
    def axes(self, value: ndarray):
        if isinstance(value, np.ndarray):
            if value.shape == self._array.shape:
                self._array = value
            else:
                raise RuntimeError("Mismatch in data dimensinons!")
        else:
            raise TypeError("Only numpy arras are supported here!")

    def show(self, target: 'ReferenceFrame' = None) -> ndarray:
        """
        Returns the components of the current frame in a target frame.
        If the target is None, the componants are returned in the ambient frame.
        """
        return self.dcm(target=target)

    def dcm(self, *args, target: 'ReferenceFrame' = None,
            source: 'ReferenceFrame' = None, **kwargs) -> ndarray:
        """
        Returns the direction cosine matrix (DCM) of a transformation
        from a source (S) to a target (T) frame. The current frame can be the 
        source or the target, depending on the arguments. 

        If called without arguments, it returns the DCM matrix from the 
        root frame to the current frame (S=root, T=self).

        If `source` is not `None`, then T=self.

        If `target` is not `None`, then S=self.
        
        Parameters
        ----------
        source : 'ReferenceFrame', Optional
            Source frame. Default is None.

        target : 'ReferenceFrame', Optional
            Target frame. Default is None.

        Returns
        -------     
        numpy.ndarray
            DCM matrix from S to T.

        """
        if source is not None:
            # source must be a 3x3 array
            S, T = source.dcm(), self.dcm() 
            return T @ S.T
        elif target is not None:
            # target must be a 3x3 array
            S, T = self.dcm(), target.dcm()
            if len(S.shape) == 3:
                return T @ transpose_dcm_multi(S)
            elif len(S.shape) == 2:
                return T @ S.T
            else:
                msg = "There is no transformation rule imlemented for" \
                    " source shape {} and target shape {}"
                raise NotImplementedError(msg.format(S.shape, T.shape))
        # We only get here if the function is called without arguments.
        # The dcm from the ambient frame to the current frame is returned.
        if self.parent is None:
            return self.axes
        else:
            # parent should (but not must) be a 3x3 array
            return self.axes @ self.parent.dcm()

    def orient(self, *args, **kwargs) -> 'ReferenceFrame':
        """
        Orients the current frame inplace. 
        See `Referenceframe.orient_new` for the possible arguments.
        
        Parameters
        ----------
        args : tuple, Optional
            A tuple of arguments to pass to the `orientnew` 
            function in `sympy`. 

        kwargs : dict, Optional
            A dictionary of keyword arguments to pass to the 
            `orientnew` function in `sympy`. 

        """
        source = SymPyFrame('source')
        target = source.orientnew('target', *args, **kwargs)
        dcm = np.array(target.dcm(source).evalf()).astype(float)
        self._array = dcm @ self.axes
        return self

    def orient_new(self, *args, name='', **kwargs) -> 'ReferenceFrame':
        """
        Returns a new frame, oriented relative to the called object. 
        The orientation can be provided by all ways supported in 
        `sympy.orientnew`.

        Parameters
        ----------
        
        name : str
            Name for the new reference frame.

        rot_type : str
            The method used to generate the direction cosine matrix. Supported
            methods are:

            - ``'Axis'``: simple rotations about a single common axis
            - ``'DCM'``: for setting the direction cosine matrix directly
            - ``'Body'``: three successive rotations about new intermediate
              axes, also called "Euler and Tait-Bryan angles"
            - ``'Space'``: three successive rotations about the parent
              frames' unit vectors
            - ``'Quaternion'``: rotations defined by four parameters which
              result in a singularity free direction cosine matrix

        amounts :
            Expressions defining the rotation angles or direction cosine
            matrix. These must match the ``rot_type``. See examples below for
            details. The input types are:

            - ``'Axis'``: 2-tuple (expr/sym/func, Vector)
            - ``'DCM'``: Matrix, shape(3, 3)
            - ``'Body'``: 3-tuple of expressions, symbols, or functions
            - ``'Space'``: 3-tuple of expressions, symbols, or functions
            - ``'Quaternion'``: 4-tuple of expressions, symbols, or
              functions

        rot_order : str or int, optional
            If applicable, the order of the successive of rotations. The string
            ``'123'`` and integer ``123`` are equivalent, for example. Required
            for ``'Body'`` and ``'Space'``.

        Returns
        -------    

        ReferenceFrame
            A new ReferenceFrame object.

        """
        source = SymPyFrame('source')
        target = source.orientnew('target', *args, **kwargs)
        dcm = np.array(target.dcm(source).evalf()).astype(float)
        return self.__class__(axes=dcm, parent=self, name=name)


if __name__ == '__main__':

    A = ReferenceFrame()
    B = A.orient_new('Body', [0, 0, 30*np.pi/180],  'XYZ')
    C = B.orient_new('Body', [0, 0, 30*np.pi/180],  'XYZ')
