# -*- coding: utf-8 -*-
import json
import numpy as np
from numpy import ndarray
from typing import Union, Iterable, Any

from dewloosh.core.tools import (allinkwargs, popfromkwargs, 
                                 float_to_str_sig)
from linkeddeepdict import LinkedDeepDict
from linkeddeepdict.tools.dtk import parsedicts_addr

from .problem import NavierProblem
from .preproc import (rhs_rect_const, rhs_conc_1d, rhs_conc_2d, 
                      rhs_line_const)


class NavierLoadError(Exception):
    """
    Exception raised for invalid load inputs.
    """

    def __init__(self, message=None):
        if message is None:
            message = ("Invalid input for loads. "
                        "It must be a dictionary, a NumPy array, or "
                        "an instance of `sigmaepsilon.solid.fourier.LoadGroup`.")
        super().__init__(message)


class LoadGroup(LinkedDeepDict):
    """
    A class to handle load groups for Navier's semi-analytic solution of
    rectangular plates and beams with specific boundary conditions.
    
    This class is also the base class of all other load types.
    
    See Also
    --------
    :class:`LinkedDeepDict`
    
    Examples
    --------
    
    >>> from sigmaepsilon.solid.fourier import LoadGroup, PointLoad
    >>> loads = LoadGroup(   
    >>>     group1 = LoadGroup(
    >>>         case1 = PointLoad(x=L/3, v=[1.0, 0.0]),
    >>>         case2 = PointLoad(x=L/3, v=[0.0, 1.0]),
    >>>     ),
    >>>     group2 = LoadGroup(
    >>>         case1 = PointLoad(x=2*L/3, v=[1.0, 0.0]),
    >>>         case2 = PointLoad(x=2*L/3, v=[0.0, 1.0]),
    >>>     ),    
    >>> )
    
    Since the LoadGroup is a subclass of LinkedDeepDict,
    a case is accessible as
    
    >>> loads['group1', 'case1']
    
    If you want to protect the object from the accidental
    creation of nested subdirectories, you can lock the layout
    by typing
    
    >>> loads.lock()
         
    """
    _typestr_ = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__problem = None

    @property
    def problem(self) -> NavierProblem:
        """
        Returns the attached problem.
        """
        if self.is_root():
            return self.__problem
        return self.root().problem
    
    @problem.setter
    def problem(self, value:NavierProblem):
        """
        Sets the attached problem.
        
        Parameters
        ----------
        value : NavierProblem
            The problem the loads are defined for.
        """
        assert self.is_root(), "The problem can only be set on the top-level object."
        self.__problem = value

    def blocks(self, *args, inclusive:bool=False, blocktype:Any=None,
               deep:bool=True, **kwargs) -> Iterable['LoadGroup']:
        """
        Returns a generator object that yields all the subgroups.
        
        Parameters
        ----------
        inclusive : bool, Optional
            If True, returns the object the call is made upon.
            Default is False.
            
        blocktype : Any, Optional
            The type of the load groups to return. Default is None, that
            returns all types.
            
        deep : bool, Optional
            If True, all deep groups are returned separately. Default is True.
        
        Yields
        ------
        LoadGroup
        
        """
        dtype = LoadGroup if blocktype is None else blocktype
        return self.containers(self, inclusive=inclusive,
                               dtype=dtype, deep=deep)

    def cases(self, *args, inclusive:bool=True, **kwargs) -> Iterable['LoadGroup']:
        """
        Returns a generator that yields the load cases in the group.
        
        Parameters
        ----------
        inclusive : bool, Optional
            If True, returns the object the call is made upon.
            Default is True.
            
        blocktype : Any, Optional
            The type of the load groups to return. Default is None, that
            returns all types.
            
        deep : bool, Optional
            If True, all deep groups are returned separately. Default is True.
        
        Yields
        ------
        LoadGroup
        
        """
        return filter(lambda i: i.__class__._typestr_ is not None, 
                      self.blocks(*args, inclusive=inclusive, **kwargs))

    @staticmethod
    def _string_to_dtype_(string: str = None):
        if string == 'group':
            return LoadGroup
        elif string == 'rectangle':
            return RectangleLoad
        elif string == 'point':
            return PointLoad
        elif string == 'line':
            return LineLoad
        else:
            raise NotImplementedError()

    def dump(self, path:str, *, mode:str='w', indent:int=4):
        """
        Dumps the content of the object to a file.
        
        Parameters
        ----------
        path : str
            The path of the file on your filesystem.
            
        mode : str, Optional
            https://www.programiz.com/python-programming/file-operation
            
        indent : int, Optional
            Governs the level to which members will be pretty-printed.
            Default is 4.

        """
        with open(path, mode) as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_json(cls, path: str = None) -> 'LoadGroup':
        """
        Loads a loadgroup from a JSON file.
        
        Parameters
        ----------
        path : str
            The path to a file on your filesystem.
            
        Returns
        -------
        LoadGroup
        
        """
        if path is not None:
            with open(path, 'r') as f:
                d = json.load(f)
            return cls.from_dict(d)

    def _encode_(self, *args, **kwargs) -> dict:
        """
        Returns the group as a dictionary. Overwrite this in child 
        implementations.
        """
        res = {}
        cls = type(self)
        res = {
            'type': cls._typestr_,
            'key': self.key,
        }
        return res

    @classmethod
    def _decode_(cls, d: dict = None, *args, **kwargs):
        """
        Returns a LoadGroup object from a dictionary. 
        Overwrite this in child implementations.
        
        Parameters
        ----------
        d : dict, Optional
            A dictionary.
        
        **kwargs : dict, Optional
            Keyword arguments defining a load group.
            
        """
        if d is None:
            d = kwargs
            kwargs = None
        if kwargs is not None:
            d.update(kwargs)
        clskwargs = {
            'key': d.pop('key', None),
        }
        clskwargs.update(d)
        return cls(**clskwargs)

    def to_dict(self) -> dict:
        """
        Returns the group as a dictionary.
        """
        res = self._encode_()
        for key, value in self.items():
            if isinstance(value, LoadGroup):
                res[key] = value.to_dict()
            else:
                res[key] = value
        return res

    @staticmethod
    def from_dict(d: dict = None, **kwargs) -> 'LoadGroup':
        """
        Reads a LoadGroup from a dictionary. The keys in the dictionaries
        must match the parameters of the corresponding load types and a type
        indicator.
        """
        if 'type' in d:
            cls = LoadGroup._string_to_dtype_(d['type'])
        else:  
            cls = LoadGroup
        res = cls(**d)
        for addr, value in parsedicts_addr(d, inclusive=True):
            if len(addr) == 0:
                continue
            if 'type' in value:
                cls = LoadGroup._string_to_dtype_(value['type'])
            else: 
                cls = LoadGroup
            value['key'] = addr[-1]
            res[addr] = cls(**value)
        return res

    def __repr__(self):
        return 'LoadGroup(%s)' % (dict.__repr__(self))


class RectangleLoad(LoadGroup):
    """
    A class to handle rectangular loads.
    
    Parameters
    ----------
    value: Iterable
        1d or 2d iterable of scalars for all 3 degrees of
        freedom in the order :math:`mx, my, fz`.
        
    points: Iterable, Optional
        The coordinates of the lower-left and upper-right points of the region
        where the load is applied. Default is None.

    **kwargs : dict, Optional
        If the region of application is not specified by the argument 'points',
        extra keyword arguments are forwarded to :func:`get_coords`. Default is None.
    
    """
    _typestr_ = 'rectangle'

    def __init__(self, *args, value:Iterable, points:Iterable=None, **kwargs):
        if points is not None:
            self.points = np.array(points, dtype=float)
        else:
            self.points = RectangleLoad.get_coords(kwargs)
        self.value = np.array(value, dtype=float)
        super().__init__()

    def _encode_(self, *args, **kwargs) -> dict:
        res = {}
        cls = type(self)
        res = {
            'type': cls._typestr_,
            'key': self.key,
            'region': float_to_str_sig(self.region(), sig=6),
            'value': float_to_str_sig(self.value, sig=6),
        }
        return res

    @classmethod
    def _decode_(cls, d: dict = None, *args, **kwargs):
        if d is None:
            d = kwargs
            kwargs = None
        if kwargs is not None:
            d.update(kwargs)
        points = RectangleLoad.get_coords(d)
        clskwargs = {
            'key': d.pop('key', None),
            'points': points,
            'value': np.array(d.pop('value'), dtype=float)
        }
        clskwargs.update(d)
        return cls(**clskwargs)

    @staticmethod
    def get_coords(d: dict = None, *args, **kwargs):
        """
        Returns the bottom-left and upper-right coordinates of the region
        of the load from several inputs.
        
        Parameters
        ----------
        d : dict, Optional
            A dictionary, which is equivalrent to a parameter set from the
            other parameters listed here. Default is None.
            
        region : Iterable, Optional
            An iterable of length 4 with values x0, y0, w, and h. Here x0 and y0 are
            the coordinates of the bottom-left corner, w and h are the width and height
            of the region.
            
        xy : Iterable, Optional
            The position of the bottom-left corner as an iterable of length 2.
            
        w : float, Optional
            The width of the region.
            
        h : float, Optional
            The height of the region.
            
        center : Iterable, Optional
            The coordinates of the center of the region.
            
        Returns
        -------
        numpy.ndarray
            A 2d float array of coordinates, where the entries of the first and second 
            rows are the coordinates of the lower-left and upper-right points of the region.
        
        Examples
        --------
        The following definitions return the same output:
        
        >>> from sigmaepsilon.solid.fourier import RectLoad
        >>> RectLoad.get_coords(region=[2, 3, 0.5, 0.7])
        >>> RectLoad.get_coords(xy=[2, 3], w=0.5, h=0.7)
        >>> RectLoad.get_coords(center=[2.25, 3.35], w=0.5, h=0.7)
        >>> RectLoad.get_coords(dict(center=[2.25, 3.35], w=0.5, h=0.7))
        
        """
        points = None
        if d is None:
            d = kwargs
        try:
            if 'points' in d:
                points = np.array(d.pop('points'))
            elif 'region' in d:
                x0, y0, w, h = np.array(d.pop('region'))
                points = np.array([[x0, y0], [x0 + w, y0 + h]])
            elif allinkwargs(['xy', 'w', 'h'], **d):
                (x0, y0), w, h = popfromkwargs(['xy', 'w', 'h'], d)
                points = np.array([[x0, y0], [x0 + w, y0 + h]])
            elif allinkwargs(['center', 'w', 'h'], **d):
                (xc, yc), w, h = popfromkwargs(['center', 'w', 'h'], d)
                points = np.array([[xc - w/2, yc - h/2],
                                   [xc + w/2, yc + h/2]])
        except Exception as e:
            print(e)
            return None
        return points

    def region(self) -> Iterable:
        """
        Returns the region as a list of 4 values x0, y0, w, and h, where x0 and y0 are
        the coordinates of the bottom-left corner, w and h are the width and height
        of the region.
        """
        assert self.points is not None, "There are no points defined."
        return _points_to_region_(self.points)

    def rhs(self, *, problem:NavierProblem=None) -> ndarray:
        """
        Returns the coefficients as a NumPy array.
        
        Parameters
        ----------
        problem : NavierProblem, Optional
            A problem the coefficients are generated for. If not specified,
            the attached problem of the object is used. Default is None.
            
        Returns
        -------
        numpy.ndarray
            2d float array of shape (H, 3), where H is the total number 
            of harmonic terms involved (defined for the problem).
            
        """
        p = problem if problem is not None else self.problem
        return rhs_rect_const(p.size, p.shape, self.value, self.points)

    def __repr__(self):
        return 'RectangleLoad(%s)' % (dict.__repr__(self))


class LineLoad(LoadGroup):
    """
    A class to handle loads over lines.
    
    Parameters
    ----------
    x : Iterable
        The point of application as an 1d iterable for a beam, a 2d iterable
        for a plate. In the latter case, the first row is the first point, the
        second row is the second point.
        
    v : Iterable
        Load intensities for each dof. The order of the dofs for a beam
        is [F, M], for a plate it is [F, Mx, My].
        
    """
    _typestr_ = 'line'

    def __init__(self, *args, x:Iterable=None, 
                 v:Iterable=None, **kwargs):
        super().__init__(*args, x=x, v=v, **kwargs)
        
    @classmethod
    def _decode_(cls, d: dict = None, *args, **kwargs):
        if d is None:
            d = kwargs
            kwargs = None
        if kwargs is not None:
            d.update(kwargs)
        clskwargs = {
            'key': d.pop('key', None),
            'x': np.array(d.pop('x')),
            'v': np.array(d.pop('v')),
        }
        clskwargs.update(d)
        return cls(**clskwargs)
    
    def _encode_(self, *args, **kwargs) -> dict:
        res = {}
        cls = type(self)
        res = {
            'type': cls._typestr_,
            'key': self.key,
            'x': float_to_str_sig(self['x'], sig=6),
            'v': float_to_str_sig(self['v'], sig=6),
        }
        return res
        
    def rhs(self, *, problem:NavierProblem=None) -> ndarray:
        """
        Returns the coefficients as a NumPy array.
        
        Parameters
        ----------
        problem : NavierProblem, Optional
            A problem the coefficients are generated for. If not specified,
            the attached problem of the object is used. Default is None.
            
        Returns
        -------
        numpy.ndarray
            2d float array of shape (H, 3), where H is the total number 
            of harmonic terms involved (defined for the problem).
            
        """
        p = problem if problem is not None else self.problem
        x = np.array(self['x'], dtype=float) 
        v = np.array(self['v'], dtype=float)
        return rhs_line_const(p.length, p.N, x, v)

    def __repr__(self):
        return 'LineLoad(%s)' % (dict.__repr__(self))


class PointLoad(LoadGroup):
    """
    A class to handle concentrated loads.
    
    Parameters
    ----------
    x : Union[float, Iterable]
        The point of application. A scalar for a beam, an iterable of
        length 2 for a plate.
        
    v : Iterable
        Load values for each dof. The order of the dofs for a beam
        is [F, M], for a plate it is [F, Mx, My].
    
    """
    _typestr_ = 'point'

    def __init__(self, *args, x:Union[float, Iterable]=None, 
                 v:Iterable=None, **kwargs):
        super().__init__(*args, x=x, v=v, **kwargs)

    @classmethod
    def _decode_(cls, d: dict = None, *args, **kwargs):
        if d is None:
            d = kwargs
            kwargs = None
        if kwargs is not None:
            d.update(kwargs)
        clskwargs = {
            'key': d.pop('key', None),
            'x': np.array(d.pop('x')),
            'v': np.array(d.pop('v')),
        }
        clskwargs.update(d)
        return cls(**clskwargs)

    def _encode_(self, *args, **kwargs) -> dict:
        res = {}
        cls = type(self)
        res = {
            'type': cls._typestr_,
            'key': self.key,
            'x': float_to_str_sig(self['x'], sig=6),
            'v': float_to_str_sig(self['v'], sig=6),
        }
        return res
    
    def rhs(self, *, problem:NavierProblem=None) -> ndarray:
        """
        Returns the coefficients as a NumPy array.
        
        Parameters
        ----------
        problem : NavierProblem, Optional
            A problem the coefficients are generated for. If not specified,
            the attached problem of the object is used. Default is None.
            
        Returns
        -------
        numpy.ndarray
            2d float array of shape (H, 3), where H is the total number 
            of harmonic terms involved (defined for the problem).
            
        """
        p = problem if problem is not None else self.problem
        x = self['x'] 
        v = np.array(self['v'])
        if hasattr(p, 'size'):
            return rhs_conc_2d(p.size, p.shape, v, x)
        else:
            return rhs_conc_1d(p.length, p.N, v, x)

    def __repr__(self):
        return 'PointLoad(%s)' % (dict.__repr__(self))


def _points_to_region_(points: ndarray):
    xmin = points[:, 0].min()
    ymin = points[:, 1].min()
    xmax = points[:, 0].max()
    ymax = points[:, 1].max()
    return xmin, ymin, xmax - xmin, ymax - ymin