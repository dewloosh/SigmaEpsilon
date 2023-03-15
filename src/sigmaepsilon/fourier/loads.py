import json
import numpy as np
from numpy import ndarray, average as avg
from typing import Union, Iterable, Any
from numbers import Number

from dewloosh.core.tools import float_to_str_sig
from linkeddeepdict import LinkedDeepDict
from linkeddeepdict.tools.dtk import parsedicts_addr
from neumann.function import Function

from .problem import NavierProblem
from .preproc import rhs_rect_const, rhs_conc_1d, rhs_conc_2d, rhs_line_const
from .utils import points_to_rectangle_region, sin1d, cos1d


class NavierLoadError(Exception):
    """
    Exception raised for invalid load inputs.
    """

    def __init__(self, message=None):
        if message is None:
            message = (
                "Invalid input for loads. "
                "It must be a dictionary, a NumPy array, or "
                "an instance of `sigmaepsilon.fourier.LoadGroup`."
            )
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

    >>> from sigmaepsilon.fourier import LoadGroup, PointLoad
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
    def problem(self, value: NavierProblem):
        """
        Sets the attached problem.

        Parameters
        ----------
        value : NavierProblem
            The problem the loads are defined for.
        """
        assert self.is_root(), "The problem can only be set on the top-level object."
        self.__problem = value

    def blocks(
        self,
        *args,
        inclusive: bool = False,
        blocktype: Any = None,
        deep: bool = True,
        **kwargs,
    ) -> Iterable["LoadGroup"]:
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
        return self.containers(self, inclusive=inclusive, dtype=dtype, deep=deep)

    def cases(self, *args, inclusive: bool = True, **kwargs) -> Iterable["LoadGroup"]:
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
        return filter(
            lambda i: i.__class__._typestr_ is not None,
            self.blocks(*args, inclusive=inclusive, **kwargs),
        )

    @staticmethod
    def _string_to_dtype_(string: str = None):
        if string == "group":
            return LoadGroup
        elif string == "rectangle":
            return RectangleLoad
        elif string == "point":
            return PointLoad
        elif string == "line":
            return LineLoad
        else:
            raise NotImplementedError()

    def dump(self, path: str, *, mode: str = "w", indent: int = 4):
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
    def from_json(cls, path: str = None) -> "LoadGroup":
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
            with open(path, "r") as f:
                d = json.load(f)
            return cls.from_dict(d)

    def _encode_(self, *_, **__) -> dict:
        """
        Returns the group as a dictionary. Overwrite this in child
        implementations.
        """
        res = {}
        cls = type(self)
        res = {
            "type": cls._typestr_,
            "key": self.key,
        }
        return res

    @classmethod
    def _decode_(cls, d: dict = None, *_, **kwargs):
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
            "key": d.pop("key", None),
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
    def from_dict(d: dict = None, **kwargs) -> "LoadGroup":
        """
        Reads a LoadGroup from a dictionary. The keys in the dictionaries
        must match the parameters of the corresponding load types and a type
        indicator.
        """
        if "type" in d:
            cls = LoadGroup._string_to_dtype_(d["type"])
        else:
            cls = LoadGroup
        res = cls(**d)
        for addr, value in parsedicts_addr(d, inclusive=True):
            if len(addr) == 0:
                continue
            if "type" in value:
                cls = LoadGroup._string_to_dtype_(value["type"])
            else:
                cls = LoadGroup
            value["key"] = addr[-1]
            res[addr] = cls(**value)
        return res

    def __repr__(self):
        return "LoadGroup(%s)" % (dict.__repr__(self))


class RectangleLoad(LoadGroup):
    """
    A class to handle rectangular loads.

    Parameters
    ----------
    v: Iterable
       Load intensities for each dof in the order :math:`fz, mx, my`.
    x: Iterable, Optional
       The coordinates of the lower-left and upper-right points of the region
       where the load is applied. Default is None.
    """

    _typestr_ = "rectangle"

    def __init__(self, *args, x: Iterable = None, v: Iterable = None, **kwargs):
        super().__init__(*args, x=x, v=v, **kwargs)

    @classmethod
    def _decode_(cls, d: dict = None, *args, **kwargs):
        if d is None:
            d = kwargs
            kwargs = None
        if kwargs is not None:
            d.update(kwargs)
        clskwargs = {
            "key": d.pop("key", None),
            "x": np.array(d.pop("x")),
            "v": np.array(d.pop("v")),
        }
        clskwargs.update(d)
        return cls(**clskwargs)

    def _encode_(self, *args, **kwargs) -> dict:
        res = {}
        cls = type(self)
        res = {
            "type": cls._typestr_,
            "key": self.key,
            "x": float_to_str_sig(self["x"], sig=6),
            "v": float_to_str_sig(self["v"], sig=6),
        }
        return res

    def region(self) -> Iterable:
        """
        Returns the region as a list of 4 values x0, y0, w, and h, where x0 and y0 are
        the coordinates of the bottom-left corner, w and h are the width and height
        of the region.
        """
        assert "x" in self, "There are no points defined."
        return points_to_rectangle_region(self["x"])

    def rhs(self, *, problem: NavierProblem = None) -> ndarray:
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
        x = np.array(self["x"], dtype=float)
        v = np.array(self["v"], dtype=float)
        return rhs_rect_const(p.size, p.shape, x, v)

    def __repr__(self):
        return "RectangleLoad(%s)" % (dict.__repr__(self))


class LineLoad(LoadGroup):
    """
    A class to handle loads over lines for both beam and plate problems.

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

    _typestr_ = "line"

    def __init__(self, *args, x: Iterable = None, v: Iterable = None, **kwargs):
        super().__init__(*args, x=x, v=v, **kwargs)

    @classmethod
    def _decode_(cls, d: dict = None, *_, **kwargs):
        if d is None:
            d = kwargs
            kwargs = None
        if kwargs is not None:
            d.update(kwargs)
        clskwargs = {
            "key": d.pop("key", None),
            "x": np.array(d.pop("x")),
            "v": np.array(d.pop("v")),
        }
        clskwargs.update(d)
        return cls(**clskwargs)

    def _encode_(self, *args, **kwargs) -> dict:
        res = {}
        cls = type(self)
        res = {
            "type": cls._typestr_,
            "key": self.key,
            "x": float_to_str_sig(self["x"], sig=6),
            "v": float_to_str_sig(self["v"], sig=6),
        }
        return res

    def rhs(self, *, problem: NavierProblem = None) -> ndarray:
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
        x = np.array(self["x"], dtype=float)
        v = self["v"]
        if len(x.shape) == 1:
            assert len(v) == 2, f"Invalid shape {v.shape} for load intensities."
            if isinstance(v[0], Number) and isinstance(v[1], Number):
                v = np.array(v, dtype=float)
                return rhs_line_const(p.length, p.N, v, x)
            else:
                rhs = np.zeros((1, p.N, 2), dtype=x.dtype)
                if isinstance(v[0], str):
                    f = Function(v[0], variables=["x"], dim=1)
                    L = p.length
                    points = np.linspace(x[0], x[1], 1000)
                    d = x[1] - x[0]
                    rhs[0, :, 0] = list(
                        map(
                            lambda i: (2 / L)
                            * d
                            * avg(sin1d(points, i, L) * f([points])),
                            np.arange(1, p.N + 1),
                        )
                    )
                elif isinstance(v[0], Number):
                    _v = np.array([v[0], 0], dtype=float)
                    rhs[0, :, 0] = rhs_line_const(p.length, p.N, _v, x)[0, :, 0]
                if isinstance(v[1], str):
                    f = Function(v[1], variables=["x"], dim=1)
                    L = p.length
                    points = np.linspace(x[0], x[1], 1000)
                    d = x[1] - x[0]
                    rhs[0, :, 1] = list(
                        map(
                            lambda i: (2 / L)
                            * d
                            * avg(cos1d(points, i, L) * f([points])),
                            np.arange(1, p.N + 1),
                        )
                    )
                elif isinstance(v[1], Number):
                    _v = np.array([0, v[1]], dtype=float)
                    rhs[0, :, 1] = rhs_line_const(p.length, p.N, _v, x)[0, :, 1]
                return rhs

    def __repr__(self):
        return "LineLoad(%s)" % (dict.__repr__(self))


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

    _typestr_ = "point"

    def __init__(
        self, *args, x: Union[float, Iterable] = None, v: Iterable = None, **kwargs
    ):
        super().__init__(*args, x=x, v=v, **kwargs)

    @classmethod
    def _decode_(cls, d: dict = None, *_, **kwargs):
        if d is None:
            d = kwargs
            kwargs = None
        if kwargs is not None:
            d.update(kwargs)
        clskwargs = {
            "key": d.pop("key", None),
            "x": np.array(d.pop("x")),
            "v": np.array(d.pop("v")),
        }
        clskwargs.update(d)
        return cls(**clskwargs)

    def _encode_(self, *args, **kwargs) -> dict:
        res = {}
        cls = type(self)
        res = {
            "type": cls._typestr_,
            "key": self.key,
            "x": float_to_str_sig(self["x"], sig=6),
            "v": float_to_str_sig(self["v"], sig=6),
        }
        return res

    def rhs(self, *, problem: NavierProblem = None) -> ndarray:
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
        x = self["x"]
        v = np.array(self["v"])
        if hasattr(p, "size"):
            return rhs_conc_2d(p.size, p.shape, v, x)
        else:
            return rhs_conc_1d(p.length, p.N, v, x)

    def __repr__(self):
        return "PointLoad(%s)" % (dict.__repr__(self))
