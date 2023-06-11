import json
from typing import Iterable, Any

from linkeddeepdict.tools.dtk import parsedicts_addr

from ..problem import NavierBeamProblem
from .meta import ABC_LoadGroup


__all__ = ["NavierLoadError", "RectangleLoad", "LineLoad", "PointLoad"]


class NavierLoadError(Exception):
    """
    Exception raised for invalid load inputs.
    """

    def __init__(self, message=None):
        if message is None:
            message = (
                "Invalid input for loads. "
                "It must be a dictionary, a NumPy array, or "
                "an instance of `sigmaepsilon.fourier.loads.LoadGroup`."
            )
        super().__init__(message)


class LoadGroup(ABC_LoadGroup):
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

    _typestr_ = "group"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__problem = None

    @property
    def problem(self) -> NavierBeamProblem:
        """
        Returns the attached problem.
        """
        if self.is_root():
            return self.__problem
        return self.root().problem

    @problem.setter
    def problem(self, value: NavierBeamProblem):
        """
        Sets the attached problem.

        Parameters
        ----------
        value: `~sigmaepsilon.fourier.problem.NavierBeamProblem`
            The problem the loads are defined for.
        """
        assert self.is_root(), "The problem can only be set on the top-level object."
        self.__problem = value

    def blocks(
        self,
        inclusive: bool = False,
        blocktype: Any = None,
        deep: bool = True,
    ) -> Iterable["LoadGroup"]:
        """
        Returns a generator object that yields all the subgroups.

        Parameters
        ----------
        inclusive: bool, Optional
            If True, returns the object the call is made upon.
            Default is False.
        blocktype: Any, Optional
            The type of the load groups to return. Default is None, that
            returns all types.
        deep: bool, Optional
            If True, all deep groups are returned separately. Default is True.

        Yields
        ------
        LoadGroup
        """
        dtype = LoadGroup if blocktype is None else blocktype
        return self.containers(self, inclusive=inclusive, dtype=dtype, deep=deep)

    def cases(self, *, inclusive: bool = True, **kwargs) -> Iterable["LoadGroup"]:
        """
        Returns a generator that yields the load cases in the group.

        Parameters
        ----------
        inclusive: bool, Optional
            If True, returns the object the call is made upon.
            Default is True.
        blocktype: Any, Optional
            The type of the load groups to return. Default is None, that
            returns all types.
        deep: bool, Optional
            If True, all deep groups are returned separately. Default is True.

        Yields
        ------
        LoadGroup
        """
        return filter(
            lambda i: i.__class__._typestr_ != LoadGroup._typestr_,
            self.blocks(inclusive=inclusive, **kwargs),
        )

    def dump(self, path: str, *, mode: str = "w", indent: int = 4):
        """
        Dumps the content of the object to a file.

        Parameters
        ----------
        path: str
            The path of the file on your filesystem.
        mode: str, Optional
            https://www.programiz.com/python-programming/file-operation
        indent: int, Optional
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
        path: str
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
        d: dict, Optional
            A dictionary.
        **kwargs: dict, Optional
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

