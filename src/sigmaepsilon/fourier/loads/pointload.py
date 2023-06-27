import numpy as np
from numpy import ndarray
from typing import Union, Iterable

from dewloosh.core.tools import float_to_str_sig

from ..problem import NavierProblem
from ..preproc import rhs_conc_1d, rhs_conc_2d
from .loadgroup import LoadGroup


__all__ = ["PointLoad"]


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
        problem : :class:`~sigmaepsilon.fourier.problem.NavierProblem`, Optional
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
