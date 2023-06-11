from typing import Iterable
from numbers import Number

import numpy as np
from numpy import ndarray

from dewloosh.core.tools import float_to_str_sig

from ..problem import NavierBeamProblem
from ..preproc import rhs_rect_const
from ..utils import points_to_rectangle_region
from .loadgroup import LoadGroup
from .mc import _coeffs_rect_load_mc

__all__ = ["RectangleLoad"]


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

    def rhs(self, *, problem: NavierBeamProblem = None) -> ndarray:
        """
        Returns the coefficients as a NumPy array.

        Parameters
        ----------
        problem: :class:`~sigmaepsilon.fourier.problem.NavierBeamProblem`, Optional
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
                
        if isinstance(v[0], Number) and isinstance(v[1], Number) and isinstance(v[2], Number):
            v = np.array(v, dtype=float) 
            return rhs_rect_const(p.size, p.shape, x, v)
        else:
            Lx, Ly = p.size
            M, N = p.shape
            N_monte_carlo = 1000
            
            rhs = np.zeros((1, M * N, 3), dtype=x.dtype)
            
            fnc = lambda x, ij : np.sin(x[:, 0] * np.pi * ij[0] / Lx) * np.sin(x[:, 1] * np.pi * ij[1] / Ly)
            if isinstance(v[0], str):
                str_expr = v[0]
            elif isinstance(v[0], Number):
                str_expr = str(v[0])
            else:
                raise TypeError("Invalid load type at position 0.")
            
            rhs[0, :, 0] = _coeffs_rect_load_mc(str_expr, x, p, fnc, N_monte_carlo)
            
            fnc = lambda x, ij : np.sin(x[:, 0] * np.pi * ij[0] / Lx) * np.cos(x[:, 1] * np.pi * ij[1] / Ly)
            if isinstance(v[1], str):
                str_expr = v[1]
            elif isinstance(v[1], Number):
                str_expr = str(v[1])
            else:
                raise TypeError("Invalid load type at position 1.")
            
            rhs[0, :, 1] = _coeffs_rect_load_mc(str_expr, x, p, fnc, N_monte_carlo)
            
            fnc = lambda x, ij : np.cos(x[:, 0] * np.pi * ij[0] / Lx) * np.sin(x[:, 1] * np.pi * ij[1] / Ly)
            if isinstance(v[2], str):
                str_expr = v[2]
            elif isinstance(v[2], Number):
                str_expr = str(v[2])
            else:
                raise TypeError("Invalid load type at position 2.")
            
            rhs[0, :, 2] = _coeffs_rect_load_mc(str_expr, x, p, fnc, N_monte_carlo)
            
            return rhs

    def __repr__(self):
        return "RectangleLoad(%s)" % (dict.__repr__(self))