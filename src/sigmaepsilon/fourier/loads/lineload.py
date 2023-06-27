from numbers import Number

import numpy as np
from numpy import ndarray

from dewloosh.core.tools import float_to_str_sig

from ..problem import NavierProblem
from ..preproc import rhs_line_const
from .loadgroup import LoadGroup
from .mc import _coeffs_line_load_1d_mc, _coeffs_line_load_2d_mc


__all__ = ["LineLoad"]


class LineLoad(LoadGroup):
    """
    A class to handle loads over lines for both beam and plate problems.

    Parameters
    ----------
    x: Iterable
        The point of application as an 1d iterable for a beam, a 2d iterable
        for a plate. In the latter case, the first row is the first point, the
        second row is the second point.
    v: Iterable
        Load intensities for each dof. The order of the dofs for a beam
        is [F, M], for a plate it is [F, Mx, My]. The provided values can either
        be numbers or strings of arbitrary expressions.

    Notes
    -----
    1) If the value for a DOF if deined using a string expression, it's domain
    is the domain of the problem at hand.
    2) If any of the DOFs is defined using a string expression, the instance cannot
    be used in bulk mode and can only stand for one single load.

    Examples
    --------
    >>> from sigmaepsilon.fourier import LoadGroup, LineLoad

    For a one dimensional problem:

    >>> loads = LoadGroup(
    >>>     LC1=LineLoad(x=[0, L], v=[1.0, 0.0]),
    >>>     LC2=LineLoad(x=[L / 2, L], v=[0.0, 1.0]),
    >>>     LC3=LineLoad(x=[L/2, L], v=['3*x + x**2', 0]),
    >>> )
    """

    _typestr_ = "line"

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

    def rhs(self, problem: NavierProblem = None) -> ndarray:
        """
        Returns the coefficients as a NumPy array.

        Parameters
        ----------
        problem: :class:`~sigmaepsilon.fourier.problem.NavierProblem`, Optional
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

            L = p.length
            N_monte_carlo = 1000

            if isinstance(v[0], Number) and isinstance(v[1], Number):
                v = np.array(v, dtype=float)
                return rhs_line_const(p.length, p.N, v, x)
            else:
                rhs = np.zeros((1, p.N, 2), dtype=x.dtype)

                if isinstance(v[0], str):
                    fnc = lambda x, i: np.sin(x * np.pi * i / L)
                    rhs[0, :, 0] = _coeffs_line_load_1d_mc(
                        v[0], x, p, fnc, N_monte_carlo
                    )
                elif isinstance(v[0], Number):
                    _v = np.array([v[0], 0], dtype=float)
                    rhs[0, :, 0] = rhs_line_const(p.length, p.N, _v, x)[0, :, 0]
                else:
                    raise TypeError("Invalid load type at position 0.")

                if isinstance(v[1], str):
                    fnc = lambda x, i: np.cos(x * np.pi * i / L)
                    rhs[0, :, 1] = _coeffs_line_load_1d_mc(
                        v[1], x, p, fnc, N_monte_carlo
                    )
                elif isinstance(v[1], Number):
                    _v = np.array([0, v[1]], dtype=float)
                    rhs[0, :, 1] = rhs_line_const(p.length, p.N, _v, x)[0, :, 1]
                else:
                    raise TypeError("Invalid load type at position 1.")

                return rhs
        elif len(x.shape) == 2:
            assert len(v) == 3, f"Invalid shape {v.shape} for load intensities."

            Lx, Ly = p.size
            M, N = p.shape
            N_monte_carlo = 1000

            rhs = np.zeros((1, M * N, 3), dtype=x.dtype)

            fnc = lambda x, ij: np.sin(x[:, 0] * np.pi * ij[0] / Lx) * np.sin(
                x[:, 1] * np.pi * ij[1] / Ly
            )
            if isinstance(v[0], str):
                str_expr = v[0]
            elif isinstance(v[0], Number):
                str_expr = str(v[0])
            else:
                raise TypeError("Invalid load type at position 0.")

            rhs[0, :, 0] = _coeffs_line_load_2d_mc(str_expr, x, p, fnc, N_monte_carlo)

            fnc = lambda x, ij: np.sin(x[:, 0] * np.pi * ij[0] / Lx) * np.cos(
                x[:, 1] * np.pi * ij[1] / Ly
            )
            if isinstance(v[1], str):
                str_expr = v[1]
            elif isinstance(v[1], Number):
                str_expr = str(v[1])
            else:
                raise TypeError("Invalid load type at position 1.")

            rhs[0, :, 1] = _coeffs_line_load_2d_mc(str_expr, x, p, fnc, N_monte_carlo)

            fnc = lambda x, ij: np.cos(x[:, 0] * np.pi * ij[0] / Lx) * np.sin(
                x[:, 1] * np.pi * ij[1] / Ly
            )
            if isinstance(v[2], str):
                str_expr = v[2]
            elif isinstance(v[2], Number):
                str_expr = str(v[2])
            else:
                raise TypeError("Invalid load type at position 2.")

            rhs[0, :, 2] = _coeffs_line_load_2d_mc(str_expr, x, p, fnc, N_monte_carlo)

            return rhs
        else:
            raise ValueError

    def __repr__(self):
        return "LineLoad(%s)" % (dict.__repr__(self))
