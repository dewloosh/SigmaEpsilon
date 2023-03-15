import numpy as np
from typing import Tuple, Union, Iterable

from linkeddeepdict import LinkedDeepDict
from neumann import atleast2d

from .problem import NavierProblem
from .loads import LoadGroup, NavierLoadError
from .preproc import lhs_Navier
from .postproc import postproc
from .proc import linsolve_Kirchhoff, linsolve_Mindlin


class RectangularPlate(NavierProblem):
    """
    A class to handle semi-analytic solutions of rectangular plates with
    specific boudary conditions.

    Parameters
    ----------
    size : Tuple[float]
        The size of the rectangle.
    shape : Tuple[int]
        Numbers of harmonic terms involved in both directions.
    """

    def __init__(
        self,
        size: Tuple[float],
        shape: Tuple[int],
        *,
        D11: float = None,
        D12: float = None,
        D22: float = None,
        D66: float = None,
        S44: float = None,
        S55: float = None,
        D: Iterable = None,
        S: Iterable = None,
    ):
        super().__init__()
        self.size = np.array(size, dtype=float)
        self.shape = np.array(shape, dtype=int)

        if D11 and D12 and D22 and D66:
            D = np.zeros((3, 3))
            D[0, 0] = self.D11
            D[0, 1] = self.D12
            D[1, 0] = self.D12
            D[1, 1] = self.D22
            D[2, 2] = self.D66
            self.D = D
        else:
            self.D = np.array(D, dtype=float)

        if S44 and S55:
            S = np.zeros((2, 2))
            S[0, 0] = self.S44
            S[1, 1] = self.S55
            self.S = S
        elif S is not None:
            self.S = np.array(S, dtype=float)
        else:
            self.S = None

    def solve(self, loads: Union[dict, LoadGroup], points: Iterable) -> LinkedDeepDict:
        """
        Solves the problem and calculates all entities at the specified points.

        Parameters
        ----------
        loads : Union[dict, LoadGroup]
            The loads.
        points : Iterable
            2d float array of coordinates, where the results are to be evaluated.

        Returns
        -------
        dict
            A dictionary with a same layout as the input.
        """
        # STIFFNESS
        LHS = lhs_Navier(self.size, self.shape, D=self.D, S=self.S)

        # LOADS
        if isinstance(loads, LoadGroup):
            _loads = loads
        elif isinstance(loads, dict):
            _loads = LoadGroup.from_dict(loads)
        else:
            raise NavierLoadError()
        _loads.problem = self
        LC = list(_loads.cases())
        RHS = np.vstack(list(lc.rhs() for lc in LC))

        # SOLUTION
        if self.S is None:
            """_RHS = rhs_Kirchhoff(RHS, self.length)
            coeffs = linsolve_Kirchhoff(LHS, _RHS)
            del _RHS"""
            # (nLHS, nRHS, nMN)
        else:
            coeffs = linsolve_Mindlin(LHS, RHS)
            # (nLHS, nRHS, nMN, 3)

        # POSTPROCESSING
        points = atleast2d(points)
        res = postproc(self.size, self.shape, points, coeffs, RHS, self.D, self.S)
        # (nLHS, nRHS, nP, nX)
        result = LinkedDeepDict()
        for i, lc in enumerate(LC):
            result[lc.address] = res[0, i, :, :]
        result.lock()
        return result
