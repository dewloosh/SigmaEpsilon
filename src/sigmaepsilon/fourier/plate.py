import numpy as np
from numpy import swapaxes as swap
from typing import Tuple, Union, Iterable

from linkeddeepdict import LinkedDeepDict

from .problem import NavierProblem
from .loads import LoadGroup, NavierLoadError
from .preproc import lhs_Navier
from .postproc import postproc
from .proc import linsolve


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
        S55: float = None
    ):
        super().__init__()
        self.size = np.array(size, dtype=float)
        self.shape = np.array(shape, dtype=int)
        self.D11 = D11
        self.D12 = D12
        self.D22 = D22
        self.D66 = D66
        self.S44 = S44
        self.S55 = S55

    def solve(self, loads: Union[dict, LoadGroup], points: Iterable):
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
        D = np.zeros((3, 3))
        D[0, 0] = self.D11
        D[0, 1] = self.D12
        D[1, 0] = self.D12
        D[1, 1] = self.D22
        D[2, 2] = self.D66
        if self.S44 is not None:
            S = np.zeros((2, 2))
            S[0, 0] = self.S44
            S[1, 1] = self.S55
        else:
            S = None
        LHS = lhs_Navier(self.size, self.shape, D, S, squeeze=False)

        # LOADS
        if isinstance(loads, LoadGroup):
            _loads = loads
        elif isinstance(loads, dict):
            _loads = LoadGroup.from_dict(loads)
        else:
            raise NavierLoadError()
        _loads.problem = self
        LC = list(_loads.cases())
        RHS = np.stack(list(lc.rhs() for lc in LC))

        # SOLUTION
        points = np.array(points)
        coeffs = linsolve(LHS, RHS, squeeze=False)
        res = postproc(self.size, self.shape, points, coeffs, RHS, D, S, squeeze=False)
        # (nRHS, nLHS, nP, nX)
        res = swap(res, 1, 2)
        # (nRHS, nP, nLHS, nX)
        result = LinkedDeepDict()
        for i, lc in enumerate(_loads.cases()):
            result[lc.address] = np.squeeze(res[i, :, :])
        result.lock()
        return result
