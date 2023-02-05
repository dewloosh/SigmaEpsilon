import numpy as np
from typing import Union, Iterable

from linkeddeepdict import LinkedDeepDict
from neumann import atleast1d

from .problem import NavierProblem
from .loads import LoadGroup, NavierLoadError
from .preproc import lhs_Navier, rhs_Bernoulli
from .postproc import postproc
from .proc import linsolve_Bernoulli, linsolve_Timoshenko


class NavierBeam(NavierProblem):
    """
    A class to handle simply-supported plates and their solution
    using Navier's approach.

    Parameters
    ----------
    length : float
        The length of the beam.
    N : int, Optional
        The number of harmonic terms involved in the approximation.
        Default is 100.
    EI : float
        Bending stiffness. Default is None.
    GA : float, Optional
        Shear stiffness. Only for Timoshenko beams. Default is None.

    Examples
    --------
    To define an Euler-Bernoulli beam of length 10.0 and
    bending stiffness 2000.0 with 100 harmonic terms involved:

    >>> from sigmaepsilon.fourier import NavierBeam
    >>> beam = NavierBeam(10.0, 100, EI=2000.0)

    To define a Timoshenko beam of length 10.0, bending stiffness
    2000.0 and shear stiffness 1500.0 with 100 harmonic terms involved:

    >>> from sigmaepsilon.fourier import NavierBeam
    >>> beam = NavierBeam(10.0, 100, EI=2000.0, GA=1500.0)

    """

    def __init__(
        self, length: float, N: int = 100, *, EI: float = None, GA: float = None
    ):
        super().__init__()
        self.length = length
        self.EI = EI
        self.GA = GA
        self.N = N

    def solve(
        self, loads: Union[dict, LoadGroup], points: Union[float, Iterable]
    ) -> LinkedDeepDict:
        """
        Solves the problem and calculates all postprocessing quantities at
        one ore more points.

        Parameters
        ----------
        loads : Union[dict, LoadGroup]
            The loads. If it is an array, it should be a 2d float array.
        points : float or Iterable
            A float or an 1d iterable of coordinates, where the results are
            to be evaluated. If it is a scalar, the resulting dictionary
            contains 1d arrays for every quantity, for every load case. If
            there are multiple points, the result attached to a load case is
            a 2d array, where the first axis goes along the points.

        Returns
        -------
        LinkedDeepDict
            A nested dictionary with a same layout as the input.
        """
        # STIFFNESS
        LHS = lhs_Navier(self.length, self.N, D=self.EI, S=self.GA)

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
        if self.GA is None:
            _RHS = rhs_Bernoulli(RHS, self.length)
            coeffs = linsolve_Bernoulli(LHS, _RHS)
            del _RHS
            # (nLHS, nRHS, nMN)
        else:
            coeffs = linsolve_Timoshenko(LHS, RHS)
            # (nLHS, nRHS, nMN, 2)

        # POSTPROCESSING
        points = atleast1d(points)
        res = postproc(self.length, self.N, points, coeffs, RHS, self.EI, self.GA)
        # (nLHS, nRHS, nP, nX)
        result = LinkedDeepDict()
        for i, lc in enumerate(LC):
            result[lc.address] = res[0, i, :, :]
        result.lock()
        return result
