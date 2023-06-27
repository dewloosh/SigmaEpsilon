from abc import abstractmethod


class NavierProblem:
    """
    Base class for Navier problems.
    """

    @abstractmethod
    def solve(self, *args, **kwargs):
        ...


class NavierPlateProblem(NavierProblem):
    """
    Base class for Navier plate problems.
    """


class NavierBeamProblem(NavierProblem):
    """
    Base class for Navier beam problems.
    """
