from abc import abstractmethod


class NavierProblem:
    """
    Base class for Navier problems. The sole reason of this class is to
    avoid circular referencing.
    """

    @abstractmethod
    def solve(self, *args, **kwargs):
        ...
