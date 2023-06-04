from .tensor2x3 import Tensor2x3

__all__ = ["CauchyStressTensor"]


class CauchyStressTensor(Tensor2x3):
    """
    A class to represent the 2nd order Cauchy stress tensor.
    """

    def transpose(self, *_, **__) -> "CauchyStressTensor":
        """
        Returns the instance itself without modification regardless
        of the parameters (since the object is symmetric).
        """
        return self
