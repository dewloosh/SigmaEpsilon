from numpy import ndarray

from .tensor2x3 import Tensor2x3
from ..utils.material.imap import _map_3x3_to_6x1


__all__ = ["SmallStrainTensor"]


class SmallStrainTensor(Tensor2x3):
    """
    A class to represent the 2nd order small strain tensor.
    """

    def __init__(self, *args, tensorial: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if self._array is not None and not tensorial:
            _strain_to_tensorial(self._array)

    def contracted_components(
        self, *args, engineering: bool = True, **kwargs
    ) -> ndarray:
        """
        Returns the engineering components.
        """
        arr = super().contracted_components(*args, **kwargs)
        if engineering:
            _strain_to_engineering(arr)
        return arr

    def transpose(self, *_, **__) -> "SmallStrainTensor":
        """
        Returns the instance itself without modification regardless
        of the parameters (since the object is symmetric).
        """
        return self


def _strain_to_tensorial(S: ndarray):
    if S.shape[-1] == 6:
        S[..., -1] /= 2
        S[..., -2] /= 2
        S[..., -3] /= 2
    else:
        S[..., 0, 1] /= 2
        S[..., 0, 2] /= 2
        S[..., 1, 2] /= 2
        S[..., 1, 0] /= 2
        S[..., 2, 0] /= 2
        S[..., 2, 1] /= 2


def _strain_to_engineering(S: ndarray):
    if S.shape[-1] == 6:
        S[..., -1] *= 2
        S[..., -2] *= 2
        S[..., -3] *= 2
    else:
        S[..., 0, 1] *= 2
        S[..., 0, 2] *= 2
        S[..., 1, 2] *= 2
        S[..., 1, 0] *= 2
        S[..., 2, 0] *= 2
        S[..., 2, 1] *= 2
