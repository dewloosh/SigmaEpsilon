from numpy import ndarray

from neumann.linalg import Tensor2
from neumann.linalg.tr import _tr_tensors2
from neumann.linalg.exceptions import TensorShapeMismatchError

from ..utils.material.imap import _map_6x1_to_3x3, _map_3x3_to_6x1


__all__ = ["Tensor2x3"]


class Tensor2x3(Tensor2):
    """
    A class to represent the 4th order stiffness tensor.

    Parameters
    ----------
    tensorial: bool, Optional
        Set this to True, if the tensor describes the relationship of stresses and
        tensorial strains. Default is False.
    symbolic: bool, Optional
        If True, the tensor is stored in symbolic form, and the components are stored as
        a `SymPy` matrix. Default is False.
    """

    def __init__(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], ndarray):
            arr = args[0]
            shape = arr.shape
            if shape[-1] == 6:
                if len(shape) >= 3:
                    is_bulk = kwargs.get("bulk", True)
                    if not is_bulk:
                        raise ValueError("Incorrect input!")
                    kwargs["bulk"] = is_bulk
                else:
                    if not len(shape) == 2:
                        raise TensorShapeMismatchError("Invalid shape!")
                    is_bulk = kwargs.get("bulk", False)
                    if is_bulk:
                        raise ValueError("Incorrect input!")
                arr = _map_6x1_to_3x3(arr)
            elif shape[-1] == 3:
                if len(shape) >= 5:
                    is_bulk = kwargs.get("bulk", True)
                    if not is_bulk:
                        raise ValueError("Incorrect input!")
                    kwargs["bulk"] = is_bulk
                else:
                    if not len(shape) == 4:
                        raise TensorShapeMismatchError("Invalid shape!")
                    is_bulk = kwargs.get("bulk", False)
                    if is_bulk:
                        raise ValueError("Incorrect input!")
            else:
                raise TensorShapeMismatchError("Invalid shape!")

            super().__init__(arr, *args[1:], **kwargs)
        else:
            super().__init__(*args, **kwargs)

    @classmethod
    def _verify_input(cls, arr: ndarray, *_, bulk: bool = False, **kwargs) -> bool:
        if bulk:
            return (len(arr.shape) >= 3 and arr.shape[-2:] == (3, 3)) or (
                len(arr.shape) >= 2 and arr.shape[-1] == 6
            )
        else:
            return (len(arr.shape) == 2 and arr.shape[-2:] == (3, 3)) or (
                len(arr.shape) == 1 and arr.shape[-1] == 6
            )

    def transform_components(self, dcm: ndarray) -> ndarray:
        """
        Returns the components of the transformed numerical tensor, based on
        the provided direction cosine matrix.
        """
        return _tr_tensors2(self.array, dcm)

    def contracted_components(self, *args, **kwargs) -> ndarray:
        """
        Returns the 1d representation of the tensor(s). The contraction
        is carried out using the Voigt index mapping.
        """
        if (len(args) + len(kwargs)) > 0:
            arr = self.show(*args, **kwargs)
        else:
            arr = self.array
        arr = _map_3x3_to_6x1(arr)
        return arr
