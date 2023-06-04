from numpy import ndarray

from neumann.linalg import Tensor4
from neumann.linalg.exceptions import TensorShapeMismatchError

from ...utils.material.imap import _map_3x3x3x3_to_6x6, _map_6x6_to_3x3x3x3
from ...utils.material.hooke import _has_elastic_params, elastic_stiffness_matrix


__all__ = ["ElasticityTensor"]


class ElasticityTensor(Tensor4):
    """
    A class to represent the 4th order stiffness tensor.
    """
        
    def __init__(self, *args, **kwargs):
        
        if len(args) > 0 and isinstance(args[0], dict):
            if _has_elastic_params(args[0]):
                args = (elastic_stiffness_matrix(args[0]),)
        elif _has_elastic_params(**kwargs):
            args = (elastic_stiffness_matrix(**kwargs),)
        
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
                    
                arr = _map_6x6_to_3x3x3x3(arr)
                
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
            return (
                (len(arr.shape) >= 5 and arr.shape[-4:] == (3, 3, 3, 3)) or
                (len(arr.shape) >= 3 and arr.shape[-2:] == (6, 6))
            )
        else:
            return (
                (len(arr.shape) == 4 and arr.shape[-4:] == (3, 3, 3, 3)) or
                (len(arr.shape) == 2 and arr.shape[-2:] == (6, 6))
            )
                                    
    def contracted_components(self, *args, **kwargs) -> ndarray:
        """
        Returns the 2d matrix representation of the tensor.
        """
        if (len(args) + len(kwargs)) > 0:
            arr = self.show(*args, **kwargs)
        else:
            arr = self.array
        return _map_3x3x3x3_to_6x6(arr)
    
    def transpose(self, *_, **__) -> "ElasticityTensor":
        """
        Returns the instance itself without modification regardless
        of the parameters (since the object is symmetric).
        """
        return self
    