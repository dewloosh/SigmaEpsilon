import numpy as np
from numpy import ndarray

from neumann.linalg import Tensor4x3
from neumann.linalg.top import tr_3333_jit


class ComplianceTensor(Tensor4x3):
    def __init__(self, *args, tensorial: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if self._array and not tensorial:
            collapsed = self.collapsed
            self.collapse()
            self._array = _compliance_to_tensorial(self._array)
            if not collapsed:
                self.expand()

    def transform_components(self, dcm: np.ndarray) -> ndarray:
        """
        Returns the components of the transformed numerical tensor,
        based on the provided direction cosine matrix.
        """
        if self.collapsed:
            self.expand()
            array = tr_3333_jit(self._array, dcm)
            self.collapse()
        else:
            array = tr_3333_jit(self._array, dcm)
        return array

    def engineering_components(self) -> ndarray:
        """
        Returns the engineering components.
        """
        collapsed = self.collapsed
        self.collapse()
        result = _compliance_from_tensorial(self._array)
        if not collapsed:
            self.expand()
        return result


def _compliance_to_tensorial(S: ndarray) -> ndarray:
    R_inv = np.diag([1, 1, 1, 1 / 2, 1 / 2, 1 / 2], dtype=float)
    return R_inv @ S


def _compliance_from_tensorial(S: ndarray) -> ndarray:
    R = np.diag([1, 1, 1, 2, 2, 2], dtype=float)
    return R @ S
