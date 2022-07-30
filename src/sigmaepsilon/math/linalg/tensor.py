# -*- coding: utf-8 -*-
import numpy as np

from ...core.tools.alphabet import latinrange

from .frame import ReferenceFrame as Frame
from .vector import Vector, VectorBase


class Tensor(Vector):

    def __init__(self, *args, symmetric=False, isotropic=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.symmetric = symmetric  # has no effect, just a reminder
        self.isotropic = isotropic  # has no effect, just a reminder

    @classmethod
    def identity(cls, dim=1):
        return cls(np.eye(dim))

    @staticmethod
    def eye(dim=1):
        return Tensor(np.eye(dim))

    def _transform(self, dcm: np.ndarray = None):
        Q = dcm.T
        dim = self.dim
        source = latinrange(dim, start='i')
        target = latinrange(dim, start=ord('i') + dim)
        command = ','.join([t + s for t, s in zip(target, source)]) + \
            ',' + ''.join(source)
        args = [Q for _ in range(dim)]
        return np.einsum(command, *args, self._array, optimize='greedy')

