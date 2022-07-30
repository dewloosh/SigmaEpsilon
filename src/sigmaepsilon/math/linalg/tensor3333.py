# -*- coding: utf-8 -*-
import sympy as sy
import numpy as np

from .tensor import Tensor
from ._tensop import tr_3333, tr_3333_jit


class Tensor3333(Tensor):

    __imap__ = {0: (0, 0), 1: (1, 1), 2: (2, 2),
                3: (1, 2), 4: (0, 2), 5: (0, 1)}

    def __init__(self, *args, symbolic=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.symmetric = True

        imap = kwargs.get('imap', None)
        if isinstance(imap, dict):
            self.__imap__ = imap

        if 'sympy' in args or symbolic:
            self._transform = self._transform_sym
            self.dtype = object

        self.collapsed = None
        if self._array is not None:
            self.collapsed = len(self._array.shape) == 2

    def expand(self):
        """
        Governs how a tensor is transformed from partial to full 
        (4d tensor) representation.
        """
        if not self.collapsed:
            return self
        T = np.zeros((3, 3, 3, 3), dtype=self._array.dtype)
        m = self._array
        imap = self.imap()
        for ij, ijkl in imap.items():
            T[ijkl] = m[ij]
        self._array = T
        self.collapsed = False
        return self

    def collapse(self):
        """
        Governs how a tensor is transformed from full (4d tensor) to
        partial representation.
        """
        if self.collapsed:
            return self
        m = np.zeros((6, 6), dtype=self._array.dtype)
        T = self._array
        imap = self.imap()
        for ij, ijkl in imap.items():
            m[ij] = T[ijkl]
        self._array = m
        self.collapsed = True
        return self

    @classmethod
    def imap(cls, imap1d=None, *args, **kwargs):
        """
        Returns a 2d-to-4d index map used to collapse or expand a tensor,
        based on the 1d-to-2d mapping of the class the function is called on,
        or on the first argument, if it is a suitable candidate for an
        index map.
        """
        if imap1d is None:
            imap1d = cls.__imap__
        indices = np.indices((6, 6))
        it = np.nditer([*indices], ['multi_index'])
        imap2d = dict()
        for _ in it:
            i, j = it.multi_index
            imap2d[(i, j)] = imap1d[i] + imap1d[j]
        return imap2d

    @classmethod
    def symbolic(cls, *args, base='C_', as_matrix=False, imap=None,
                 **kwargs):
        """
        Returns a symbolic representation of a 4th order 3x3x3x3 tensor.
        If the argument 'as_matrix' is True, the function returns a 6x6 matrix,
        that unfolds according to the argument 'imap', or if it's not provided,
        the index map of the class the function is called on. If 'imap' is
        provided, it must be a dictionary including exactly 6 keys and
        values. The keys must be integers in the integer range (0, 6), the
        values must be tuples on the integer range (0, 3).
        The default mapping is

                0 : (0, 0) --> normal stress x
                1 : (1, 1) --> normal stress y
                2 : (2, 2) --> normal stress z
                3 : (1, 2) --> shear stress yz
                4 : (0, 2) --> shear stress xz
                5 : (0, 1) --> shear stress xy

        and it means the classical Voigt unfolding of the tensor indices.
        """
        res = np.zeros((3, 3, 3, 3), dtype=object)
        indices = np.indices((3, 3, 3, 3))
        it = np.nditer([*indices], ['multi_index'])
        for _ in it:
            p, q, r, s = it.multi_index
            if q >= p and s >= r:
                sinds = np.array([p, q, r, s], dtype=np.int16) + 1
                sym = sy.symbols(base + '_'.join(sinds.astype(str)))
                res[p, q, r, s] = sym
                res[q, p, r, s] = sym
                res[p, q, s, r] = sym
                res[q, p, s, r] = sym
                res[r, s, p, q] = sym
                res[r, s, q, p] = sym
                res[s, r, p, q] = sym
                res[s, r, q, p] = sym
        if as_matrix:
            mat = np.zeros((6, 6), dtype=object)
            imap = cls.imap(imap) if imap is None else imap
            for ij, ijkl in imap.items():
                mat[ij] = res[ijkl]
            if 'sympy' in args:
                res = sy.Matrix(mat)
            else:
                res = mat
        return res

    def _transform(self, dcm: np.ndarray):
        """
        Returns the components of the transformed numerical tensor, based on
        the provided direction cosine matrix.
        """
        if self.collapsed:
            self.expand()
            array = tr_3333_jit(self._array, dcm)
            self.collapse()
        else:
            array = tr_3333_jit(self._array, dcm)
        return array

    def _transform_sym(self, dcm: np.ndarray):
        """
        Returns the components of the transformed symbolic tensor, based on
        the provided direction cosine matrix.
        """
        if self.collapsed:
            self.expand()
            array = tr_3333(self._array, dcm, dtype=object)
            self.collapse()
        else:
            array = tr_3333(self._array, dcm, dtype=object)
        return array


class ComplianceTensor(Tensor3333):

    def __init__(self, *args, imap=None, **kwargs):
        super().__init__(*args, imap=imap, **kwargs)


if __name__ == '__main__':
    from dewloosh.math.linalg.frame import ReferenceFrame

    A = ReferenceFrame()
    B = A.orient_new('Body', [0, 0, 90*np.pi/180], 'XYZ')

    tA = Tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]], frame=A)
    tA.transform_to_frame(B)

    C = Tensor3333(np.zeros((3, 3, 3, 3)), frame=A)
    C.collapse()
    C.orient(B)