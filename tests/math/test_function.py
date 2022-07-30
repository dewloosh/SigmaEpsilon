# -*- coding: utf-8 -*-
import unittest
import sympy as sy
from sympy import symbols
import numpy as np

from sigmaepsilon.math.function import VariableManager, Function
from sigmaepsilon.math.function import Equality, InEquality


class TestFunction(unittest.TestCase):

    def test_linearity(self):
        def f0(x=None, y=None):
            return x**2 + y

        def f1(x=None, y=None):
            return np.array([2*x, 1])

        def f2(x=None, y=None):
            return np.array([[0, 0], [0, 0]])
        f = Function(f0, f1, f2, d=2)
        assert f.linear


class TestRelations(unittest.TestCase):

    def test_InEquality(self):
        gt = InEquality('x + y', op='>')
        assert not gt.relate([0.0, 0.0])

        ge = InEquality('x + y', op='>=')
        assert ge.relate([0.0, 0.0])

        le = InEquality('x + y', op=lambda x, y: x <= y)
        assert le.relate([0.0, 0.0])

        lt = InEquality('x + y', op=lambda x, y: x < y)
        assert not lt.relate([0.0, 0.0])


if __name__ == "__main__":

    unittest.main()
