# -*- coding: utf-8 -*-
import unittest
from sigmaepsilon.math.optimize import BinaryGeneticAlgorithm


def Rosenbrock(a, b, x, y):
    return (a-x)**2 + b*(y-x**2)**2


class TestBGA(unittest.TestCase):

    def test_Rosenbrock(self):
        def f(x):
            return Rosenbrock(1, 100, x[0], x[1])
        f.dimension = 2
        ranges = [
            [-10, 10],
            [-10, 10]
        ]
        BGA = BinaryGeneticAlgorithm(f, ranges, length=12, nPop=200)
        BGA.solve()
        
        
if __name__ == "__main__":
        
    unittest.main()