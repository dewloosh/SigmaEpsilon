# -*- coding: utf-8 -*-
import unittest
import numpy as np

from sigmaepsilon.math.array import atleast2d
from sigmaepsilon.math.function import Function, Equality, InEquality
from sigmaepsilon.math.optimize import LinearProgrammingProblem as LPP, \
    DegenerateProblemError, NoSolutionError
import sympy as sy

       
class TestLPP(unittest.TestCase):

    def test_unique_solution(self):
        x1, x2 = sy.symbols(['x1', 'x2'], positive=True)
        syms = [x1, x2]
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op='>=', variables=syms)
        ieq2 = InEquality(x2 - 1, op='>=', variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op='<=', variables=syms)
        lpp = LPP(cost=f, constraints=[ieq1, ieq2, ieq3], variables=syms)
        x = atleast2d(lpp.solve()['x'])
        _x = np.array([1.0, 1.0])
        assert np.all(np.isclose(_x, x))
    
    def test_degenerate_solution(self):
        variables = ['x1', 'x2', 'x3', 'x4']
        x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
        obj2 = Function(3*x1 + x2 + 9*x3 + x4, variables=syms)
        eq21 = Equality(x1 + 2*x3 + x4, variables=syms)
        eq22 = Equality(x2 + x3 - x4 - 2, variables=syms)
        P2 = LPP(cost=obj2, constraints=[eq21, eq22], variables=syms)
        try:
            P2.solve(raise_errors=True)
        except DegenerateProblemError:
            pass
        except Exception:
            assert False
                
    def test_no_solution(self):
        """
        Example for no solution.
        """
        variables = ['x1', 'x2', 'x3', 'x4']
        x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
        obj3 = Function(-3*x1 + x2 + 9*x3 + x4, variables=syms)
        eq31 = Equality(x1 - 2*x3 - x4 + 2, variables=syms)
        eq32 = Equality(x2 + x3 - x4 - 2, variables=syms)
        P3 = LPP(cost=obj3, constraints=[eq31, eq32], variables=syms)
        try:
            P3.solve(raise_errors=True)
        except NoSolutionError:
            pass
        except Exception:
            assert False
        
    def test_multiple_solution(self):
        """
        Example for multiple solutions.
        (0, 1, 1, 0)
        (0, 4, 0, 2)
        """
        variables = ['x1', 'x2', 'x3', 'x4']
        x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
        obj4 = Function(3*x1 + 2*x2 + 8*x3 + x4, variables=syms)
        eq41 = Equality(x1 - 2*x3 - x4 + 2, variables=syms)
        eq42 = Equality(x2 + x3 - x4 - 2, variables=syms)
        P4 = LPP(cost=obj4, constraints=[eq41, eq42], variables=syms)
        x = P4.solve(return_all=True, raise_errors=True)['x']
        assert len(x.shape) == 2
        assert x.shape[0] == 2
        
    def test_1(self):
        variables = ['x1', 'x2']
        x1, x2 = syms = sy.symbols(variables, positive=True)
        f = Function(x1 + x2, variables=syms)
        eq = Equality(x1 - 2, variables=syms)
        lpp = LPP(cost=f, constraints=[eq], variables=syms)
        x = lpp.solve(return_all=True, raise_errors=True)['x']
        assert np.all(np.isclose(x, np.array([2., 0.])))
        variables = ['x1', 'x2']
        x1, x2 = syms = sy.symbols(variables, positive=True)
        f = Function(x1 + x2, variables=syms)
        eq = Equality(x2 - 2, variables=syms)
        lpp = LPP(cost=f, constraints=[eq], variables=syms)
        x = lpp.solve(return_all=True, raise_errors=True)['x']
        assert np.all(np.isclose(x, np.array([0., 2.])))
        
    def test_2(self):
        x1, x2 = sy.symbols(['x1', 'x2'], positive=True)
        syms = [x1, x2]
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op='>=', variables=syms)
        ieq2 = InEquality(x2 - 1, op='>=', variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op='<=', variables=syms)
        lpp = LPP(cost=f, constraints=[ieq1, ieq2, ieq3], variables=syms)
        x = lpp.solve(return_all=True, raise_errors=True)['x']
        assert np.all(np.isclose(x, np.array([1., 1.])))
        x1, x2 = sy.symbols(['x1', 'x2'], positive=True)
        syms = [x1, x2]
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op='>=', variables=syms)
        ieq2 = InEquality(x2 - 1, op='>=', variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op='>=', variables=syms)
        lpp = LPP(cost=f, constraints=[ieq1, ieq2, ieq3], variables=syms)
        e = lpp.solve(return_all=True, raise_errors=True, as_dict=True)['e']
        assert len(e) == 0
    
    
if __name__ == "__main__":
        
    unittest.main()