# -*- coding: utf-8 -*-
import sympy as sy
from sympy import symbols, Matrix


def inv_sym_3x3(m: Matrix, as_adj_det=False):
    P11, P12, P13, P21, P22, P23, P31, P32, P33 = \
        symbols('P_{11} P_{12} P_{13} P_{21} P_{22} P_{23} P_{31} \
                P_{32} P_{33}', real=True)
    Pij = [[P11, P12, P13], [P21, P22, P23], [P31, P32, P33]]
    P = sy.Matrix(Pij)
    detP = P.det()
    adjP = P.adjugate()
    invP = adjP / detP
    subs = {s: r for s, r in zip(sy.flatten(P), sy.flatten(m))}
    if as_adj_det:
        return detP.subs(subs), adjP.subs(subs)
    else:
        return invP.subs(subs)
