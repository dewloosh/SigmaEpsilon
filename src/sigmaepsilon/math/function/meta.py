# -*- coding: utf-8 -*-
from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify, derive_by_array, symbols, Expr
from sympy.core.numbers import One
from collections import OrderedDict

from ...core.abc import ABCMeta_Weak


class ABCMeta_MetaFunction(ABCMeta_Weak):
    """
    Metaclass for defining ABCs for algebraic structures.
    """

    def __new__(metaclass, name, bases, namespace, *args, **kwargs):
        cls = super().__new__(metaclass, name, bases, namespace,
                              *args, **kwargs)
        if 'value' in namespace:
            cls.f = namespace['value']
            
        if 'gradient' in namespace:
            cls.g = namespace['gradient']
            
        if 'Hessian' in namespace:
            cls.G = namespace['Hessian']
        
        return cls 


class MetaFunction(metaclass=ABCMeta_MetaFunction):
    __slots__ = ('f0', 'f1', 'f2', 'dimension', 'domain', 'expr',
                 'variables', 'vmap')

    def __call__(self, *args, **kwargs):
        return self.f0(*args, **kwargs)

    def f(self, *args, **kwargs):
        """
        Returns the function value.

        For this operation the object must have an implementation of

        value(self, *args, **kwargs):
            <...>
            return <...>
        """
        try:
            return self.f0(*args, **kwargs)
        except Exception:
            return None

    def g(self, *args, **kwargs):
        """
        Returns the gradient vector if available.

        For this operation the object must have an implementation of

        gradient(self, *args, **kwargs):
            <...>
            return <...>
        """
        try:
            return self.f1(*args, **kwargs)
        except Exception:
            return None

    def G(self, *args, **kwargs):
        """
        Returns the Hessian matrix if available.

        For this operation the object must have an implementation of

        Hessian(self,*args,**kwargs):
            <...>
            return <...>
        """

        try:
            return self.f2(*args, **kwargs)
        except Exception:
            return None

    @classmethod
    def _str_to_func(cls, str_expr: str, *args, **kwargs):
        return symbolize(*args, str_expr=str_expr, **kwargs)

    @classmethod
    def _sympy_to_func(cls, expr: Expr, *args, **kwargs):
        return symbolize(*args, expr=expr, **kwargs)


def decode(*args, expr=None, str_expr: str = None, variables=None,
           **kwargs):
    try:
        if str_expr is not None:
            expr = parse_expr(str_expr, evaluate=False)
        if not variables:
            variables = []
            """
            for arg in expr.args:
                syms = list(arg.free_symbols)
                if len(syms) == 1:
                    variables.append(syms[0])
            """
            variables = tuple(expr.free_symbols)
        else:
            try:
                variables = list(symbols(variables))
            except Exception:
                pass
        return expr, variables
    except Exception:
        return None, None


def symbolize(*args, **kwargs):
    expr, variables = decode(*args, **kwargs)
    f0 = lambdify([variables], expr, 'numpy')
    g = derive_by_array(expr, variables)
    f1 = lambdify([variables], g, 'numpy')
    G = derive_by_array(g, variables)
    f2 = lambdify([variables], G, 'numpy')
    return {'value': f0, 'gradient': f1,
            'Hessian': f2, 'd': len(variables),
            'variables': variables,
            'expr': expr}


def substitute(expr, values, variables=None, as_string=False):
    if variables is None:
        variables = tuple(expr.free_symbols)
    if not as_string:
        return expr.subs([(v, val) for v, val in zip(variables, values)])
    else:
        return expr.subs([(str(v), val) for v, val in zip(variables, values)])


def coefficients(expr=None, variables=None, normalize=False):
    try:
        if variables is None:
            variables = tuple(expr.free_symbols)
        d = OrderedDict({x: 0 for x in variables})
        d.update(expr.as_coefficients_dict())
        if not normalize:
            return d
        else:
            res = OrderedDict()
            for key, value in d.items():
                if len(key.free_symbols) == 0:
                    res[One()] = value*key
                else:
                    res[key] = value
            return res
    except Exception:
        return None


if __name__ == '__main__':

    str_expr = 'x*y + y**2 + 6*b + 2'
    d = decode(str_expr=str_expr)
