# -*- coding: utf-8 -*-
import sympy as sy
from sympy import latex
import numpy as np


def gen_Lagrange_1d(N, x=None, *args, xsym:str=r'\xi', fsym:str=r'\phi', sym=False):
    """
    Generates Lagrange polynomials and their derivatives up to 3rd for appeoximation 
    in 1d space, based on N input pairs of position and value. Geometrical parameters 
    can be numeric or symbolic.
    
    Parameters
    ----------
    N : int or List[int]
        The number of data points or a list of indices. If it is a single integer,
        indices are assumed as [1, ..., N], but this is only relevant for the symbolic
        representation, not the calculation itself, which only cares about the number
        of data points, regardless of their actual indices.
        
    x : Iterable, Optional
        The locations of the data points. If not specified and `sym=False`, a range 
        of [-1, 1] is assumed and the locations are generated as `np.linspace(-1, 1, N)`, 
        where N is the number of data points. If `sym=True`, the calculation is entirely
        symbolic. Default is None.
        
    xsym : str, Optional
        Symbol of the variable in the symbolic representation of the generated functions.
        Default is :math:`$\xi$`.
        
    fsym : str, Optional
        Symbol of the function in the symbolic representation of the generated functions.
        Default is 'f'.
        
    Returns
    -------
    dict
        A dictionary containing the generated functions for the reuested nodes.
        The keys of the dictionary are the indices of the points, the values are
        dictionaries with the following keys and values:
        
            symbol : the symbol of the function
            
            0 : the function
            
            1 : the first derivative
            
            2 : the second derivative
            
            3 : the third derivative
    
    Notes
    -----
    Inversion of a heavily symbolic matrix may take quite some time, and is not suggested
    for N > 3. This is why isoparametric finite elements make sense. Fixing the locations
    as constant real numbers symplifies the process and makes the solution much faster.
    
    """
    module_data = {}
    xvar = sy.symbols('x')
    inds = list(range(1, N + 1)) if isinstance(N, int) else  N
    N = len(inds)
    def var_tmpl(i): return r'\Delta_{}'.format(i)
    def var_str(i): return var_tmpl(inds[i])
    coeffs = sy.symbols(', '.join(['c_{}'.format(i+1) for i in range(N)]))
    variables = sy.symbols(', '.join([var_str(i) for i in range(N)]))
    if x is None:
        if xsym is None or not sym:
            x = np.linspace(-1, 1, N)
        else:
            x = sy.symbols(
                ', '.join([xsym + '_{}'.format(i+1) for i in range(N)]))
    poly = sum([c * xvar**i for i, c in enumerate(coeffs)])
    #
    evals = [poly.subs({'x': x[i]}) for i in range(N)]
    A = sy.zeros(N, N)
    for i in range(N):
        A[i, :] = sy.Matrix([evals[i].coeff(c) for c in coeffs]).T
    coeffs_new = A.inv() * sy.Matrix(variables)
    subs = {coeffs[i]: coeffs_new[i] for i in range(N)}
    approx = poly.subs(subs).simplify().expand()
    #
    shp = [approx.coeff(v).factor().simplify() for v in variables]
    #
    def diff(fnc): return fnc.diff(xvar).expand().simplify().factor().simplify()
    dshp1 = [diff(fnc) for fnc in shp]
    dshp2 = [diff(fnc) for fnc in dshp1]
    dshp3 = [diff(fnc) for fnc in dshp2]
    #
    for i, ind in enumerate(inds):
        module_data[ind] = {}
        fnc_str = latex(sy.symbols(fsym + '_{}'.format(ind)))
        module_data[ind]['symbol'] = fnc_str
        module_data[ind][0] = shp[i]
        module_data[ind][1] = dshp1[i]
        module_data[ind][2] = dshp2[i]
        module_data[ind][3] = dshp3[i]
        
        
def gen_Hermite_Bernoulli(inds, *args, xsym='x', fsym='\phi', N: int = None, sign=1, positions=None, **kwargs):
    module_data = {}
    x, L = sy.symbols('x L')
    nC = 2 * N
    def var_tmpl(i): return '\\Delta_{}'.format(i)
    def var_str(i): return var_tmpl(inds[i])
    coeffs = sy.symbols(', '.join(['c_{}'.format(i+1) for i in range(nC)]))
    variables = sy.symbols(', '.join([var_str(i) for i in range(nC)]))
    if positions is None:
        positions = sy.symbols(
            ', '.join([xsym + '_{}'.format(i+1) for i in range(N)]))
    poly = sum([c * x**i for i, c in enumerate(coeffs)])
    poly = poly.expand()
    dpoly = sign * poly.diff(x, 1) * 2/L
    dpoly = dpoly.expand()
    #
    evals = [poly.subs({'x': positions[i]}) for i in range(N)]
    devals = [dpoly.subs({'x': positions[i]}) for i in range(N)]
    A = sy.zeros(nC, nC)
    c = 0
    for i in range(N):
        A[c, :] = sy.Matrix([evals[i].coeff(c) for c in coeffs]).T
        A[c + 1, :] = sy.Matrix([devals[i].coeff(c) for c in coeffs]).T
        c += 2
    coeffs_new = A.inv() * sy.Matrix(variables)
    subs = {coeffs[i]: coeffs_new[i] for i in range(2*N)}
    approx = poly.subs(subs).simplify().expand()
    #
    shp = [approx.coeff(v).factor().simplify() for v in variables]
    #
    def diff(fnc): return fnc.diff(x).expand().simplify().factor().simplify()
    dshp1 = [diff(fnc) for fnc in shp]
    dshp2 = [diff(fnc) for fnc in dshp1]
    dshp3 = [diff(fnc) for fnc in dshp2]
    #
    for i, ind in enumerate(inds):
        module_data[ind] = {}
        fnc_str = latex(sy.symbols('\phi_{}'.format(ind)))
        module_data[ind]['symbol'] = fnc_str
        module_data[ind][0] = shp[i]
        module_data[ind][1] = dshp1[i]
        module_data[ind][2] = dshp2[i]
        module_data[ind][3] = dshp3[i]
