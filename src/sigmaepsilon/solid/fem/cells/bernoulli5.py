# -*- coding: utf-8 -*-
from neumann.numint import GaussPoints as Gauss
from polymesh.cells import QuadraticLine as Line

from .bernoulli import BernoulliBase as Bernoulli
from .gen.b5 import shape_function_values_bulk as shpB, \
    shape_function_derivatives_bulk as dshpB
    
from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC
    
    
__all__ = ['Bernoulli5']


class Bernoulli4(ABC, Bernoulli, Line, FiniteElement):
    """
    Finite element class to handle 5-noded Bernoulli beams.
    """

    qrule = 'full'
    quadrature = {
        'full': Gauss(10),
        'mass' : Gauss(12)
    }
    shpfnc = shpB
    dshpfnc = dshpB
