# -*- coding: utf-8 -*-
from dewloosh.math.numint import GaussPoints as Gauss

from dewloosh.mesh.cells import QuadraticLine as Line

from dewloosh.solid.fem.cells import FiniteElement, ABCFiniteElement as ABC

from .bernoulli import BernoulliBase as Bernoulli

from .gen.b3 import shape_function_values_bulk as shpB3, \
    shape_function_derivatives_bulk as dshpB3
    
    
__all__ = ['Bernoulli3']


class Bernoulli3(ABC, Bernoulli, Line, FiniteElement):

    qrule = 'full'
    quadrature = {
        'full': Gauss(6),
        'mass' : Gauss(8)
    }
    shpfnc = shpB3
    dshpfnc = dshpB3
