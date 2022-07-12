# -*- coding: utf-8 -*-
from dewloosh.math.numint import GaussPoints as Gauss

from dewloosh.mesh.cells import L2 as Line

from dewloosh.solid.fem.cells import FiniteElement, ABCFiniteElement as ABC

from .bernoulli import BernoulliBase as Bernoulli

from .gen.b2 import shape_function_values_bulk as shpB2, \
    shape_function_derivatives_bulk as dshpB2


__all__ = ['Bernoulli2']


class Bernoulli2(ABC, Bernoulli, Line, FiniteElement):

    qrule = 'full'
    quadrature = {
        'full': Gauss(2),
        'selective': {
            (0, 1): 'full',
            (2): 'reduced'
        },
        'reduced': Gauss(1),
        'mass' : Gauss(4)
    }
    shpfnc = shpB2
    dshpfnc = dshpB2