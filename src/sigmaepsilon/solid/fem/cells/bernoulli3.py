# -*- coding: utf-8 -*-
from neumann.numint import GaussPoints as Gauss
from polymesh.cells import QuadraticLine as Line

from .bernoulli import BernoulliBase as Bernoulli
from .gen.b3 import (shape_function_values_bulk, 
                     shape_function_derivatives_bulk)
    
from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC
    
    
__all__ = ['Bernoulli3']


class Bernoulli3(ABC, Bernoulli, Line, FiniteElement):
    """
    Finite element class to handle 3-noded Bernoulli beams.
    """

    qrule = 'full'
    quadrature = {
        'full': Gauss(6),
        'mass' : Gauss(8)
    }
    shpfnc = shape_function_values_bulk
    dshpfnc = shape_function_derivatives_bulk
