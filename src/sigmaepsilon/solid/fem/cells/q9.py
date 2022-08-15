# -*- coding: utf-8 -*-
from neumann.numint import GaussPoints as Gauss

from polymesh.cells import Q9 as Quadrilateral

from ..model.membrane import Membrane
from ..model.plate import Plate

from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC

class Q9M(ABC, Membrane, Quadrilateral, FiniteElement):

    qrule = 'full'
    quadrature = {
        'full': Gauss(3, 3),
        'selective': {
            (0, 1): 'full',
            (2,): 'reduced'
        },
        'reduced': Gauss(2, 2)
    }


class Q9P(ABC, Plate, Quadrilateral, FiniteElement):

    qrule = 'selective'
    quadrature = {
        'full': Gauss(3, 3),
        'selective': {
            (0, 1, 2): 'full',
            (3, 4): 'reduced'
        },
        'reduced': Gauss(2, 2)
    }
