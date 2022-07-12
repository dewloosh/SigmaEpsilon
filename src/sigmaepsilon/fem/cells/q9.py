# -*- coding: utf-8 -*-
from dewloosh.math.numint import GaussPoints as Gauss

from dewloosh.mesh.cells import Q9 as Quadrilateral

from dewloosh.solid.fem.cells import FiniteElement, ABCFiniteElement as ABC

from ..model.membrane import Membrane
from ..model.plate import Plate


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
