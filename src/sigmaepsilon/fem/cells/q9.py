# -*- coding: utf-8 -*-
from dewloosh.math.numint import GaussPoints as Gauss

from dewloosh.mesh.cells import Q9 as Quadrilateral

from ..elem import FiniteElement
from ..model.membrane import Membrane
from ..model.plate import Plate


class Q9M(Quadrilateral, Membrane, FiniteElement):

    qrule = 'full'
    quadrature = {
        'full': Gauss(3, 3),
        'selective': {
            (0, 1): 'full',
            (2,): 'reduced'
        },
        'reduced': Gauss(2, 2)
    }


class Q9P(Quadrilateral, Plate, FiniteElement):

    qrule = 'full'
    quadrature = {
        'full': Gauss(3, 3),
        'selective': {
            (0, 1, 2): 'full',
            (3, 4): 'reduced'
        },
        'reduced': Gauss(2, 2)
    }
