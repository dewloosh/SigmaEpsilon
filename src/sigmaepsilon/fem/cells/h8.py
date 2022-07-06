# -*- coding: utf-8 -*-
from dewloosh.math.numint import GaussPoints as Gauss

from dewloosh.mesh.cells import H8 as HexaHedron

from ..model.solid3d import Solid3d
from ..elem import FiniteElement


class H8(HexaHedron, Solid3d, FiniteElement):

    qrule = 'selective'
    quadrature = {
        'full': Gauss(2, 2, 2),
        'selective': {
            (0, 1, 2): 'full',
            (3, 4, 5): 'reduced'
        },
        'reduced': Gauss(1, 1, 1)
    }
