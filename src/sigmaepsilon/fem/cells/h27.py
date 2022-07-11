# -*- coding: utf-8 -*-
from dewloosh.math.numint import GaussPoints as Gauss

from dewloosh.mesh.cells import H27 as HexaHedron

from dewloosh.solid.fem.model.solid3d import Solid3d
from dewloosh.solid.fem.elem import FiniteElement


class H27(HexaHedron, Solid3d, FiniteElement):

    qrule = 'selective'
    quadrature = {
        'full': Gauss(3, 3, 3),
        'selective': {
            (0, 1, 2): 'full',
            (3, 4, 5): 'reduced'
        },
        'reduced': Gauss(2, 2, 2)
    }
