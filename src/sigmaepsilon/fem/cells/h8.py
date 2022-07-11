# -*- coding: utf-8 -*-
from dewloosh.math.numint import GaussPoints as Gauss

from dewloosh.mesh.cells import H8 as HexaHedron

from dewloosh.solid.fem.elem import FiniteElement
from dewloosh.solid.fem.meta import ABCFiniteElement as ABC

from ..model.solid3d import Solid3d


class H8(ABC, Solid3d, HexaHedron, FiniteElement):

    qrule = 'selective'
    quadrature = {
        'full': Gauss(2, 2, 2),
        'selective': {
            (0, 1, 2): 'full',
            (3, 4, 5): 'reduced'
        },
        'reduced': Gauss(1, 1, 1)
    }
