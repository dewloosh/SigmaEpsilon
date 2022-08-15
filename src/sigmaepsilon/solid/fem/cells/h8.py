# -*- coding: utf-8 -*-
from neumann.numint import GaussPoints as Gauss

from polymesh.cells import H8 as HexaHedron

from ..model.solid3d import Solid3d

from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC


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
