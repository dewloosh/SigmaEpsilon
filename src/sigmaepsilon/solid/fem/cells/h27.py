# -*- coding: utf-8 -*-
from neumann.numint import GaussPoints as Gauss

from polymesh.cells import H27 as HexaHedron

from ..model.solid3d import Solid3d

from .elem import FiniteElement
from .metaelem import ABCFiniteElement as ABC


class H27(ABC, Solid3d, HexaHedron, FiniteElement):

    qrule = 'full'
    quadrature = {
        'full': Gauss(3, 3, 3),
        'selective': {
            (0, 1, 2): 'full',
            (3, 4, 5): 'reduced'
        },
        'reduced': Gauss(2, 2, 2)
    }
