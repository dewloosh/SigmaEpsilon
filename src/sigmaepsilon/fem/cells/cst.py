# -*- coding: utf-8 -*-
import numpy as np

from dewloosh.mesh.cells import T3 as Triangle

from dewloosh.solid.fem.elem import FiniteElement
from dewloosh.solid.fem.meta import ABCFiniteElement as ABC

from ..model.membrane import Membrane
from ..model.plate import Plate



class CSTM(Triangle, Membrane, FiniteElement):
    """
    The constant-strain triangle (a.k.a., CST triangle, Turner triangle)
    """

    qrule = 'full'
    quadrature = {
        'full': (np.array([[1/3, 1/3]]), np.array([1/2])),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CSTP(Triangle, Plate, FiniteElement):
    """
    The constant-strain triangle (a.k.a., CST triangle, Turner triangle)
    """

    qrule = 'full'
    quadrature = {
        'full': (np.array([[1/3, 1/3]]), np.array([1/2])),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    pass
