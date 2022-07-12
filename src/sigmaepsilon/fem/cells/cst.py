# -*- coding: utf-8 -*-
import numpy as np

from dewloosh.mesh.cells import T3 as Triangle

from dewloosh.solid.fem.cells import FiniteElement, ABCFiniteElement as ABC

from ..model.membrane import Membrane
from ..model.plate import Plate



class CSTM(ABC, Membrane, Triangle, FiniteElement):
    """
    The constant-strain triangle (a.k.a., CST triangle, Turner triangle)
    """

    qrule = 'full'
    quadrature = {
        'full': (np.array([[1/3, 1/3]]), np.array([1/2])),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CSTP(ABC, Plate, Triangle, FiniteElement):
    """
    The constant-strain triangle (a.k.a., CST triangle, Turner triangle)
    """

    qrule = 'full'
    quadrature = {
        'full': (np.array([[1/3, 1/3]]), np.array([1/2])),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
