# -*- coding: utf-8 -*-
import numpy as np

from ....mesh.cells import T3 as Triangle

from ..model.membrane import Membrane
from ..model.plate import Plate

from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC



class CSTM(ABC, Membrane, Triangle, FiniteElement):
    """
    The constant-strain triangle (a.k.a., CST triangle, Turner triangle)
    """

    qrule = 'full'
    quadrature = {
        'full': (np.array([[1/3, 1/3]]), np.array([1/2])),
    }


class CSTP(ABC, Plate, Triangle, FiniteElement):
    """
    The constant-strain triangle (a.k.a., CST triangle, Turner triangle)
    """

    qrule = 'full'
    quadrature = {
        'full': (np.array([[1/3, 1/3]]), np.array([1/2])),
    }

