# -*- coding: utf-8 -*-
import numpy as np

from dewloosh.mesh.cells import TET4 as Tetra

from ..model.solid3d import Solid3d
from ..elem import FiniteElement


class TET4(Tetra, Solid3d, FiniteElement):

    qrule = 'full'
    quadrature = {
        'full': (np.array([[1/3, 1/3, 1/3]]), np.array([1/6])),
    }
