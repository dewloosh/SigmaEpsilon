# -*- coding: utf-8 -*-
import numpy as np

from dewloosh.mesh.cells import TET4 as Tetra

from dewloosh.solid.fem.elem import FiniteElement
from dewloosh.solid.fem.meta import ABCFiniteElement as ABC

from ..model.solid3d import Solid3d



class TET4(Tetra, Solid3d, FiniteElement):

    qrule = 'full'
    quadrature = {
        'full': (np.array([[1/3, 1/3, 1/3]]), np.array([1/6])),
    }
