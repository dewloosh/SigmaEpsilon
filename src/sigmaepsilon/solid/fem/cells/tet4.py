# -*- coding: utf-8 -*-
import numpy as np

from polymesh.cells import TET4 as Tetra
from polymesh.tet import nat_to_loc_tet as n2l

from ..model.solid3d import Solid3d
from .elem import FiniteElement
from .metaelem import ABCFiniteElement as ABC



class TET4(ABC, Solid3d, Tetra, FiniteElement):

    qrule = 'full'
    quadrature = {
        'full': (
            np.array([[1/3, 1/3, 1/3]]), 
            np.array([1.])
            ),
    }
