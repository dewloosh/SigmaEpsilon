# -*- coding: utf-8 -*-
import numpy as np

from polymesh.cells import TET10 as Tetra

from ..model.solid3d import Solid3d

from .elem import FiniteElement
from .metaelem import ABCFiniteElement as ABC


gauss_pos_nat = np.array(
    [
        [0.585410196624968, 0.138196601125010, 0.138196601125010, 0.138196601125010],
        [0.138196601125010, 0.585410196624968, 0.138196601125010, 0.138196601125010],
        [0.138196601125010, 0.138196601125010, 0.585410196624968, 0.138196601125010],
        [0.138196601125010, 0.138196601125010, 0.138196601125010, 0.585410196624968]
    ]
)
gauss_weights = np.array([0.25, 0.25, 0.25, 0.25])


class TET10(ABC, Solid3d, Tetra, FiniteElement):

    qrule = 'full'
    quadrature = {
        'full': (gauss_pos_nat, gauss_weights),
    }
