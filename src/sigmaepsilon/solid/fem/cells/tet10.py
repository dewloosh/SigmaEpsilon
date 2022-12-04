# -*- coding: utf-8 -*-
import numpy as np

from polymesh.cells import TET10 as Tetra

from ..model.solid3d import Solid3d

from .elem import FiniteElement
from .metaelem import ABCFiniteElement as ABC


class TET10(ABC, Solid3d, Tetra, FiniteElement):

    qrule = 'full'
    quadrature = {
        'full': (
            np.array(
                [
                    [0.138196601125010, 0.138196601125010, 0.138196601125010],
                    [0.585410196624968, 0.138196601125010, 0.138196601125010],
                    [0.138196601125010, 0.585410196624968, 0.138196601125010],
                    [0.138196601125010, 0.138196601125010, 0.585410196624968]
                ]
            ),
            np.full(4, 1/24)
        ),
    }
