# -*- coding: utf-8 -*-
import numpy as np

from dewloosh.mesh.cells import TET10 as Tetra

from dewloosh.solid.fem.cells import FiniteElement, ABCFiniteElement as ABC

from ..model.solid3d import Solid3d


gauss_pos_nat = np.array(
    [
        [0.585410196624968, 0.138196601125010, 0.138196601125010, 0.138196601125010],
        [0.138196601125010, 0.585410196624968, 0.138196601125010, 0.138196601125010],
        [0.138196601125010, 0.138196601125010, 0.585410196624968, 0.138196601125010],
        [0.138196601125010, 0.138196601125010, 0.138196601125010, 0.585410196624968]
    ]
)
gauss_weights = np.array([0.25, 0.25, 0.25, 0.25])


class TET10(Tetra, Solid3d, FiniteElement):

    qrule = 'full'
    quadrature = {
        'full': (gauss_pos_nat, gauss_weights),
    }
