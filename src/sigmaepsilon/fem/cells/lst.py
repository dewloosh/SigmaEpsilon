# -*- coding: utf-8 -*-
import numpy as np

from dewloosh.mesh.cells import T6 as Triangle

from dewloosh.solid.fem.elem import FiniteElement

from dewloosh.solid.fem.model.membrane import Membrane
from dewloosh.solid.fem.model.plate import Plate


class LSTM(Triangle, Membrane, FiniteElement):

    qrule = 'selective'
    quadrature = {
        'full': (np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]]),
                 np.array([1/6, 1/6, 1/6])),
        'full2': (np.array([[1/2, 1/2], [0, 1/2], [1/2, 0]]),
                  np.array([1/6, 1/6, 1/6])),
        'selective': {
            (0, 1): 'full',
            (2,): 'reduced'
        },
        'reduced': (np.array([[1/3, 1/3]]), np.array([1/2]))
    }


class LSTP(Triangle, Plate, FiniteElement):

    qrule = 'selective'
    quadrature = {
        'full': (np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]]),
                 np.array([1/6, 1/6, 1/6])),
        'full2': (np.array([[1/2, 1/2], [0, 1/2], [1/2, 0]]),
                  np.array([1/6, 1/6, 1/6])),
        'selective': {
            (0, 1, 2): 'full',
            (3, 4): 'reduced'
        },
        'reduced': (np.array([[1/3, 1/3]]), np.array([1/2]))
    }
