import numpy as np

from polymesh.cells import T6 as Triangle

from ..material.membrane import Membrane
from ..material.mindlinplate import MindlinPlate

from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC


class LST_M(ABC, Membrane, Triangle, FiniteElement):
    """
    Linear strain triangle or LST. Also called quadratic triangle and Veubeke
    triangle. Developed by Fraeijs de Veubeke in 1962-63, published in 1965 [1].

    References
    ----------
    .. [1] B. M. Fraeijs de Veubeke, Displacement and equilibrium models, in Stress
       Analysis, ed. by O. C. Zienkiewicz and G. Hollister, Wiley, London, 1965, 145-197.
       Reprinted in Int. J. Numer. Meth. Engrg., 52, 287-342, 2001.
    """

    qrule = "selective"
    quadrature = {
        "full": (
            np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]]),
            np.array([1 / 6, 1 / 6, 1 / 6]),
        ),
        "full2": (
            np.array([[1 / 2, 1 / 2], [0, 1 / 2], [1 / 2, 0]]),
            np.array([1 / 6, 1 / 6, 1 / 6]),
        ),
        "selective": {(0, 1): "full", (2,): "reduced"},
        "reduced": (np.array([[1 / 3, 1 / 3]]), np.array([1 / 2])),
    }


class LST_P_MR(ABC, MindlinPlate, Triangle, FiniteElement):
    qrule = "selective"
    quadrature = {
        "full": (
            np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]]),
            np.array([1 / 6, 1 / 6, 1 / 6]),
        ),
        "full2": (
            np.array([[1 / 2, 1 / 2], [0, 1 / 2], [1 / 2, 0]]),
            np.array([1 / 6, 1 / 6, 1 / 6]),
        ),
        "selective": {(0, 1, 2): "full", (3, 4): "reduced"},
        "reduced": (np.array([[1 / 3, 1 / 3]]), np.array([1 / 2])),
    }
