from polymesh.cells import T6 as Triangle
from polymesh.utils.cells.gauss import (
    Gauss_Legendre_Tri_1,
    Gauss_Legendre_Tri_3a,
    Gauss_Legendre_Tri_3b,
)

from ..material.membrane import Membrane
from ..material.mindlinplate import MindlinPlate
from ..material.mindlinshell import MindlinShell

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
        "full": Gauss_Legendre_Tri_3a(),
        "full2": Gauss_Legendre_Tri_3b(),
        "selective": {(0, 1): "full", (2,): "reduced"},
        "reduced": Gauss_Legendre_Tri_1(),
    }


class LST_P_MR(ABC, MindlinPlate, Triangle, FiniteElement):
    qrule = "selective"
    quadrature = {
        "full": Gauss_Legendre_Tri_3a(),
        "full2": Gauss_Legendre_Tri_3b(),
        "selective": {(0, 1, 2): "full", (3, 4): "reduced"},
        "reduced": Gauss_Legendre_Tri_1(),
    }


class LST_S_MR(ABC, MindlinShell, Triangle, FiniteElement):
    qrule = "selective"
    quadrature = {
        "full": Gauss_Legendre_Tri_3a(),
        "full2": Gauss_Legendre_Tri_3b(),
        "selective": {(0, 1, 3, 4, 5): "full", (2, 6, 7): "reduced"},
        "reduced": Gauss_Legendre_Tri_1(),
    }
