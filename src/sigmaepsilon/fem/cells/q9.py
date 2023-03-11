from polymesh.cells import Q9 as Quadrilateral
from polymesh.utils.cells.gauss import (
    Gauss_Legendre_Quad_9,
    Gauss_Legendre_Quad_4,
)

from ..material.membrane import Membrane
from ..material.mindlinplate import MindlinPlate
from ..material.mindlinshell import MindlinShell

from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC


class Q9_M(ABC, Membrane, Quadrilateral, FiniteElement):
    qrule = "full"
    quadrature = {
        "full": Gauss_Legendre_Quad_9(),
        "selective": {(0, 1): "full", (2,): "reduced"},
        "reduced": Gauss_Legendre_Quad_4(),
    }


class Q9_P_MR(ABC, MindlinPlate, Quadrilateral, FiniteElement):
    qrule = "selective"
    quadrature = {
        "full": Gauss_Legendre_Quad_9(),
        "selective": {(0, 1, 2): "full", (3, 4): "reduced"},
        "reduced": Gauss_Legendre_Quad_4(),
    }


class Q9_S_MR(ABC, MindlinShell, Quadrilateral, FiniteElement):
    qrule = "selective"
    quadrature = {
        "full": Gauss_Legendre_Quad_9(),
        "selective": {(0, 1, 3, 4, 5): "full", (2, 6, 7): "reduced"},
        "reduced": Gauss_Legendre_Quad_4(),
    }
