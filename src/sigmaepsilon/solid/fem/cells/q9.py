from neumann.numint import gauss_points as gp

from polymesh.cells import Q9 as Quadrilateral

from ..material.membrane import Membrane
from ..material.mindlinplate import MindlinPlate

from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC


class Q9_M(ABC, Membrane, Quadrilateral, FiniteElement):
    qrule = "full"
    quadrature = {
        "full": gp(3, 3),
        "selective": {(0, 1): "full", (2,): "reduced"},
        "reduced": gp(2, 2),
    }


class Q9_P_MR(ABC, MindlinPlate, Quadrilateral, FiniteElement):
    qrule = "selective"
    quadrature = {
        "full": gp(3, 3),
        "selective": {(0, 1, 2): "full", (3, 4): "reduced"},
        "reduced": gp(2, 2),
    }
