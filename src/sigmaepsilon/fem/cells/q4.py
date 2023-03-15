from polymesh.utils.cells.q4 import area_Q4_bulk
from polymesh.cells import Q4 as Quadrilateral
from polymesh.utils.cells.gauss import (
    Gauss_Legendre_Quad_1,
    Gauss_Legendre_Quad_4,
)

from ..material.membrane import Membrane
from ..material.mindlinplate import MindlinPlate
from ..material.mindlinshell import MindlinShell

from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC


class Q4_M(ABC, Membrane, Quadrilateral, FiniteElement):
    qrule = "full"
    quadrature = {
        "full": Gauss_Legendre_Quad_4(),
        "selective": {(0, 1): "full", (2,): "reduced"},
        "reduced": Gauss_Legendre_Quad_1(),
    }

    def areas(self, *args, **kwargs):
        """This shadows the original geometrical implementation."""
        topo = self.topology().to_numpy() if topo is None else topo
        return area_Q4_bulk(self.local_coordinates(topo=topo))


class Q4_P_MR(ABC, MindlinPlate, Quadrilateral, FiniteElement):
    qrule = "selective"
    quadrature = {
        "full": Gauss_Legendre_Quad_4(),
        "selective": {(0, 1, 2): "full", (3, 4): "reduced"},
        "reduced": Gauss_Legendre_Quad_1(),
    }


class Q4_S_MR(ABC, MindlinShell, Quadrilateral, FiniteElement):
    qrule = "selective"
    quadrature = {
        "full": Gauss_Legendre_Quad_4(),
        "selective": {(0, 1, 3, 4, 5): "full", (2, 6, 7): "reduced"},
        "reduced": Gauss_Legendre_Quad_1(),
    }
