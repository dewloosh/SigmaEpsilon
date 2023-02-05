from neumann.numint import gauss_points as gp

from polymesh.utils.cells.q4 import area_Q4_bulk
from polymesh.cells import Q4 as Quadrilateral

from ..material.membrane import Membrane
from ..material.mindlinplate import MindlinPlate
from ..material.mindlinshell import MindlinShell

from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC


class Q4_M(ABC, Membrane, Quadrilateral, FiniteElement):
    qrule = "full"
    quadrature = {
        "full": gp(2, 2),
        "selective": {(0, 1): "full", (2,): "reduced"},
        "reduced": gp(1, 1),
    }

    def areas(self, *args, **kwargs):
        """This shadows the original geometrical implementation."""
        topo = self.topology().to_numpy() if topo is None else topo
        return area_Q4_bulk(self.local_coordinates(topo=topo))


class Q4_P_MR(ABC, MindlinPlate, Quadrilateral, FiniteElement):
    qrule = "selective"
    quadrature = {
        "full": gp(2, 2),
        "selective": {(0, 1, 2): "full", (3, 4): "reduced"},
        "reduced": gp(1, 1),
    }


class Q4_S_MR(ABC, MindlinShell, Quadrilateral, FiniteElement):
    qrule = "selective"
    quadrature = {
        "full": gp(2, 2),
        "selective": {(0, 1, 3, 4, 5): "full", (2, 6, 7): "reduced"},
        "reduced": gp(1, 1),
    }
