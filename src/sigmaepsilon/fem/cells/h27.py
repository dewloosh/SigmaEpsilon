from neumann.numint import gauss_points as gp

from polymesh.cells import H27 as HexaHedron

from ..material.solid3d import Solid3d

from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC


class H27(ABC, Solid3d, HexaHedron, FiniteElement):
    qrule = "full"
    quadrature = {
        "full": gp(3, 3, 3),
        "selective": {(0, 1, 2): "full", (3, 4, 5): "reduced"},
        "reduced": gp(2, 2, 2),
    }
