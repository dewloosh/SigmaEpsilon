from polymesh.cells import H27 as HexaHedron
from polymesh.utils.cells.gauss import Gauss_Legendre_Hex_Grid

from ..material.solid3d import Solid3d
from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC


class H27(ABC, Solid3d, HexaHedron, FiniteElement):
    qrule = "full"
    quadrature = {
        "full": Gauss_Legendre_Hex_Grid(3, 3, 3),
        "selective": {(0, 1, 2): "full", (3, 4, 5): "reduced"},
        "reduced": Gauss_Legendre_Hex_Grid(2, 2, 2),
    }
