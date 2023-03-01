from polymesh.cells import H8 as HexaHedron
from polymesh.utils.cells.gauss import Gauss_Legendre_Hex_Grid

from ..material.solid3d import Solid3d
from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC


class H8(ABC, Solid3d, HexaHedron, FiniteElement):
    qrule = "selective"
    quadrature = {
        "full": Gauss_Legendre_Hex_Grid(2, 2, 2),
        "selective": {(0, 1, 2): "full", (3, 4, 5): "reduced"},
        "reduced": Gauss_Legendre_Hex_Grid(1, 1, 1),
    }
