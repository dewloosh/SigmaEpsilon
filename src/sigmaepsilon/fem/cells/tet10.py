from polymesh.cells import TET10 as Tetra
from polymesh.utils.cells.gauss import Gauss_Legendre_Tet_1, Gauss_Legendre_Tet_4

from ..material.solid3d import Solid3d
from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC


class TET10(ABC, Solid3d, Tetra, FiniteElement):
    qrule = "full"
    quadrature = {
        "full": Gauss_Legendre_Tet_4(),
        "selective": {(0, 1, 2): "full", (3, 4, 5): "reduced"},
        "reduced": Gauss_Legendre_Tet_1(),
    }
