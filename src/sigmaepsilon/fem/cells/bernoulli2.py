from polymesh.cells import L2 as Line
from polymesh.utils.cells.gauss import Gauss_Legendre_Line_Grid

from .bernoulli import BernoulliBase as Bernoulli
from .gen.b2 import (
    shape_function_values_bulk,
    shape_function_derivatives_bulk,
    shape_function_derivatives_multi_L,
)

from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC

__all__ = ["Bernoulli2"]


class Bernoulli2(ABC, Bernoulli, Line, FiniteElement):
    """
    Finite element class to handle 2-noded Bernoulli beams.
    """

    qrule = "full"
    quadrature = {
        "full": Gauss_Legendre_Line_Grid(2),
        "selective": {(0, 1): "full", (2): "reduced"},
        "reduced": Gauss_Legendre_Line_Grid(1),
        "mass": Gauss_Legendre_Line_Grid(4),
    }
    shpfnc = shape_function_values_bulk
    dshpfnc = shape_function_derivatives_bulk
    dshpfnc_geom = shape_function_derivatives_multi_L
