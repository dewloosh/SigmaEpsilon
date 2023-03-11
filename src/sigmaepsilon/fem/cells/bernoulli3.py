from polymesh.cells import QuadraticLine as Line
from polymesh.utils.cells.gauss import Gauss_Legendre_Line_Grid

from .bernoulli import BernoulliBase as Bernoulli
from .gen.b3 import (
    shape_function_values_bulk,
    shape_function_derivatives_bulk,
    shape_function_derivatives_multi_L,
)

from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC


__all__ = ["Bernoulli3"]


class Bernoulli3(ABC, Bernoulli, Line, FiniteElement):
    """
    Finite element class to handle 3-noded Bernoulli beams.
    """

    qrule = "full"
    quadrature = {
        "full": Gauss_Legendre_Line_Grid(6),
        "mass": Gauss_Legendre_Line_Grid(8),
    }
    shpfnc = shape_function_values_bulk
    dshpfnc = shape_function_derivatives_bulk
    dshpfnc_geom = shape_function_derivatives_multi_L
