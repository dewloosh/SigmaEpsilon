from neumann.numint import gauss_points as gp
from polymesh.cells import L2 as Line

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
        "full": gp(2),
        "selective": {(0, 1): "full", (2): "reduced"},
        "reduced": gp(1),
        "mass": gp(4),
    }
    shpfnc = shape_function_values_bulk
    dshpfnc = shape_function_derivatives_bulk
    dshpfnc_geom = shape_function_derivatives_multi_L
