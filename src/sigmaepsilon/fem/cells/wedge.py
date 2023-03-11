from polymesh.cells import W6 as Wedge6, W18 as Wedge18

from ..material.solid3d import Solid3d
from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC


class W6(ABC, Solid3d, Wedge6, FiniteElement):
    qrule = "full"


class W18(ABC, Solid3d, Wedge18, FiniteElement):
    qrule = "full"
