from polymesh.cells import TET4 as Tetra

from ..material.solid3d import Solid3d
from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC


class TET4(ABC, Solid3d, Tetra, FiniteElement):
    qrule = "full"
