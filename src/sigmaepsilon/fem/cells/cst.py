from polymesh.cells import T3 as Triangle

from ..material.membrane import Membrane
from ..material.mindlinplate import MindlinPlate
from ..material.mindlinshell import MindlinShell

from .elem import FiniteElement
from .meta import ABCFiniteElement as ABC


class CST_M(ABC, Membrane, Triangle, FiniteElement):
    """
    The constant-strain triangle a.k.a., CST triangle, Turner triangle or
    linear triangle for membranes. Developed as a plane stress element by
    John Turner, Ray Clough and Harold Martin in 1952-53 [1], published in 1956 [2].

    Notes
    -----
    The element has poor performance and is represented for historycal reasons.
    Don't use it in a production enviroment, unless your mesh is extremely dense.

    References
    ----------
    .. [1] R. W. Clough, The finite element method - a personal view of its original
       formulation, in From Finite Elements to the Troll Platform - the Ivar Holand
       70th Anniversary Volume, ed. by K. Bell, Tapir, Trondheim, Norway, 89-100, 1994.
    .. [2] M. J. Turner, R. W. Clough, H. C. Martin, and L. J. Topp, Stiffness and
       deflection analysis of complex structures, J. Aero. Sco., 23, pp. 805-824, 1956.
    """

    qrule = "full"


class CST_P_MR(ABC, MindlinPlate, Triangle, FiniteElement):
    """
    The constant-strain triangle (a.k.a., CST triangle, Turner triangle)
    """

    qrule = "full"


class CST_S_MR(ABC, MindlinShell, Triangle, FiniteElement):
    qrule = "full"
