# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit

from dewloosh.math.numint import GaussPoints as Gauss

from dewloosh.mesh.tri.triutils import area_tri_bulk
from dewloosh.mesh.utils import cells_coords
from dewloosh.mesh.cells import Q4 as Quadrilateral

from dewloosh.solid.fem.elem import FiniteElement
from dewloosh.solid.fem.meta import ABCFiniteElement as ABC

from ..model.membrane import Membrane
from ..model.plate import Plate

__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def area_Q4_bulk(ecoords : np.ndarray):
    nE = len(ecoords)
    res = np.zeros(nE, dtype=ecoords.dtype)
    res += area_tri_bulk(ecoords[:, :3, :])
    res += area_tri_bulk(ecoords[:, np.array([0, 2, 3]), :])
    return res


class Q4M(ABC, Membrane, Quadrilateral, FiniteElement):
    
    qrule = 'full'
    quadrature = {
        'full' : Gauss(2, 2), 
        'selective' : {
            (0, 1) : 'full',
            (2,) : 'reduced'
            },
        'reduced' : Gauss(1, 1)
        }
        
    def areas(self, *args, coords=None, topo=None, **kwargs):
        """This shadows the original geometrical implementation."""
        if coords is None:
            coords = self.container.root().coords()
        topo = self.topology().to_numpy() if topo is None else topo
        return area_Q4_bulk(cells_coords(coords, topo))
       

class Q4P(ABC, Plate, Quadrilateral, FiniteElement):
    
    qrule = 'selective'
    quadrature = {
        'full' : Gauss(2, 2), 
        'selective' : {
            (0, 1, 2) : 'full',
            (3, 4) : 'reduced'
            },
        'reduced' : Gauss(1, 1)
        }
        
    def areas(self, *args, coords=None, topo=None, **kwargs):
        """This shadows the original geometrical implementation."""
        if coords is None:
            coords = self.container.root().coords()
        topo = self.topology().to_numpy() if topo is None else topo
        self.local_coordinates(topo=topo)
        return area_Q4_bulk(cells_coords(coords, topo))
    
    
if __name__ == '__main__':
    from sympy import symbols

    r, s = symbols('r s')

    N1 = 0.125*(1-r)*(1-s)
    N2 = 0.125*(1+r)*(1-s)
    N3 = 0.125*(1+r)*(1+s)
    N4 = 0.125*(1-r)*(1+s)
