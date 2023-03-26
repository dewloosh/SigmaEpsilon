from typing import Tuple

import numpy as np
from numpy import ndarray
from numba import njit, prange

from polymesh.utils.space import index_of_closest_point
from polymesh.space import PointCloud
from polymesh.cell import PolyCell3d

__cache = True


def link_points_to_points(
    source: PointCloud,
    target: PointCloud,
) -> ndarray:
    i = index_of_closest_point(source.show(), target.show())
    link = np.stack([source.id[i], target.id], axis=1)
    return link


def link_points_to_body(
    points: PointCloud, 
    body: PolyCell3d,
    lazy: bool = True,
    tol: float = 1e-12,
    k: int = 4,
) -> Tuple[ndarray]:
    source_coords = points.show()
    source_indices = points.id
    topo_target = body.topology()
        
    i_source, i_target, locations_target = \
        body.locate(source_coords, lazy, tol, k)
    source_indices = source_indices[i_source]
    topo_target = topo_target[i_target]
    
    shp_target = body.shape_function_values(locations_target)
            
    nE, nNE = topo_target.shape
    factors = np.zeros((nE, nNE + 1), dtype=float)
    indices = np.zeros((nE, nNE + 1), dtype=int)
    factors[:, -1] = -1
    factors[:, :nNE] = shp_target
    indices[:, -1] = source_indices
    indices[:, :nNE] = topo_target
    
    return factors, indices


@njit(nogil=True, parallel=True, cache=__cache)
def _body_to_body_stiffness_data_(
    nodal_factors:ndarray, nodal_indices:ndarray, dofmap:ndarray, ndof:int
) -> Tuple[ndarray]:
    nN, nV = nodal_factors.shape
    nDOF = dofmap.shape[0]
    nR = nN * nDOF
    factors = np.zeros((nR, nV), dtype=nodal_factors.dtype)
    indices = np.zeros((nR, nV), dtype=nodal_indices.dtype)
    for i in prange(nN):
        for j in prange(nDOF):
            ii = i*nDOF + j
            for k in prange(nV):
                ik = nodal_indices[i, k] * ndof + dofmap[j]
                factors[ii, k] = nodal_factors[i, k]
                indices[ii, k] = ik
    return factors, indices