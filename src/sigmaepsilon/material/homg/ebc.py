from typing import Iterable, Union

import numpy as np
from numpy import ndarray

from polymesh.utils.space import index_of_closest_point
from polymesh.space import PointCloud
from polymesh.utils import locate_tri_2d
from polymesh.cells import T3 as Triangle

from sigmaepsilon.fem import FemMesh
from sigmaepsilon.fem.ebc import NodeToNode
from sigmaepsilon.fem.ebc import BodyToBody


def _link_opposite_sides_sym(
    points: PointCloud, axis: Union[int, Iterable[int]] = 0
) -> NodeToNode:
    coords = points.show()
    bounds = points.bounds()
    links = []
    if isinstance(axis, int):
        axis = [axis]
    assert isinstance(
        axis, Iterable
    ), "'axis' must be an integer or an 1d iterable of integers."
    for axid in axis:
        bmin, bmax = bounds[axid]
        mask_source = coords[:, axid] < (bmin + 1e-12)
        i_source = np.where(mask_source)[0]
        source = coords[i_source]
        mask_target = coords[:, axid] > (bmax - 1e-12)
        i_target = np.where(mask_target)[0]
        target = coords[i_target]
        i = index_of_closest_point(source, target)
        id_source = points[i_source].id[i]
        id_target = points[i_target].id
        links.append(np.stack([id_source, id_target], axis=1))
    return NodeToNode(np.vstack(links))


def _link_opposite_sides(
    mesh: FemMesh, axis: Union[int, Iterable[int]] = 0
) -> ndarray:
    if isinstance(axis, int):
        axis = [axis]
    assert isinstance(
        axis, Iterable
    ), "'axis' must be an integer or an 1d iterable of integers."
        
    surface = mesh.surface()
    points = surface.points()
    triangles = surface.topology()
    centers = surface.centers()
    coords = points.show()
    bounds = points.bounds()
    
    def _link_points_to_cells_(source_indices, target_indices):
        source_coords = coords[source_indices]
        target_triangles = triangles[target_indices]
        
        i_source, i_target, locations_target = \
            locate_tri_2d(source_coords[:, mask], coords[:, mask], target_triangles)
        source_indices = source_indices[i_source]   
        target_triangles = target_triangles[i_target]
        
        shp_target = Triangle.shape_function_values(locations_target)
        
        nE, nNE = target_triangles.shape
        factors = np.zeros((nE, nNE + 1), dtype=float)
        indices = np.zeros((nE, nNE + 1), dtype=int)
        factors[:, -1] = -1
        factors[:, :nNE] = shp_target
        indices[:, -1] = source_indices
        indices[:, :nNE] = target_triangles
        
        return factors, indices
    
    factors, indices = [], []
    
    for axid in axis:
        bmin, bmax = bounds[axid]
        
        mask = np.ones(3, dtype=bool)
        mask[axid] = False
        
        mask_source = coords[:, axid] < (bmin + 1e-12)
        source_indices = np.where(mask_source)[0]
        mask_target = centers[:, axid] > (bmax - 1e-12)
        target_indices = np.where(mask_target)[0]
        f, i = _link_points_to_cells_(source_indices, target_indices)
        factors.append(f)
        indices.append(i)
        
        mask_source = coords[:, axid] > (bmax - 1e-12)
        source_indices = np.where(mask_source)[0]
        mask_target = centers[:, axid] < (bmin + 1e-12)
        target_indices = np.where(mask_target)[0]
        f, i = _link_points_to_cells_(source_indices, target_indices)
        factors.append(f)
        indices.append(i)
        
    return BodyToBody(factors=np.vstack(factors), indices=np.vstack(indices))
