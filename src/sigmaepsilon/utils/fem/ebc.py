from typing import Iterable, Union

import numpy as np
from numpy import ndarray

from polymesh.utils.space import index_of_closest_point
from polymesh.space import PointCloud


def link_opposite_sides(
    points: PointCloud, axis: Union[int, Iterable[int]] = 0
) -> ndarray:
    coords = points.show()
    bounds = points.bounds()
    links = []
    if isinstance(axis, int):
        axis = [axis]
    assert isinstance(
        axis, Iterable
    ), "'axis' must be an integer or an 1d iterable of integers."
    for iD in axis:
        bmin, bmax = bounds[iD]
        mask_source = coords[:, iD] < (bmin + 1e-12)
        i_source = np.where(mask_source)[0]
        source = coords[i_source]
        mask_target = coords[:, iD] > (bmax - 1e-12)
        i_target = np.where(mask_target)[0]
        target = coords[i_target]
        i = index_of_closest_point(source, target)
        id_source = points[i_source].id[i]
        id_target = points[i_target].id
        links.append(np.stack([id_source, id_target], axis=1))
    return np.vstack(links)


def link_points_to_points(
    source: PointCloud,
    target: PointCloud,
) -> ndarray:
    i = index_of_closest_point(source.show(), target.show())
    link = np.stack([source.id[i], target.id], axis=1)
    return link
