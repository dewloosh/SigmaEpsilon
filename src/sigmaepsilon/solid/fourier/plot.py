# -*- coding: utf-8 -*-
import numpy as np

from ...math.linalg import normalize as norm
from ...mesh.utils import distances_of_points


def plot_path(ax, data, coords, *args, 
              normalize=False, markersize=3, 
              markercolor='r', title=None, **kwargs):
    x = np.cumsum(distances_of_points(coords))
    x = norm(x) if normalize else x
    ax.plot(x, data, zorder=1, color='k', lw=2)
    ax.scatter(x, data, s=markersize, zorder=2, color=markercolor)
    if title is not None:
        ax.set_title(title)
    return ax