from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

import numpy as np


def parallel_plot_mpl(data, *args, labels=None, padding=0.05, color=None,
                      colors=None, lw=0.2, bezier=True, axis=0, figsize=None, 
                      title=None, **kwargs):

    if isinstance(data, dict):
        if labels is None:
            labels = list(data.keys())
        ys = np.dstack(list(data.values()))[0]
    elif isinstance(data, np.ndarray):
        assert labels is not None
        ys = data
    elif isinstance(data, Iterable):
        assert labels is not None
        ys = np.dstack(data)[0]
    else:
        raise TypeError('Invalid data type!')
    
    ynames = labels
    N, nY = ys.shape

    if colors is not None:
        pass

    figsize = (7.5, 3) if figsize is None else figsize
    fig, host = plt.subplots(figsize=figsize)

    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * padding
    ymaxs += dys * padding
    dys = ymaxs - ymins

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    axes = [host] + [host.twinx() for i in range(nY - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (nY - 1)))

    host.set_xlim(0, nY - 1)
    host.set_xticks(range(nY))
    host.set_xticklabels(ynames, fontsize=8)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()
    if title is not None:
        host.set_title(title, fontsize=12)

    for j in range(N):
        if not bezier:
            # to just draw straight lines between the axes:
            host.plot(range(nY), zs[j, :], c=colors[j])
        else:
            # create bezier curves
            # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
            #   at one third towards the next axis; the first and last axis have one less control vertex
            # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
            # y-coordinate: repeat every point three times, except the first and last only twice
            verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                            np.repeat(zs[j, :], 3)[1:-1]))
            # for x,y in verts: host.plot(x, y, 'go') # to show the control points of the beziers
            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
            path = Path(verts, codes)
            patch = PathPatch(path, facecolor='none', lw=lw, edgecolor=colors[j])
            host.add_patch(patch)