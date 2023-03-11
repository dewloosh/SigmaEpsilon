from typing import Iterable

import numpy as np


def points_to_rectangle_region(points: np.ndarray):
    xmin = points[:, 0].min()
    ymin = points[:, 1].min()
    xmax = points[:, 0].max()
    ymax = points[:, 1].max()
    return xmin, ymin, xmax - xmin, ymax - ymin


def rectangle_coordinates(
    *_,
    region: Iterable = None,
    xy: Iterable = None,
    w: float = None,
    h: float = None,
    center: Iterable = None
) -> Iterable:
    """
    Returns the bottom-left and upper-right coordinates of the region
    of the load from several inputs.

    Parameters
    ----------
    region : Iterable, Optional
        An iterable of length 4 with values x0, y0, w, and h. Here x0 and y0 are
        the coordinates of the bottom-left corner, w and h are the width and height
        of the region.
    xy : Iterable, Optional
        The position of the bottom-left corner as an iterable of length 2.
    w : float, Optional
        The width of the region.
    h : float, Optional
        The height of the region.
    center : Iterable, Optional
        The coordinates of the center of the region.

    Returns
    -------
    numpy.ndarray
        A 2d float array of coordinates, where the entries of the first and second
        rows are the coordinates of the lower-left and upper-right points of the region.

    Examples
    --------
    The following definitions return the same output:

    >>> from sigmaepsilon.fourier.utils import rectangle_coordinates
    >>> rectangle_coordinates(region=[2, 3, 0.5, 0.7])
    >>> rectangle_coordinates(xy=[2, 3], w=0.5, h=0.7)
    >>> rectangle_coordinates(center=[2.25, 3.35], w=0.5, h=0.7)
    """
    points = None
    if region:
        x0, y0, w, h = np.array(region)
        points = np.array([[x0, y0], [x0 + w, y0 + h]])
    elif xy and w and h:
        (x0, y0), w, h = xy, w, h
        points = np.array([[x0, y0], [x0 + w, y0 + h]])
    elif center and w and h:
        (xc, yc), w, h = center, w, h
        points = np.array([[xc - w / 2, yc - h / 2], [xc + w / 2, yc + h / 2]])
    else:
        ValueError("Invalid definition!")
    return points


def sin1d(x, i=1, L=1):
    return np.sin(x * np.pi * i / L)


def cos1d(x, i=1, L=1):
    return np.cos(x * np.pi * i / L)
