import numpy as np
from numpy import ndarray, average as avg
from typing import Iterable, Callable

from neumann.function import Function
from neumann.linalg import linspace
from polymesh.grid import grid

from ..problem import NavierBeamProblem, NavierPlateProblem


def _coeffs_line_load_1d_mc(
    v: str, x: Iterable, p: NavierBeamProblem, fnc: Callable, N: int
) -> ndarray:
    f = Function(v, variables=["x"], dim=1)
    L = p.length
    points = np.linspace(x[0], x[1], N)
    d = x[1] - x[0]
    return np.array(
        list(
            map(
                lambda i: (2 / L) * d * avg(fnc(points, i) * f([points])),
                np.arange(1, p.N + 1),
            )
        )
    )


def _coeffs_line_load_2d_mc(
    v: str, x: Iterable, p: NavierPlateProblem, fnc: Callable, N: int
) -> ndarray:
    f = Function(v, variables=["x", "y"], dim=2)
    Lx, Ly = p.size
    points = linspace(x[0], x[1], N)
    d = np.sqrt((x[1, 0] - x[0, 0]) ** 2 + (x[1, 1] - x[0, 1]) ** 2)
    Nx, Ny = p.shape
    I = np.repeat(np.arange(1, Nx + 1), Ny)
    J = np.tile(np.arange(1, Ny + 1), Nx)
    return np.array(
        list(
            map(
                lambda ij: (4 / Lx / Ly)
                * d
                * avg(fnc(points, ij) * f([points[:, 0], points[:, 1]])),
                zip(I, J),
            )
        )
    )


def _coeffs_rect_load_mc(
    v: str, x: Iterable, p: NavierPlateProblem, fnc: Callable, N: int
) -> ndarray:
    f = Function(v, variables=["x", "y"], dim=2)
    Lx, Ly = p.size

    rect_width = x[1, 0] - x[0, 0]
    rect_height = x[1, 1] - x[0, 1]
    rect_area = rect_width * rect_height

    N = int(np.sqrt(N))
    points, _ = grid(
        size=(rect_width, rect_height), shape=(N, N), eshape=(2, 2), shift=x[0]
    )

    Nx, Ny = p.shape
    I = np.repeat(np.arange(1, Nx + 1), Ny)
    J = np.tile(np.arange(1, Ny + 1), Nx)
    return np.array(
        list(
            map(
                lambda ij: (4 / Lx / Ly)
                * rect_area
                * avg(fnc(points, ij) * f([points[:, 0], points[:, 1]])),
                zip(I, J),
            )
        )
    )
