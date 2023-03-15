import numpy as np
from linkeddeepdict import LinkedDeepDict
from linkeddeepdict.tools import getallfromkwargs

from polymesh.space import StandardFrame, PointCloud
from polymesh.grid import gridH8 as grid
from polymesh.utils.topology.tr import H8_to_L2
from polymesh.utils.space import index_of_closest_point, frames_of_lines

from sigmaepsilon import BeamSection
from sigmaepsilon import Structure, LineMesh, PointData
from sigmaepsilon.fem.cells import B2 as Beam


def console_grid_bernoulli() -> Structure:
    """
    Returns a simple console as a grid of 2-noded Bernoulli beams.

    Example
    -------
    >>> from sigmaepsilon.examples import console_grid_bernoulli
    >>> structure = console_grid_bernoulli()
    >>> structure.linsolve()
    """
    # units in kN and cm

    Lx, Ly, Lz = 100.0, 15.0, 10.0  # size of the grid
    nx, ny, nz = 4, 2, 2  # desity of the grid

    d = 1.2  # outer diameter of the tube
    t = 0.4  # thickness of the tube

    Ex = 21000.0
    nu = 0.25
    fy = 35.0

    loads = LinkedDeepDict()

    # load case 1
    loads["LC1", "position"] = [Lx, 0, 0]
    loads["LC1", "value"] = [0, 0, 2.0, 0, 0, 0]

    # load case 2
    loads["LC2", "position"] = [Lx, Ly / 2, 0]
    loads["LC2", "value"] = [0, 2.0, 0, 0, 0, 0]

    # load case 3
    loads["LC3", "position"] = [Lx, -Ly / 2, 0]
    loads["LC3", "value"] = [0, -2.0, 0, 0, 0, 0]

    gridparams = {
        "size": (Lx, Ly, Lz),
        "shape": (nx, ny, nz),
        "origo": (0, 0, 0),
        "start": 0,
    }

    coords, topo = grid(**gridparams)
    coords, topo = H8_to_L2(coords, topo)

    GlobalFrame = StandardFrame(dim=3)

    points = PointCloud(coords, frame=GlobalFrame).centralize()
    dx = -np.array([points[:, 0].min(), 0.0, 0.0])
    points.move(dx)
    coords = points.show()

    section = BeamSection("CHS", d=d, t=t, n=16)
    section.calculate_section_properties()
    section_props = section.section_properties
    A, Ix, Iy, Iz = getallfromkwargs(["A", "Ix", "Iy", "Iz"], **section_props)

    G = Ex / (2 * (1 + nu))
    Hooke = np.array(
        [[Ex * A, 0, 0, 0], [0, G * Ix, 0, 0], [0, 0, Ex * Iy, 0], [0, 0, 0, Ex * Iz]]
    )

    # essential boundary conditions
    ebcinds = np.where(coords[:, 0] < 1e-12)[0]
    fixity = np.zeros((coords.shape[0], 6)).astype(bool)
    fixity[ebcinds, :] = True

    # natural boundary conditions
    nLoadCase = len(loads)
    nodal_loads = np.zeros((coords.shape[0], 6, nLoadCase))
    for iLC, key in enumerate(loads):
        x = loads[key]["position"]
        f = loads[key]["value"]
        iN = index_of_closest_point(coords, np.array(x))
        loads[key]["node"] = iN
        nodal_loads[iN, :, iLC] = f

    # pointdata
    pd = PointData(coords=coords, frame=GlobalFrame, loads=nodal_loads, fixity=fixity)

    # celldata
    frames = frames_of_lines(coords, topo)
    cd = Beam(topo=topo, frames=frames)

    # set up mesh and structure
    mesh = LineMesh(pd, cd, model=Hooke, frame=GlobalFrame)
    structure = Structure(mesh=mesh)

    return structure
