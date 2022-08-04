# -*- coding: utf-8 -*-
import unittest
import numpy as np

from sigmaepsilon.solid import Structure, LineMesh, PointData, FemMesh
from sigmaepsilon.math.linalg import linspace, Vector
from sigmaepsilon.mesh.space import StandardFrame, PointCloud, frames_of_lines
from sigmaepsilon.solid.fem.cells import B2 as Beam, Q4M, Q9P, H8
from sigmaepsilon.math.array import repeat
from sigmaepsilon.mesh.grid import gridQ4, gridQ9, gridH8
from sigmaepsilon.solid.model import Membrane, MindlinPlate

"""def test_console_1(*args, 
                   L=3000,  # length of the console 
                   R=25.0, # radius of the tube
                   nElem=10,  # number of finite elements to use 
                   F=-1000,  # value of the vertical load at the free end
                   E=210000.0,  # Young's modulus
                   nu=0.3,  # Poisson's ratio
                   angles=None,
                   tol=1e-3
                   ):    
    # cross section
    A = PI * R**2
    Ix = PI * R**4 / 2
    Iy = PI* R**4 / 4
    Iz = PI * R**4 / 4

    # model stiffness matrix
    G = E / (2 * (1 + nu))
    Hooke = np.array([
        [E*A, 0, 0, 0],
        [0, G*Ix, 0, 0],
        [0, 0, E*Iy, 0],
        [0, 0, 0, E*Iz]
        ])
    
    # space
    GlobalFrame = StandardFrame(dim=3)
    if angles is None:
        angles = np.random.rand(3) * PI
    local_frame = GlobalFrame.fork('Body', angles, 'XYZ')
    
    # mesh
    p0 = np.array([0., 0., 0.])
    p1 = np.array([L, 0., 0.])
    coords = linspace(p0, p1, nElem+1)
    coords = PointCloud(coords, frame=local_frame).show()
    topo = np.zeros((nElem, 2), dtype=int)
    topo[:, 0] = np.arange(nElem)
    topo[:, 1] = np.arange(nElem) + 1
    
    # support at the leftmost, load at the rightmost node
    loads = np.zeros((coords.shape[0], 6))
    fixity = np.zeros((coords.shape[0], 6)).astype(bool)
    global_load_vector = \
        Vector([0., F, F], frame=local_frame).show()
    loads[-1, :3] = global_load_vector
    fixity[0, :] = True
            
    # set up objects
    mesh = LineMesh(coords=coords, topo=topo, loads=loads, fixity=fixity, 
                    celltype=Bernoulli, frame=GlobalFrame, model=Hooke)
    frames = repeat(local_frame.dcm(), nElem)
    frames = frames_of_lines(coords, topo)
    mesh.frames = frames
    structure = Structure(mesh=mesh)
        
    # solution
    structure.linsolve()
    
    # postproc
    # 1) displace the mesh
    dofsol = structure.nodal_dof_solution()[:, :3]
    coords_new = coords + dofsol
    structure.mesh.pointdata['x'] = coords_new
    local_dof_solution = \
        Vector(dofsol[-1, :3], frame=GlobalFrame).show(local_frame)
    sol_fem_Y = local_dof_solution[1]
    sol_fem_Z = local_dof_solution[2]
    check1 = np.abs(sol_fem_Y - sol_fem_Z) < tol
    
    # Bernoulli solution
    EI = E * Iy
    sol_exact = F * L**3 / (3 * EI)
    diffY = 100 * (sol_fem_Y - sol_exact) / sol_exact
    check2 = np.abs(diffY) < tol
    diffZ = 100 * (sol_fem_Z - sol_exact) / sol_exact
    check3 = np.abs(diffZ) < tol
    
    return all([check1, check2, check3])"""


"""def test_console_2(*args, 
                   L=3000,  # length of the console 
                   R=25.0, # radius of the tube
                   nElem=10,  # number of finite elements to use 
                   F=-1000,  # value of the vertical load at the free end
                   E=210000.0,  # Young's modulus
                   nu=0.3,  # Poisson's ratio
                   rotX=0,
                   tol=1e-3
                   ):    
    # cross section
    A = PI * R**2
    Ix = PI * R**4 / 2
    Iy = PI* R**4 / 4
    Iz = PI * R**4 / 4

    # model stiffness matrix
    G = E / (2 * (1 + nu))
    Hooke = np.array([
        [E*A, 0, 0, 0],
        [0, G*Ix, 0, 0],
        [0, 0, E*Iy, 0],
        [0, 0, 0, E*Iz]
        ])
    
    # space
    GlobalFrame = StandardFrame(dim=3)
    random_frame = GlobalFrame.fork('Body', [rotX, 0, 0], 'XYZ')
    
    # mesh
    p0 = np.array([0., 0., 0.])
    p1 = np.array([L, 0., 0.])
    coords = linspace(p0, p1, nElem+1)
    topo = np.zeros((nElem, 2), dtype=int)
    topo[:, 0] = np.arange(nElem)
    topo[:, 1] = np.arange(nElem) + 1
    
    # support at the leftmost, load at the rightmost node
    loads = np.zeros((coords.shape[0], 6))
    fixity = np.zeros((coords.shape[0], 6)).astype(bool)
    global_load_vector = \
        Vector([0., F, F], frame=random_frame).show()
    loads[-1, :3] = global_load_vector
    fixity[0, :] = True
            
    # set up objects
    mesh = LineMesh(coords=coords, topo=topo, loads=loads, fixity=fixity, 
                    celltype=Bernoulli, frame=GlobalFrame, model=Hooke)
    mesh.frames = repeat(GlobalFrame.dcm(), nElem)
    structure = Structure(mesh=mesh)
        
    # solution
    structure.linsolve()
    
    # postproc
    # 1) displace the mesh
    dofsol = structure.nodal_dof_solution()[:, :3]
    coords_new = coords + dofsol
    structure.mesh.pointdata['x'] = coords_new
    local_dof_solution = \
        Vector(dofsol[-1, :3], frame=GlobalFrame).show(random_frame)
    sol_fem_Y = local_dof_solution[1]
    sol_fem_Z = local_dof_solution[2]
    check1 = np.abs(sol_fem_Y - sol_fem_Z) < tol
    
    # Bernoulli solution
    EI = E * Iy
    sol_exact = F * L**3 / (3 * EI)
    diffY = 100 * (sol_fem_Y - sol_exact) / sol_exact
    check2 = np.abs(diffY) < tol
    diffZ = 100 * (sol_fem_Z - sol_exact) / sol_exact
    check3 = np.abs(diffZ) < tol
    
    return all([check1, check2, check3])"""
    


class TestConsole(unittest.TestCase):
    
    def test_console_Bernoulli_1d(self):
        L = 100.  # length of the console
        w, h = 10., 10.  # width and height
        F = -100.  # value of the vertical load at the free end
        E = 210000.0  # Young's modulus
        nu = 0.3  # Poisson's ratio
        
        # cross section
        A = w * h
        Iy = w * h**3 / 12
        Iz = w * h**3 / 12
        Ix = Iy + Iz
        
        # Bernoulli solution
        EI = E * Iy
        sol_exact = F * L**3 / (3 * EI)
        tol = np.abs(sol_exact / 1000)
        sol_exact
        
        nElem = 20  # number of finite elements to use
        
        # model stiffness matrix
        G = E / (2 * (1 + nu))
        Hooke = np.array([
            [E*A, 0, 0, 0],
            [0, G*Ix, 0, 0],
            [0, 0, E*Iy, 0],
            [0, 0, 0, E*Iz]
        ])
        
        # space
        GlobalFrame = StandardFrame(dim=3)

        # mesh
        p0 = np.array([0., 0., 0.])
        p1 = np.array([L, 0., 0.])
        coords = linspace(p0, p1, nElem+1)
        coords = PointCloud(coords, frame=GlobalFrame).show()
        topo = np.zeros((nElem, 2), dtype=int)
        topo[:, 0] = np.arange(nElem)
        topo[:, 1] = np.arange(nElem) + 1

        # support at the leftmost, load at the rightmost node
        loads = np.zeros((coords.shape[0], 6))
        fixity = np.zeros((coords.shape[0], 6)).astype(bool)
        global_load_vector = Vector([0., 0, F], frame=GlobalFrame).show()
        loads[-1, :3] = global_load_vector
        fixity[0, :] = True

        # pointdata
        pd = PointData(coords=coords, frame=GlobalFrame,
                    loads=loads, fixity=fixity)

        # celldata
        frames = frames_of_lines(coords, topo)
        cd = Beam(topo=topo, frames=frames)

        # set up mesh and structure
        mesh = LineMesh(pd, cd, model=Hooke, frame=GlobalFrame)
        structure = Structure(mesh=mesh)
        structure.linsolve()

        # postproc
        # 1) displace the mesh
        dofsol = structure.nodal_dof_solution()[:, :3]
        local_dof_solution = dofsol[-1, :3]
        sol_fem_1d_B2 = local_dof_solution[2]
        
    def test_console_Bernoulli_2dM(self):
        L = 100.  # length of the console
        w, h = 10., 10.  # width and height
        F = -100.  # value of the vertical load at the free end
        E = 210000.0  # Young's modulus
        nu = 0.3  # Poisson's ratio

        size = Lx, Lz = (L, h)
        shape = nx, nz = (200, 20)

        gridparams = {
            'size': size,
            'shape': shape,
            'origo': (0, 0),
            'start': 0
        }

        coords2d, topo = gridQ4(**gridparams)
        coords = np.zeros((coords2d.shape[0], 3))
        coords[:, [0, 2]] = coords2d[:, :]

        # fix points at x==0
        cond = coords[:, 0] <= 0.001
        ebcinds = np.where(cond)[0]
        fixity = np.zeros((coords.shape[0], 6), dtype=bool)
        fixity[ebcinds, :] = True
        fixity[:, 3:] = True
        fixity[:, 1] = True

        # loads
        loads = np.zeros((coords.shape[0], 6))
        cond = (coords[:, 0] > (Lx-(1e-12))) & (np.abs(coords[:, 2] - (Lz/2)) < 1e-12)
        nbcinds = np.where(cond)[0]
        loads[nbcinds, 2] = F

        membrane = {
            '0': {
                'hooke': Membrane.Hooke(E=E, nu=nu),
                'angle': 0.,
                'thickness': w
            },
        }
        A = Membrane.from_dict(membrane).stiffness_matrix()

        GlobalFrame = StandardFrame(dim=3)

        # pointdata
        pd = PointData(coords=coords, frame=GlobalFrame,
                    loads=loads, fixity=fixity)

        # celldata
        frame = GlobalFrame.orient_new('Body', [np.pi/2, 0, 0], 'XYZ')
        frames = repeat(frame.show(), topo.shape[0])
        cd = Q4M(topo=topo, frames=frames)

        # set up mesh and structure
        mesh = FemMesh(pd, cd, model=A, frame=GlobalFrame)
        structure = Structure(mesh=mesh)
        structure.linsolve()

        dofsol = structure.nodal_dof_solution()
        sol_fem_2d_M = dofsol[:, 2].min()
        
    def test_console_Bernoulli_2dP(self):
        L = 100.  # length of the console
        w, h = 10., 10.  # width and height
        F = -100.  # value of the vertical load at the free end
        E = 210000.0  # Young's modulus
        nu = 0.3  # Poisson's ratio
        
        size = Lx, Ly = (L, w)
        shape = nx, ny = (200, 20)

        gridparams = {
            'size': size,
            'shape': shape,
            'origo': (0, 0),
            'start': 0
        }

        coords2d, topo = gridQ9(**gridparams)
        coords = np.zeros((coords2d.shape[0], 3))
        coords[:, :2] = coords2d[:, :]

        # fix points at x==0
        cond = coords[:, 0] <= 0.001
        ebcinds = np.where(cond)[0]
        fixity = np.zeros((coords.shape[0], 6), dtype=bool)
        fixity[ebcinds, :] = True
        fixity[:, :2] = True
        fixity[:, -1] = True

        # loads
        loads = np.zeros((coords.shape[0], 6))
        cond = (coords[:, 0] > (Lx-(1e-12))) & (np.abs(coords[:, 1] - (Ly/2)) < 1e-12)
        nbcinds = np.where(cond)[0]
        loads[nbcinds, 2] = F

        model = {
            '0': {
                'hooke': MindlinPlate.Hooke(E=E, nu=nu),
                'angle': 0.,
                'thickness': h
            },
        }
        C = MindlinPlate.from_dict(model).stiffness_matrix()

        GlobalFrame = StandardFrame(dim=3)

        # pointdata
        pd = PointData(coords=coords, frame=GlobalFrame,
                    loads=loads, fixity=fixity)

        # celldata
        frames = repeat(np.eye(3), topo.shape[0])
        cd = Q9P(topo=topo, frames=frames)

        # set up mesh and structure
        mesh = FemMesh(pd, cd, model=C, frame=GlobalFrame)
        structure = Structure(mesh=mesh)

        structure.linsolve()

        dofsol = structure.nodal_dof_solution()
        sol_fem_2d_P = dofsol[:, 2].min()
        
    def test_console_Bernoulli_3d(self):
        L = 100.  # length of the console
        w, h = 10., 10.  # width and height
        F = -100.  # value of the vertical load at the free end
        E = 210000.0  # Young's modulus
        nu = 0.3  # Poisson's ratio
        
        size = Lx, Ly, Lz = (L, w, h)
        shape = nx, ny, nz = (100, 10, 10)

        gridparams = {
            'size': size,
            'shape': shape,
            'origo': (0, 0, 0),
            'start': 0
        }

        coords, topo = gridH8(**gridparams)

        A = np.array([
            [1, nu, nu, 0, 0, 0],
            [nu, 1, nu, 0, 0, 0],
            [nu, nu, 1, 0, 0, 0],
            [0., 0, 0, (1-nu)/2, 0, 0],
            [0., 0, 0, 0, (1-nu)/2, 0],
            [0., 0, 0, 0, 0, (1-nu)/2]]) * (E / (1-nu**2))

        # fix points at x==0
        cond = coords[:, 0] <= 0.001
        ebcinds = np.where(cond)[0]
        fixity = np.zeros((coords.shape[0], 6), dtype=bool)
        fixity[ebcinds, :] = True
        fixity[:, 3:] = True

        # unit vertical load at (Lx, Ly)
        cond = (coords[:, 0] > (Lx-(1e-12))) & \
            (np.abs(coords[:, 1] - (Ly/2)) < 1e-12) & \
            (np.abs(coords[:, 2] - (Lz/2)) < 1e-12)
        nbcinds = np.where(cond)[0]
        loads = np.zeros((coords.shape[0], 6))
        loads[nbcinds, 2] = F

        GlobalFrame = StandardFrame(dim=3)

        # pointdata
        pd = PointData(coords=coords, frame=GlobalFrame,
                    loads=loads, fixity=fixity)

        # celldata
        frames = repeat(GlobalFrame.show(), topo.shape[0])
        cd = H8(topo=topo, frames=frames)

        # set up mesh and structure
        mesh = FemMesh(pd, cd, model=A, frame=GlobalFrame)
        structure = Structure(mesh=mesh)

        structure.linsolve()
        dofsol = structure.nodal_dof_solution()
        structure.mesh.pointdata['x'] = coords + dofsol[:, :3]
        sol_fem_3d = dofsol[:, 2].min()
        
                        
    
if __name__ == "__main__":
    
    unittest.main()