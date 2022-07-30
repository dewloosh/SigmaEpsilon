# -*- coding: utf-8 -*-
from sigmaepsilon.math.linalg.vector import Vector
from sigmaepsilon.math.linalg import linspace, Vector
from sigmaepsilon.solid.fem.cells import B2 as Bernoulli
from sigmaepsilon.mesh.space import StandardFrame, \
    PointCloud, frames_of_lines
from sigmaepsilon.solid.fem import LineMesh
from sigmaepsilon.solid.fem import Structure
from sigmaepsilon.math.array import repeat
from hypothesis import given, settings, HealthCheck, strategies as st
import unittest
import numpy as np
from numpy import pi as PI


settings.register_profile(
    "fem_test",
    max_examples=50,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
)


def test_console_1(*args, 
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
    
    return all([check1, check2, check3])


def test_console_2(*args, 
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
    
    return all([check1, check2, check3])
    


class TestConsole(unittest.TestCase):
    
    def test_console_Bernoulli_1a(self):
        assert test_console_1(tol=1e-3)
                
    @given(st.integers(min_value=5, max_value=20), 
           st.floats(min_value=3000., max_value=3500.),
           st.floats(min_value=0., max_value=2*PI),
           st.floats(min_value=0., max_value=2*PI),
           st.floats(min_value=0., max_value=2*PI))
    @settings(settings.load_profile("fem_test"))
    def test_console_Bernoulli_1b(self, N, L, a1, a2, a3):
        angles = np.array([a1, a2, a3])
        assert test_console_1(L=L, nElem=N, angles=angles, tol=1e-3)
        
    def test_console_Bernoulli_2a(self):
        assert test_console_2(tol=1e-3)
        
    @given(st.integers(min_value=5, max_value=20), 
           st.floats(min_value=3000., max_value=3500.),
           st.floats(min_value=0., max_value=2*PI))
    @settings(settings.load_profile("fem_test"))
    def test_console_Bernoulli_2b(self, N, L, rotX):
        assert test_console_2(L=L, nElem=N, rotX=rotX, tol=1e-3)
        
    
if __name__ == "__main__":
    
    #print(test_console_1())
    #print(test_console_2(rotX=90))
    unittest.main()