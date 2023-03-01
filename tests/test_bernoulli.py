import unittest
from neumann.linalg import Vector, linspace
from neumann import repeat, minmax
from neumann.logical import allclose
from polymesh.space import StandardFrame, PointCloud
from polymesh.utils.space import index_of_closest_point, index_of_furthest_point
from polymesh.utils.topology import L2_to_L3
from sigmaepsilon.fem.cells import B2, B3
from sigmaepsilon import Structure, LineMesh, PointData
from sigmaepsilon.fem import NodalSupport
from sigmaepsilon.utils.fem.dyn import Rayleigh_quotient, effective_modal_masses
import numpy as np
from numpy import pi as PI


class TestBernoulliLinearStatics(unittest.TestCase):
    def check_relative_error(self, v_analytic, v_fem, tol: float = 0.1):
        """
        Compares the absolute relative error between the analytic and fem
        results against the given tolerance. The tolerance is understood
        as maximum percentage.
        """
        if v_analytic < 0:
            v_analytic -= 1
            v_fem -= 1
        else:
            v_analytic += 1
            v_fem += 1
        return np.abs(100 * (v_analytic - v_fem) / v_analytic) < tol

    def test_conc_tr(self):
        """
        Straight console, with starting point at (0, 0, 0), length L, and
        random orientation of axes. The same concentrated forces and moments
        are applied at the free end in the coordinate system of the console.
        The results should be the same when understood in the frame of the
        console, independently of the random orientation angles.
        """
        L, d, t = 1000.0, 100.0, 5.0  # geometry
        Ex, nu = 210000.0, 0.25  # material
        penalty = 1e20  # penalty value for essential BCs
        np.random.RandomState()
        # section
        w, h = 5.0, 100.0  # width and height of the rectangular cross section
        A = w * h  # area
        Iy = w * h**3 / 12  # second moment of inertia around the y axis
        Iz = h * w**3 / 12  # second moment of inertia around the z axis
        Ix = Iy + Iz  # torsional inertia
        # material
        G = Ex / (2 * (1 + nu))
        Hooke = np.array(
            [
                [Ex * A, 0, 0, 0],
                [0, G * Ix, 0, 0],
                [0, 0, Ex * Iy, 0],
                [0, 0, 0, Ex * Iz],
            ]
        )

        def solve(n, angles, loads, celltype=B2):
            # space
            GlobalFrame = StandardFrame(dim=3)
            TargetFrame = GlobalFrame.rotate("Body", angles, "XYZ", inplace=False)
            # mesh
            p0 = np.array([0.0, 0.0, 0.0])
            p1 = np.array([L, 0.0, 0.0])
            coords = linspace(p0, p1, n + 1)
            points = PointCloud(coords, frame=TargetFrame)
            coords = points.show(GlobalFrame)
            topo = np.zeros((n, 2), dtype=int)
            topo[:, 0] = np.arange(n)
            topo[:, 1] = np.arange(n) + 1
            if celltype.NNODE == 3:
                coords, topo = L2_to_L3(coords, topo)
            i_first = index_of_closest_point(coords, np.array([0.0, 0.0, 0.0]))
            i_last = index_of_furthest_point(coords, np.array([0.0, 0.0, 0.0]))
            # essential boundary conditions
            fixity = np.zeros((coords.shape[0], 6)).astype(bool)
            fixity[i_first, :] = True
            fixity = fixity.astype(float) * penalty
            # natural boundary conditions
            loads = np.array(loads)
            nodal_loads = np.zeros((coords.shape[0], 6))
            vF = Vector(loads[:3], frame=TargetFrame).show(GlobalFrame)
            vM = Vector(loads[3:], frame=TargetFrame).show(GlobalFrame)
            nodal_loads[i_last, :3] = vF
            nodal_loads[i_last, 3:] = vM
            # pointdata
            pd = PointData(
                coords=coords, frame=GlobalFrame, loads=nodal_loads, fixity=fixity
            )
            # celldata
            frames = repeat(TargetFrame.axes, topo.shape[0])
            cd = celltype(topo=topo, material=Hooke, frames=frames)
            # set up mesh and structure
            mesh = LineMesh(pd, cd, frame=GlobalFrame)
            structure = Structure(mesh=mesh)
            structure.linear_static_analysis()
            dofsol = structure.nodal_dof_solution(store="dofsol")
            u = np.zeros(6)
            u[:3] = Vector(dofsol[i_last, :3], frame=GlobalFrame).show(TargetFrame)
            u[3:] = Vector(dofsol[i_last, 3:], frame=GlobalFrame).show(TargetFrame)
            reactions = structure.reaction_forces()
            r = np.zeros(6)
            r[:3] = Vector(reactions[i_first, :3], frame=GlobalFrame).show(TargetFrame)
            r[3:] = Vector(reactions[i_first, 3:], frame=GlobalFrame).show(TargetFrame)
            forces = structure.internal_forces()
            f = np.zeros(6)
            f[:3] = forces[0, 0, :3, 0]
            f[3:] = forces[0, 0, 3:, 0]
            return u, r, f

        # ######################################################
        # CONCENTRATED FORCES AT THE FREE END
        # ######################################################

        Fx, Fy, Fz = 1.0, 1.0, 1.0
        loads = [Fx, Fy, Fz, 0, 0, 0]
        n = 20  # number of elements
        cells = [B2, B3]
        # analytical solutions
        UX = Fx * L / (Ex * A)  # displacement at the free end
        UY = Fy * L**3 / (3 * Ex * Iz)  # displacement at the free end
        UZ = Fz * L**3 / (3 * Ex * Iy)  # displacement at the free end
        UXX = 0.0  # rotation at the free end
        UYY = -Fz * L**2 / (2 * Ex * Iy)  # rotation at the free end
        UZZ = Fy * L**2 / (2 * Ex * Iz)  # rotation at the free end
        RX = -Fx  # reaction force
        RY = -Fy  # reaction force
        RZ = -Fz  # reaction force
        RXX = 0.0  # reaction moment
        RYY = Fz * L  # reaction moment
        RZZ = -Fy * L  # reaction moment
        FX = Fx  # internal force at the support
        FY = Fy  # internal force at the support
        FZ = Fz  # internal force at the support
        FXX = 0.0  # internal moment at the support
        FYY = -Fz * L  # internal moment at the support
        FZZ = Fy * L  # internal moment at the support
        for cell in cells:
            for _ in range(3):
                angles = np.random.rand(3) * np.pi * 2
                u, r, f = solve(n, angles, loads, cell)
                UX_fem, UXX_fem = u[0], u[3]
                RX_fem, RXX_fem = r[0], r[3]
                FX_fem, FXX_fem = f[0], f[3]
                UY_fem, UZZ_fem = u[1], u[5]
                RY_fem, RZZ_fem = r[1], r[5]
                FY_fem, FZZ_fem = f[1], f[5]
                UZ_fem, UYY_fem = u[2], u[4]
                RZ_fem, RYY_fem = r[2], r[4]
                FZ_fem, FYY_fem = f[2], f[4]
                assert self.check_relative_error(UX, UX_fem)
                assert self.check_relative_error(UY, UY_fem)
                assert self.check_relative_error(UZ, UZ_fem)
                assert self.check_relative_error(UXX, UXX_fem)
                assert self.check_relative_error(UYY, UYY_fem)
                assert self.check_relative_error(UZZ, UZZ_fem)
                assert self.check_relative_error(RX, RX_fem)
                assert self.check_relative_error(RY, RY_fem)
                assert self.check_relative_error(RZ, RZ_fem)
                assert self.check_relative_error(RXX, RXX_fem)
                assert self.check_relative_error(RYY, RYY_fem)
                assert self.check_relative_error(RZZ, RZZ_fem)
                assert self.check_relative_error(FX, FX_fem)
                assert self.check_relative_error(FY, FY_fem)
                assert self.check_relative_error(FZ, FZ_fem)
                assert self.check_relative_error(FXX, FXX_fem)
                assert self.check_relative_error(FYY, FYY_fem)
                assert self.check_relative_error(FZZ, FZZ_fem)

        # ######################################################
        # CONCENTRATED MOMENTS AT THE FREE END
        # ######################################################

        Mx, My, Mz = 1.0, 1.0, 1.0
        loads = [0, 0, 0, Mx, My, Mz]
        n = 2  # number of elements
        cells = [B2, B3]
        # analytical solutions
        UX = 0.0  # displacement at the free end
        UY = Mz * L**2 / (2 * Ex * Iz)  # displacement at the free end
        UZ = -My * L**2 / (2 * Ex * Iy)  # displacement at the free end
        UXX = Mx * L / (G * Ix)  # rotation at the free end
        UYY = My * L / (Ex * Iy)  # rotation at the free end
        UZZ = Mz * L / (Ex * Iz)  # rotation at the free end
        RX = 0.0  # reaction force
        RY = 0.0  # reaction force
        RZ = 0.0  # reaction force
        RXX = -Mx  # reaction moment
        RYY = -My  # reaction moment
        RZZ = -Mz  # reaction moment
        FX = 0.0  # internal force at the support
        FY = 0.0  # internal force at the support
        FZ = 0.0  # internal force at the support
        FXX = Mx  # internal moment at the support
        FYY = My  # internal moment at the support
        FZZ = Mz  # internal moment at the support
        for cell in cells:
            for _ in range(3):
                angles = np.random.rand(3) * np.pi * 2
                u, r, f = solve(n, angles, loads, cell)
                UX_fem, UXX_fem = u[0], u[3]
                RX_fem, RXX_fem = r[0], r[3]
                FX_fem, FXX_fem = f[0], f[3]
                UY_fem, UZZ_fem = u[1], u[5]
                RY_fem, RZZ_fem = r[1], r[5]
                FY_fem, FZZ_fem = f[1], f[5]
                UZ_fem, UYY_fem = u[2], u[4]
                RZ_fem, RYY_fem = r[2], r[4]
                FZ_fem, FYY_fem = f[2], f[4]
                assert self.check_relative_error(UX, UX_fem)
                assert self.check_relative_error(UY, UY_fem)
                assert self.check_relative_error(UZ, UZ_fem)
                assert self.check_relative_error(UXX, UXX_fem)
                assert self.check_relative_error(UYY, UYY_fem)
                assert self.check_relative_error(UZZ, UZZ_fem)
                assert self.check_relative_error(RX, RX_fem)
                assert self.check_relative_error(RY, RY_fem)
                assert self.check_relative_error(RZ, RZ_fem)
                assert self.check_relative_error(RXX, RXX_fem)
                assert self.check_relative_error(RYY, RYY_fem)
                assert self.check_relative_error(RZZ, RZZ_fem)
                assert self.check_relative_error(FX, FX_fem)
                assert self.check_relative_error(FY, FY_fem)
                assert self.check_relative_error(FZ, FZ_fem)
                assert self.check_relative_error(FXX, FXX_fem)
                assert self.check_relative_error(FYY, FYY_fem)
                assert self.check_relative_error(FZZ, FZZ_fem)

    def test_temperature(self):
        """
        Uniform temperature load on a clamped-clamped beam cousing
        normal force only.
        """
        # units in kN, cm and Celsius
        L = 100.0
        Ex, nu = 21000.0, 0.25
        dT = 20
        alpha = 1.2e-5
        # section
        w, h = 5.0, 10.0  # width and height of the rectangular cross section
        A = w * h  # area
        Iy = w * h**3 / 12  # second moment of inertia around the y axis
        Iz = h * w**3 / 12  # second moment of inertia around the z axis
        Ix = Iy + Iz  # torsional inertia
        # material
        G = Ex / (2 * (1 + nu))
        Hooke = np.array(
            [
                [Ex * A, 0, 0, 0],
                [0, G * Ix, 0, 0],
                [0, 0, Ex * Iy, 0],
                [0, 0, 0, Ex * Iz],
            ]
        )

        def build_structure(n: int):
            # space
            GlobalFrame = StandardFrame(dim=3)
            # mesh
            p0 = np.array([0.0, 0.0, 0.0])
            p1 = np.array([L, 0.0, 0.0])
            coords = linspace(p0, p1, n + 1)
            coords = PointCloud(coords, frame=GlobalFrame).show()
            topo = np.zeros((n, 2), dtype=int)
            topo[:, 0] = np.arange(n)
            topo[:, 1] = np.arange(n) + 1
            i_first = index_of_closest_point(coords, np.array([0, 0, 0]))
            i_last = index_of_closest_point(coords, np.array([L, 0, 0]))
            # essential boundary conditions
            fixity = np.zeros((coords.shape[0], 6)).astype(bool)
            fixity[i_first, :] = True
            fixity[i_last, :] = True
            # natural boundary conditions
            nLoadCase = 2
            # nodal loads
            nodal_loads = np.zeros((coords.shape[0], 6, nLoadCase))
            # cell loads
            strain_loads = np.zeros((topo.shape[0], 4, nLoadCase))
            strain_loads[:, 0, 0] = dT * alpha
            strain_loads[:, 0, 1] = -dT * alpha
            # pointdata
            pd = PointData(
                coords=coords, fixity=fixity, loads=nodal_loads, frame=GlobalFrame
            )
            # celldata
            frames = repeat(GlobalFrame.show(), topo.shape[0])
            cd = B2(topo=topo, frames=frames, material=Hooke, strain_loads=strain_loads)
            # set up mesh and structure
            mesh = LineMesh(pd, cd, frame=GlobalFrame)
            structure = Structure(mesh=mesh)
            return structure

        structure = build_structure(2)
        structure.linear_static_analysis()
        forces = structure.internal_forces()
        FX = Ex * A * alpha * dT
        assert self.check_relative_error(-FX, forces[0, 0, 0, 0])
        assert self.check_relative_error(-FX, forces[-1, 0, 0, 0])
        assert self.check_relative_error(FX, forces[0, 0, 0, 1])
        assert self.check_relative_error(FX, forces[-1, 0, 0, 1])

    def test_vs_Navier_LinStat(self):
        """
        Tests all result components against Navier's solution for a
        sinusoidal distributed load. The beam is bent in the x-y plane.
        """
        L, w, h = 1000.0, 20.0, 80.0  # geometry
        Ex, nu = 210000.0, 0.25  # material
        # section properties
        Iy = h * w**3 / 12
        Iz = w * h**3 / 12
        Ix = (Iy + Iz) / 2
        A = w * h
        EI = Ex * Iz
        # material
        G = Ex / (2 * (1 + nu))
        Hooke = np.array(
            [
                [Ex * A, 0, 0, 0],
                [0, G * Ix, 0, 0],
                [0, 0, Ex * Iy, 0],
                [0, 0, 0, Ex * Iz],
            ]
        )
        # load
        qy_max = 1.0  # max intensity
        ny = 1  # number of half waves, should be odd

        def qy(x):
            return np.sin(PI * ny * x / L) * qy_max

        def uy(x):
            return (
                qy_max * (L**4 / (EI * PI**4 * ny**4)) * np.sin(PI * ny * x / L)
            )

        def rotz(x):
            return (
                qy_max * (L**3 / (EI * PI**3 * ny**3)) * np.cos(PI * ny * x / L)
            )

        def kz(x):
            return (
                -qy_max * (L**2 / (EI * PI**2 * ny**2)) * np.sin(PI * ny * x / L)
            )

        def mz(x):
            return -qy_max * (L**2 / (PI**2 * ny**2)) * np.sin(PI * ny * x / L)

        def vy(x):
            return qy_max * (L / (PI * ny)) * np.cos(PI * ny * x / L)

        def solve(n: int, fnc, celltype=B2):
            # space
            GlobalFrame = StandardFrame(dim=3)
            # mesh
            p0 = np.array([0.0, 0.0, 0.0])
            p1 = np.array([L, 0.0, 0.0])
            coords = linspace(p0, p1, n + 1)
            topo = np.zeros((n, 2), dtype=int)
            topo[:, 0] = np.arange(n)
            topo[:, 1] = np.arange(n) + 1
            if celltype.NNODE == 3:
                coords, topo = L2_to_L3(coords, topo)
            # mark some points
            i_first = index_of_closest_point(coords, np.array([0.0, 0.0, 0.0]))
            i_last = index_of_furthest_point(coords, np.array([0.0, 0.0, 0.0]))
            # generate load function
            fnc_loads = fnc(coords[:, 0])
            # essential boundary conditions
            penalty = 1e20  # penalty value for essential BCs
            fixity = np.zeros((coords.shape[0], 6)).astype(bool)
            fixity[i_first, [0, 1, 2, 3, 4]] = True
            fixity[i_last, [1, 2, 3, 4]] = True
            fixity = fixity.astype(float) * penalty
            # natural boundary conditions
            nodal_loads = np.zeros((coords.shape[0], 6))
            cell_loads = np.zeros((topo.shape[0], topo.shape[1], 6))
            # pointdata
            pd = PointData(
                coords=coords, frame=GlobalFrame, loads=nodal_loads, fixity=fixity
            )
            # celldata
            frames = repeat(np.eye(3), topo.shape[0])
            cd = celltype(topo=topo, material=Hooke, frames=frames)
            cell_loads[:, :, 1] = cd.pull(data=fnc_loads)[:, :, 0]
            cd.loads = cell_loads
            # set up mesh and structure
            mesh = LineMesh(pd, cd, frame=GlobalFrame)
            structure = Structure(mesh=mesh)
            structure.linear_static_analysis()
            return structure

        structure = solve(20, qy, B3)
        ui = structure.mesh.cell_dof_solution(
            points=[1 / 4, 3 / 4], rng=[0, 1], target="global", flatten=False
        )
        fi = structure.internal_forces(points=[1 / 4, 3 / 4], rng=[0, 1])
        k = structure.mesh.strains(points=[1 / 4, 3 / 4], rng=[0, 1])
        xi = structure.mesh.cells_coords(points=[1 / 4, 3 / 4], rng=[0, 1])
        u_navier = uy(xi[:, :, 0].flatten())
        u_fem = ui[:, :, 1].flatten()
        self.assertTrue(allclose(u_fem, u_navier, atol=1e-2, rtol=1e-1))
        u_navier = rotz(xi[:, :, 0].flatten())
        u_fem = ui[:, :, -1].flatten()
        self.assertTrue(allclose(u_fem, u_navier, atol=1e-5, rtol=1e-2))
        u_navier = kz(xi[:, :, 0].flatten())
        u_fem = k[:, :, -1].flatten()
        self.assertTrue(allclose(u_fem, u_navier, atol=1e-8, rtol=1e-2))
        u_navier = mz(xi[:, :, 0].flatten())
        u_fem = fi[:, :, -1].flatten()
        self.assertTrue(allclose(u_fem, u_navier, atol=600, rtol=8e-3))
        u_navier = vy(xi[:, :, 0].flatten())
        u_fem = fi[:, :, 1].flatten()
        self.assertTrue(allclose(u_fem, u_navier, atol=8, rtol=0.3))

    def test_Dirichlet(self):
        """
        Tests linear static results when the essential boundary conditions
        are formulated with 'fixity' or by using objects. It is checked if
        the displacement results from the two approaches are the same or not.
        """
        L = 100.0  # length of the console
        w, h = 10.0, 10.0  # width and height of the rectangular cross section
        F = -100.0  # value of the vertical load at the free end
        E = 210000.0  # Young's modulus
        nu = 0.3  # Poisson's ratio
        # cross section
        A = w * h  # area
        Iy = w * h**3 / 12  # second moment of inertia around the y axis
        Iz = h * w**3 / 12  # second moment of inertia around the z axis
        Ix = Iy + Iz  # torsional inertia
        # model stiffness matrix
        G = E / (2 * (1 + nu))
        Hooke = np.array(
            [[E * A, 0, 0, 0], [0, G * Ix, 0, 0], [0, 0, E * Iy, 0], [0, 0, 0, E * Iz]]
        )
        # space
        GlobalFrame = StandardFrame(dim=3)
        # mesh
        nElem = 2  # number of finite elements to use
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([L, 0.0, 0.0])
        coords = linspace(p0, p1, nElem + 1)
        coords = PointCloud(coords, frame=GlobalFrame).show()
        topo = np.zeros((nElem, 2), dtype=int)
        topo[:, 0] = np.arange(nElem)
        topo[:, 1] = np.arange(nElem) + 1
        # load at the rightmost node
        nodal_loads = np.zeros((coords.shape[0], 6))
        nodal_loads[-1, 2] = F

        # ####################################################
        # DEFINE ESSENTIAL BOUNDARY CONDITIONS THROUGH FIXITY
        # ####################################################

        # support at the leftmost node
        fixity = np.zeros((coords.shape[0], 6)).astype(bool)
        fixity[0, :] = True
        # pointdata
        pd = PointData(
            coords=coords, frame=GlobalFrame, loads=nodal_loads, fixity=fixity
        )
        # celldata
        frames = repeat(GlobalFrame.show(), topo.shape[0])
        cd = B2(topo=topo, material=Hooke, frames=frames)
        # set up mesh and structure
        mesh = LineMesh(pd, cd, frame=GlobalFrame)
        structure = Structure(mesh=mesh)
        # solve
        structure.linear_static_analysis()
        dofsol = structure.nodal_dof_solution()
        umin, umax = minmax(dofsol)

        # ####################################################
        # DEFINE ESSENTIAL BOUNDARY CONDITIONS USING OBJECTS
        # ####################################################

        # support at the leftmost node
        support = NodalSupport(
            x=[0, 0, 0], UX=0.0, UY=0.0, UZ=0.0, UXX=0.0, UYY=0.0, UZZ=0.0
        )
        # pointdata
        pd = PointData(coords=coords, frame=GlobalFrame, loads=nodal_loads)
        # celldata
        frames = repeat(GlobalFrame.show(), topo.shape[0])
        cd = B2(topo=topo, material=Hooke, frames=frames)
        # set up mesh and structure
        mesh = LineMesh(pd, cd, frame=GlobalFrame)
        structure = Structure(mesh=mesh)
        # add the nodal constraints
        structure.constraints.append(support)
        # solve and check
        structure.linear_static_analysis()
        dofsol = structure.nodal_dof_solution()
        _umin, _umax = minmax(dofsol)
        self.assertTrue(allclose(_umin, umin, atol=1e-5))
        self.assertTrue(allclose(_umax, umax, atol=1e-5))

    def test_temperature_inclined(self):
        """
        Simply supported beam with an inclined support at the right end
        and a vertical concentrated force at the middle. The values of
        the normal forces are measured against analytic solution.
        """
        L = 100.0  # length of the console
        E = 210000.0  # Young's modulus
        nu = 0.3  # Poisson's ratio
        F = -100.0  # value of the vertical load at the middle
        # cross section
        w, h = 10.0, 10.0  # width and height of the rectangular cross section
        A = w * h  # area
        Iy = w * h**3 / 12  # second moment of inertia around the y axis
        Iz = h * w**3 / 12  # second moment of inertia around the z axis
        Ix = Iy + Iz  # torsional inertia
        # model stiffness matrix
        G = E / (2 * (1 + nu))
        Hooke = np.array(
            [[E * A, 0, 0, 0], [0, G * Ix, 0, 0], [0, 0, E * Iy, 0], [0, 0, 0, E * Iz]]
        )
        # space
        GlobalFrame = StandardFrame(dim=3)
        # mesh
        nElem = 2  # number of finite elements to use
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([L, 0.0, 0.0])
        coords = linspace(p0, p1, nElem + 1)
        coords = PointCloud(coords, frame=GlobalFrame).show()
        topo = np.zeros((nElem, 2), dtype=int)
        topo[:, 0] = np.arange(nElem)
        topo[:, 1] = np.arange(nElem) + 1
        # concentrated load at the middle
        x_middle = np.array([L / 2, 0, 0], dtype=float)
        i = index_of_closest_point(coords, x_middle)
        nodal_loads = np.zeros((coords.shape[0], 6))
        nodal_loads[i, 2] = F
        # hinged support at the leftmost node
        support_left = NodalSupport(
            x=[0, 0, 0], UX=0.0, UY=0.0, UZ=0.0, UZZ=0.0, UXX=0.0
        )
        # inclined roller support at the rightmost node
        angle = -30 * PI / 180
        frame = GlobalFrame.rotate_new("Body", [0, angle, 0], "XYZ")
        support_right = NodalSupport(x=[L, 0, 0], frame=frame, UZ=0.0)
        # pointdata
        pd = PointData(coords=coords, frame=GlobalFrame, loads=nodal_loads)
        # celldata
        frames = repeat(GlobalFrame.show(), topo.shape[0])
        cd = B2(topo=topo, material=Hooke, frames=frames)
        # set up mesh and structure
        mesh = LineMesh(pd, cd, frame=GlobalFrame)
        structure = Structure(mesh=mesh)
        # add the nodal constraints
        structure.constraints.append(support_left)
        structure.constraints.append(support_right)
        # solve and check
        structure.linear_static_analysis()
        f = structure.internal_forces(flatten=False)[:, :, 0]
        fmin, fmax = minmax(f)
        self.assertTrue(allclose(fmin, -28.867, atol=1e-3, rtol=None))
        self.assertTrue(allclose(fmax, -28.867, atol=1e-3, rtol=None))


class TestBernoulliNaturalVibrations(unittest.TestCase):
    def test_console(self):
        """
        Simple console with distributed mass. The finite element approximation
        is measured agains a lower (Dunkerley) and an upper bound (Rayleigh)
        estimation. The integrated mass is also verified.
        """
        # all units in kN and cm
        L = 150.0  # length of the console [cm]
        F = 1.0  # value of the vertical load at the free end [kN]
        E = 21000.0  # Young's modulus [kN/cm3]
        nu = 0.3  # Poisson's ratio [-]
        w, h = 5.0, 15.0  # width and height of the rectangular cross section
        g = 9.81  # gravitational acceleration [m/s2]
        density = 7750.0 * 1e-6  # mass density [g/cm3]
        nElem = 20  # number of subdivisons to use
        # calculated data
        A = w * h  # area
        Iy = w * h**3 / 12  # second moment of inertia around the y axis
        Iz = h * w**3 / 12  # second moment of inertia around the z axis
        Ix = Iy + Iz  # torsional inertia
        weight = density * 1e-3 * g  # [kN/cm3]
        dpa = density * A  # density per area [g/cm]
        wpa = weight * A  # weight per area [kN/cm]
        mass = L * dpa  # total mass

        def Dunkerley(N: int = 1):
            mi = mass / N
            h = L / N
            i1 = np.sum(list(range(1, N + 1)))
            i3 = np.sum([i**3 for i in range(1, N + 1)])
            acc_x = i1 * mi * h / (E * A)
            acc_y = i3 * mi * h**3 / (3 * E * Iz)
            acc_z = i3 * mi * h**3 / (3 * E * Iy)
            return np.sqrt(1 / acc_x), np.sqrt(1 / acc_y), np.sqrt(1 / acc_z)

        def Rayleigh(structure: Structure):
            M = structure.mass_matrix(penalize=True)
            u = structure.nodal_dof_solution(flatten=True)
            f = structure.nodal_forces(flatten=True)
            return np.sqrt(Rayleigh_quotient(M, u=u, f=f))

        # model stiffness matrix
        G = E / (2 * (1 + nu))
        Hooke = np.array(
            [[E * A, 0, 0, 0], [0, G * Ix, 0, 0], [0, 0, E * Iy, 0], [0, 0, 0, E * Iz]]
        )
        # space
        GlobalFrame = StandardFrame(dim=3)
        # mesh
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([L, 0.0, 0.0])
        coords = linspace(p0, p1, nElem + 1)
        coords = PointCloud(coords, frame=GlobalFrame).show()
        topo = np.zeros((nElem, 2), dtype=int)
        topo[:, 0] = np.arange(nElem)
        topo[:, 1] = np.arange(nElem) + 1
        # load at the rightmost node in Y and Z directions
        nodal_loads = np.zeros((coords.shape[0], 6, 3))
        global_load_vector = Vector([F, 0, 0.0], frame=GlobalFrame).show()
        nodal_loads[-1, :3, 0] = global_load_vector
        global_load_vector = Vector([0.0, F, 0.0], frame=GlobalFrame).show()
        nodal_loads[-1, :3, 1] = global_load_vector
        global_load_vector = Vector([0.0, 0, F], frame=GlobalFrame).show()
        nodal_loads[-1, :3, 2] = global_load_vector
        # support at the leftmost node (all degrees)
        fixity = np.zeros((coords.shape[0], 6)).astype(bool)
        fixity[0, :] = True
        fixity = fixity.astype(float) * 1e40
        # mass and density
        nodal_masses = np.zeros((coords.shape[0],))
        densities = np.full((topo.shape[0],), dpa)
        # pointdata
        pd = PointData(
            coords=coords, loads=nodal_loads, fixity=fixity, mass=nodal_masses
        )
        # celldata
        frames = repeat(GlobalFrame.show(), topo.shape[0])
        cd = B2(topo=topo, frames=frames, material=Hooke, density=densities)
        # set up mesh and structure
        mesh = LineMesh(pd, cd, frame=GlobalFrame)
        structure = Structure(mesh=mesh)
        # perform linear analysis
        structure.linear_static_analysis()
        # calculate natural vibrations
        structure.cleanup()
        structure.free_vibration_analysis(normalize=True, as_dense=True)
        f, v = structure.natural_circular_frequencies(return_vectors=True)
        # calculate effective masses
        nN, nD = len(coords), 6
        action_x = np.zeros((nN, nD))
        action_x[:, 0] = 1.0
        action_x = action_x.reshape(nN * nD)
        action_y = np.zeros((nN, nD))
        action_y[:, 1] = 1.0
        action_y = action_y.reshape(nN * nD)
        action_z = np.zeros((nN, nD))
        action_z[:, 2] = 1.0
        action_z = action_z.reshape(nN * nD)
        actions = np.stack([action_x, action_y, action_z], axis=1)
        M = structure.mass_matrix(penalize=False)
        m_eff = effective_modal_masses(M, actions, v)
        mass_fem = np.sum(m_eff[:, 0]), np.sum(m_eff[:, 1]), np.sum(m_eff[:, 2])
        mass_percentages = np.array([100 * m / mass for m in mass_fem])
        self.assertTrue(np.all(mass_percentages > 98.0))
        # check values
        imax = [np.argmax(m_eff[:, i]) for i in range(3)]
        f_lower = Dunkerley(20)
        f_upper = Rayleigh(structure)
        f_fem = f[imax]
        c = [f_fem[i] > f_lower[i] and f_fem[i] < f_upper[i] for i in range(3)]
        self.assertTrue(np.all(c))
        # check mass
        mass_fem = structure.mesh.mass()
        self.assertTrue(allclose(mass, mass_fem, atol=1e-3, rtol=None))

    def test_beam_ss(self, celltype=B3):
        """
        Simply supported beam with a hinge at the midspan. The essential
        boundary conditions are specified with using both 'fixity' and
        objects. Mass calculation is checked.
        """
        # all units in kN and cm
        L = 150.0  # length of the console [cm]
        F = 1.0  # value of the vertical load at the free end [kN]
        E = 21000.0  # Young's modulus [kN/cm3]
        nu = 0.3  # Poisson's ratio [-]
        w, h = 5.0, 15.0  # width and height of the rectangular cross section
        g = 9.81  # gravitational acceleration [m/s2]
        density = 7750.0 * 1e-6  # mass density [g/cm3]
        nElem = 4  # number of subdivisons to use
        # calculated data
        A = w * h  # area
        Iy = w * h**3 / 12  # second moment of inertia around the y axis
        Iz = h * w**3 / 12  # second moment of inertia around the z axis
        Ix = Iy + Iz  # torsional inertia
        weight = density * 1e-3 * g  # [kN/cm3]
        dpa = density * A  # density per area [g/cm]
        wpa = weight * A  # weight per area [kN/cm]
        mass = L * dpa  # total mass
        # model stiffness matrix
        G = E / (2 * (1 + nu))
        Hooke = np.array(
            [[E * A, 0, 0, 0], [0, G * Ix, 0, 0], [0, 0, E * Iy, 0], [0, 0, 0, E * Iz]]
        )
        # space
        GlobalFrame = StandardFrame(dim=3)
        # mesh
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([L, 0.0, 0.0])
        coords = linspace(p0, p1, nElem + 1)
        coords = PointCloud(coords, frame=GlobalFrame).show()
        topo = np.zeros((nElem, 2), dtype=int)
        topo[:, 0] = np.arange(nElem)
        topo[:, 1] = np.arange(nElem) + 1
        if celltype.NNODE == 3:
            coords, topo = L2_to_L3(coords, topo)
        # load at the rightmost node in Y and Z directions
        nodal_loads = np.zeros((coords.shape[0], 6, 3))
        global_load_vector = Vector([F, 0, 0.0], frame=GlobalFrame).show()
        nodal_loads[-1, :3, 0] = global_load_vector
        global_load_vector = Vector([0.0, F, 0.0], frame=GlobalFrame).show()
        nodal_loads[-1, :3, 1] = global_load_vector
        global_load_vector = Vector([0.0, 0, F], frame=GlobalFrame).show()
        nodal_loads[-1, :3, 2] = global_load_vector
        # support at the leftmost node (all degrees)
        fixity = np.zeros((coords.shape[0], 6)).astype(bool)
        fixity[0, :] = True
        fixity = fixity.astype(float) * 1e12
        # support at the rightmost node
        support = NodalSupport(
            x=[L, 0, 0], UX=0.0, UY=0.0, UZ=0.0, UXX=0.0, UYY=0.0, UZZ=0.0
        )
        # mass and density
        nodal_masses = np.zeros((coords.shape[0],))
        densities = np.full((topo.shape[0],), dpa)
        # pointdata
        pd = PointData(
            coords=coords, loads=nodal_loads, fixity=fixity, mass=nodal_masses
        )
        # cell fixity / end releases
        cell_fixity = np.full((topo.shape[0], topo.shape[1], 6), True)
        cell_fixity[int(nElem / 2 - 1), -1, 4] = False
        # celldata
        frames = repeat(GlobalFrame.show(), topo.shape[0])
        cd = celltype(
            topo=topo,
            frames=frames,
            density=densities,
            fixity=cell_fixity,
            material=Hooke,
        )
        # set up mesh and structure
        mesh = LineMesh(pd, cd, frame=GlobalFrame)
        structure = Structure(mesh=mesh, essential_penalty=1e12, mass_penalty_ratio=1e5)
        structure.constraints.append(support)
        # perform vinration analysis
        structure.free_vibration_analysis(normalize=True, as_dense=True)
        structure.natural_circular_frequencies(return_vectors=True)
        # check mass
        mass_fem = structure.mesh.mass()
        self.assertTrue(allclose(mass, mass_fem, atol=1e-3, rtol=None))


if __name__ == "__main__":
    unittest.main()
