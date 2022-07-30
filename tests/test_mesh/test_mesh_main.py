# -*- coding: utf-8 -*-
import numpy as np
import unittest

from sigmaepsilon.mesh import TriMesh, grid, Grid, PolyData, CartesianFrame, LineData
from sigmaepsilon.mesh.recipes import circular_disk, cylinder
from sigmaepsilon.mesh.voxelize import voxelize_cylinder
from sigmaepsilon.mesh.cells import H8, TET4
from sigmaepsilon.mesh.topo import detach_mesh_bulk
from sigmaepsilon.mesh.extrude import extrude_T3_TET4
from sigmaepsilon.mesh.topo.tr import Q4_to_T3, H8_to_L2, H8_to_TET4
from sigmaepsilon.mesh.tri.trimesh import triangulate
from sigmaepsilon.math.array import minmax


class TestMeshing(unittest.TestCase):
    
    def test_basic_example_vox(self):
        min_radius = 5
        max_radius = 25
        h = 50
        angle=1
        shape = (min_radius, max_radius), angle, h
        frame = CartesianFrame(dim=3)
        cyl = cylinder(shape, size=10., voxelize=True, frame=frame)

        coords = cyl.coords()
        topo = cyl.topology()
        centers = cyl.centers()

        cxmin, cxmax = minmax(centers[:, 0])
        czmin, czmax = minmax(centers[:, 2])
        cxavg = (cxmin + cxmax)/2
        czavg = (czmin + czmax)/2
        b_upper = centers[:, 2] > czavg
        b_lower = centers[:, 2] <= czavg
        b_left = centers[:, 0] < cxavg
        b_right = centers[:, 0] >= cxavg
        iL2 = np.where(b_upper & b_right)[0]
        iTET4 = np.where(b_upper & b_left)[0]
        iH8 = np.where(b_lower)[0]
        _, topoL2 = H8_to_L2(coords, topo[iL2])
        _, topoTET4 = H8_to_TET4(coords, topo[iTET4])
        topoH8 = topo[iH8]
        
        mesh = PolyData(coords=coords, frame=frame)

        mesh['lines', 'L2'] = LineData(topo=topoL2, frame=frame)
        mesh['lines', 'L2'].config['pyvista', 'plot', 'color'] = 'blue'
        mesh['lines', 'L2'].config['pyvista', 'plot', 'line_width'] = 1

        mesh['bodies', 'TET4'] = PolyData(topo=topoTET4, celltype=TET4, frame=frame)
        mesh['bodies', 'TET4'].config['pyvista', 'plot', 'color'] = 'green'
        mesh['bodies', 'H8'].config['pyvista', 'plot', 'opacity'] = 0.7

        mesh['bodies', 'H8'] = PolyData(topo=topoH8, celltype=H8, frame=frame)
        mesh['bodies', 'H8'].config['pyvista', 'plot', 'color'] = 'red'
        mesh['bodies', 'H8'].config['pyvista', 'plot', 'show_edges'] = True