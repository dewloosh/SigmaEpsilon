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
    
    def test_trimesh(self):
        TriMesh(size=(800, 600), shape=(10, 10))
        circular_disk(120, 60, 5, 25)
        
        gridparams = {
            'size' : (1200, 600),
            'shape' : (30, 15),
            'eshape' : (2, 2),
            'origo' : (0, 0),
            'start' : 0
        }
        coordsQ4, topoQ4 = grid(**gridparams)
        points, triangles = Q4_to_T3(coordsQ4, topoQ4, path='grid')
        triangulate(points=points[:, :2], triangles=triangles)
        
    def test_grid(self):
        Grid(size=(80, 60, 20), shape=(8, 6, 2), eshape='H8')
        
        Grid(size=(80, 60, 20), shape=(8, 6, 2), eshape='H27')
        
        Lx, Ly, Lz = 800, 600, 100
        nx, ny, nz = 8, 6, 2
        xbins = np.linspace(0, Lx, nx+1)
        ybins = np.linspace(0, Ly, ny+1)
        zbins = np.linspace(0, Lz, nz+1)
        bins = xbins, ybins, zbins
        coords, topo = grid(bins=bins, eshape='H8')
        PolyData(coords=coords, topo=topo, celltype=H8)
        
        gridparams = {
            'size' : (1200, 600),
            'shape' : (30, 15),
            'eshape' : (2, 2),
            'origo' : (0, 0),
            'start' : 0
        }
        grid(**gridparams)
        
        gridparams = {
            'size' : (1200, 600),
            'shape' : (30, 15),
            'eshape' : (4, 2),
            'origo' : (0, 0),
            'start' : 1
        }
        grid(**gridparams)
        
        gridparams = {
            'size' : (1200, 600),
            'shape' : (30, 15),
            'eshape' : (3, 6),
            'origo' : (1., 1.),
            'start' : 5
        }
        grid(**gridparams)
        
    def test_voxelize(self):
        h, a, b = 0.8, 1.5, 0.5  
        voxelize_cylinder(radius=[b/2, a/2], height=h, size=h/20)
        
    def test_extrude(self):
        A = CartesianFrame(dim=3)
        n_angles = 120
        n_radii = 60
        min_radius = 5
        max_radius = 25
        h = 20
        zres = 20
        disk = circular_disk(n_angles, n_radii, min_radius, max_radius)        
        points = disk.coords()
        triangles = disk.topology()
        points, triangles = detach_mesh_bulk(points, triangles)            
        coords, topo = extrude_T3_TET4(points, triangles, h, zres)       
        PolyData(coords=coords, topo=topo, celltype=TET4, frame=A)


if __name__ == "__main__":

    unittest.main()
