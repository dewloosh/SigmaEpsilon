import numpy as np

from sigmaepsilon.fem import Structure

from .utils import (
    _periodic_essential_bc, 
    _strain_field_mindlin_3d_bulk,
    _postproc_3d_gauss_stresses,
    _calc_avg_ABDS_3d,
    )


class RepresentativeVolumeElement(Structure):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def to_Mindlin(self):
        mesh = self.mesh
        NDOFN = mesh.NDOFN
        NSTRE = 8
        
        # mesh data
        points = mesh.points()
        [(xmin, xmax), (ymin, ymax), (_, _)] = points.bounds()
        center = points.center()
        area = (xmax - xmin) * (ymax - ymin)
        
        # nodal loads
        nodal_loads = np.zeros((len(points), NDOFN, NSTRE), dtype=float)
        mesh.pd.loads = nodal_loads
        
        # nodal supports against rigid body motion
        nodal_fixity = np.zeros((len(points), NDOFN), dtype=bool)
        i = points.index_of_closest(center)
        nodal_fixity[i, :3] = True
        i = points.index_of_closest([xmax, 0, 0])
        nodal_fixity[i, [1, 2]] = True
        i = points.index_of_closest([0, ymax, 0])
        nodal_fixity[i, 2] = True
        mesh.pd.fixity = nodal_fixity
        
        # nodal supports to guarantee peridic displacement solution
        periodicity_constraints = _periodic_essential_bc(mesh, axis=[0, 1])
        
        # initial strain loads
        blocks = list(mesh.cellblocks(inclusive=True))
        for block in blocks:
            centers = block.centers()
            if num_block_dim := block.cd.NDIM == 3:        
                strain_loads = np.zeros((centers.shape[0], block.cd.NSTRE, NSTRE))
                _strain_field_mindlin_3d_bulk(centers, strain_loads)
                block.cd.strain_loads = strain_loads
            elif num_block_dim == 2:
                raise NotImplementedError
            else:
                raise NotImplementedError
        
        # solve BVP
        self.clear_constraints()
        self.constraints.append(periodicity_constraints)
        self.linear_static_analysis()
        
        # postproc
        ABDS = np.zeros((NSTRE, NSTRE), dtype=float)
        ABDS_avg = np.zeros_like(ABDS)
        for block in blocks:
            block.cd.strain_loads = None
            if num_block_dim := block.cd.NDIM == 3:
                gp, gw = block.cd.quadrature["full"]
                cell_forces = block.internal_forces(points=gp, flatten=False)        
                ec = block.cells_coords(points=gp)
                dshp = block.cd.shape_function_derivatives(gp)
                jac = block.cd.jacobian_matrix(dshp=dshp, ecoords=ec)
                djac = block.cd.jacobian(jac=jac)
                _postproc_3d_gauss_stresses(cell_forces, ec, gw, djac, ABDS)
                hooke = block.cd.elastic_material_stiffness_matrix()
                _calc_avg_ABDS_3d(hooke, ec, gw, djac, ABDS_avg)
            elif num_block_dim == 2:
                raise NotImplementedError
            else:
                raise NotImplementedError
        
        ABDS /= area
        ABDS_avg /= area
        result = np.zeros_like(ABDS)
        result[:6, :6] = ABDS_avg[:6, :6] - ABDS[:6, :6]
        result[6:8, 6:8] = ABDS[6:8, 6:8]
        result[6:8, 6:8] *= 5/6        
        return result        
    
    