from .fem.mesh import FemMesh, SolidMesh
from .fem.pointdata import PointData
from .fem.structure import Structure
from .fem.linemesh import LineMesh
from .fem.surfacemesh import SurfaceMesh
from .material.beam.bernoulli.section import BeamSection

__all__ = [
    "FemMesh", 
    "SolidMesh", 
    "PointData",
    "Structure", 
    "LineMesh", 
    "SurfaceMesh", 
    "BeamSection"
]

__version__ = "0.0.35"
__description__ = "High-Performance Computational Mechanics in Python."
__project_name__ = "SigmaEpsilon"  # for Sphinx
