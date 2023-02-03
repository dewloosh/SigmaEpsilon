from .structure import Structure
from ..mesh import FemMesh


class Domain(Structure):
    def __init__(self, *args, mesh: FemMesh = None, **kwargs):
        super().__init__(wrap=mesh)


class VariableThicknessSheet(Domain):
    ...
