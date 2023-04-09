from typing import Iterable

import numpy as np
from numpy import ndarray

from neumann import atleastnd
from polymesh.abcdata import ABCMeta_MeshData
from polymesh.cell import PolyCell

from ..dofmap import DOF
from ...utils.fem.fem import (
    expand_shape_function_matrix_bulk,
    element_dofmap_bulk,
)


class FemMixin:
    dofs = ()
    dofmap = ()

    # optional
    qrule: str = None
    quadrature = None

    # advanced settings
    compatible = True

    # must be reimplemented
    NNODE: int = None  # number of nodes, normally inherited

    # from the mesh object
    # it affects sizing of all kinds of intermediate variables
    # during calculations (eg. element transformation matrix)
    NDOFN: int = None  # numper of dofs per node, normally

    # inherited form the model object
    NDIM: int = None  # number of geometrical dimensions,
    # normally inherited from the mesh object

    # inherited form the model object
    NSTRE: int = None  # number of internal force components,

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def geometry(self):
        pd, cd = self.pdb, self.db
        return self.__class__.Geometry(pointdata=pd, celldata=cd)

    @property
    def pdb(self):
        return self.pointdata

    @property
    def db(self):
        return self._wrapped

    def topology(self) -> ndarray:
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    @classmethod
    def lcoords(cls, *args, **kwargs) -> ndarray:
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    @classmethod
    def lcenter(cls, *args, **kwargs) -> ndarray:
        raise NotImplementedError

    def local_coordinates(self) -> ndarray:
        # implemented at PolyCell
        raise NotImplementedError

    def points_of_cells(self) -> ndarray:
        # implemented at PolyCell
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    # !TODO : can be reimplemented if the element is not IsoP
    def jacobian_matrix(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    # !TODO : can be reimplemented if the element is not IsoP
    def jacobian(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError

    # !TODO : this should be implemented at material
    def strain_displacement_matrix(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    def shape_function_values(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    # !TODO : can be reimplemented if the element is not IsoP
    def shape_function_matrix(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    def shape_function_derivatives(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError

    # !TODO : this should be implemented at material
    def elastic_material_stiffness_matrix(self) -> ndarray:
        raise NotImplementedError

    # !TODO : this should be implemented at material
    def stresses_at_centers(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError

    # !TODO : this should be implemented at material
    def stresses_at_cells_nodes(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError

    def direction_cosine_matrix(self) -> ndarray:
        return None

    def integrate_body_loads(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError

    def masses(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError

    def areas(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError

    def lengths(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError

    def volumes(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError

    def densities(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError

    def weights(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError

    def dof_solution(self, *, cells: Iterable, flatten: bool) -> ndarray:
        raise NotImplementedError

    def kinetic_strains(self, *, cells: Iterable, points: Iterable) -> ndarray:
        raise NotImplementedError
    
    def centers(self, *args, **kwargs) -> ndarray:
        raise NotImplementedError


class FemMaterial(FemMixin):
    """
    Base class for FEM material classes.
    """


class MetaFiniteElement(ABCMeta_MeshData):
    """
    Python metaclass for safe inheritance. Throws a TypeError
    if a method tries to shadow a definition in any of the base
    classes.
    """

    def __init__(self, name, bases, namespace, *args, **kwargs):
        super().__init__(name, bases, namespace, *args, **kwargs)

    def __new__(metaclass, name, bases, namespace, *args, **kwargs):
        cls = super().__new__(metaclass, name, bases, namespace, *args, **kwargs)
        for base in bases:
            if issubclass(base, PolyCell):
                cls.Geometry = base
            if issubclass(base, FemMaterial):
                if len(base.dofs) > 0:
                    cls.Material = base
                    cls.dofs = base.dofs
                    cls.NDOFN = len(cls.dofs)
                    cls.dofmap = np.array(DOF.dofmap(cls.dofs), dtype=int)
        return cls


class ABCFiniteElement(FemMixin, metaclass=MetaFiniteElement):
    """
    Helper class that provides a standard way to create an ABC using
    inheritance.
    """

    __slots__ = ()
    dofs = ()
    dofmap = ()

    def shape_function_matrix(self, *args, expand: bool = False, **kwargs):
        N = super().shape_function_matrix(*args, **kwargs)
        if expand:
            constant_metric = len(N.shape) == 3
            N = atleastnd(N, 4, front=True)
            nDOFN = self.container.NDOFN
            dofmap = self.__class__.dofmap
            if len(dofmap) < nDOFN:
                # the model has more dofs than the element
                nE, nP, nX = N.shape[:3]
                nTOTV = nDOFN * self.NNODE
                # nE, nP, nX, nDOF * nNE
                N_ = np.zeros((nE, nP, nX, nTOTV), dtype=float)
                dofmap = element_dofmap_bulk(dofmap, nDOFN, self.NNODE)
                N = expand_shape_function_matrix_bulk(N, N_, dofmap)
            return N[0] if constant_metric else N
        else:
            return N
