import numpy as np

from dewloosh.core.meta import ABCMeta_Weak

from .dofmap import DOF


class ABCMeta_FemMesh(ABCMeta_Weak):
    """
    Meta class for PointData and CellData classes.

    It merges attribute maps with those of the parent classes.

    """

    def __init__(self, name, bases, namespace, *args, **kwargs):
        super().__init__(name, bases, namespace, *args, **kwargs)

    def __new__(metaclass, name, bases, namespace, *args, **kwargs):
        cls = super().__new__(metaclass, name, bases, namespace, *args, **kwargs)
        if hasattr(cls, "dofs"):
            cls.NDOFN = len(cls.dofs)
            cls.dofmap = np.array(DOF.dofmap(cls.dofs), dtype=int)
        return cls


class ABC_FemMesh(metaclass=ABCMeta_FemMesh):
    """Helper class that provides a standard way to create an ABC using
    inheritance.
    """

    __slots__ = ()
