# -*- coding: utf-8 -*-
from abc import abstractmethod
from inspect import signature, Parameter
import numpy as np

from polymesh.abc import ABCMeta_MeshData
from polymesh.cell import PolyCell

from copy import deepcopy
from functools import partial
from typing import Callable, Any


dofs = ('UX', 'UY', 'UZ', 'ROTX', 'ROTY', 'ROTZ')
dofmap = {d : i for i, d in enumerate(dofs)}


class FemMixin:
    
    dofs = ()
    dofmap = ()
            
    # must be reimplemented
    NNODE: int = None  # number of nodes, normally inherited

    # from the mesh object
    NDOFN: int = None  # numper of dofs per node, normally

    # inherited form the model object
    NDIM: int = None  # number of geometrical dimensions,
    # normally inherited from the mesh object

    # inherited form the model object
    NSTRE: int = None  # number of internal force components,

    # optional
    qrule: str = None
    quadrature = None

    # advanced settings
    compatible = True
    
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
    
    def topology(self):
        raise NotImplementedError
    
    # !TODO : this should be implemented at geometry
    @classmethod
    def lcoords(cls, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    @classmethod
    def lcenter(cls, *args, **kwargs):
        raise NotImplementedError
    
    def local_coordinates(self, *args, frames=None, _topo=None, **kwargs):
        # implemented at PolyCell
        raise NotImplementedError

    def points_of_cells(self):
        # implemented at PolyCell
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    # !TODO : can be reimplemented if the element is not IsoP
    def jacobian_matrix(self, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    # !TODO : can be reimplemented if the element is not IsoP
    def jacobian(self, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at model
    def strain_displacement_matrix(self, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    def shape_function_values(self, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    # !TODO : can be reimplemented if the element is not IsoP
    def shape_function_matrix(self, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    def shape_function_derivatives(self, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at model
    def model_stiffness_matrix(self, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at model
    def stresses_at_centers(self, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at model
    def stresses_at_cells_nodes(self, *args, **kwargs):
        raise NotImplementedError

    def direction_cosine_matrix(self):
        return None
    
    def integrate_body_loads(self, *args, **kwargs):
        raise NotImplementedError
    
    def masses(self, *args, **kwargs):
        raise NotImplementedError
    
    def volumes(self, *args, **kwargs):
        raise NotImplementedError
    
    def densities(self, *args, **kwargs):
        raise NotImplementedError
        
    def weights(self, *args, **kwargs):
        raise NotImplementedError
    
    def _postproc_local_internal_forces(self, forces, *args, points, rng, cells, **kwargs):
        """
        The aim of this function os to guarantee a standard shape of output, that contains
        values for each of the 6 internal force compoents, irrespective of the kinematical
        model being used.

        Example use case : the Bernoulli beam element, where, as a consequence of the absence
        of shear forces, there are only 4 internal force components, and the shear forces
        must be calculated a-posteriori.

        Parameters
        ----------
        forces : numpy array of shape (nE, nP, nSTRE, nRHS)
            4d float array of internal forces for many elements, evaluation points,
            and load cases. The number of force components (nSTRE) is dictated by the
            reduced kinematical model (eg. 4 for the Bernoulli beam).

        dofsol (nE, nEVAB, nRHS)

        Notes
        -----
        Arguments are based on the reimplementation in the Bernoulli base element.
        
        Returns
        -------
        numpy.ndarray 
            shape : (nE, nP, 6, nRHS)
        
        """
        return forces
    


class FemModel(FemMixin):
    ...
        

class FemModel1d(FemModel):
    ...


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
            if issubclass(base, FemModel):
                if len(base.dofs) > 0:
                    cls.Model = base
                    cls.dofs = base.dofs
                    cls.NDOFN = len(cls.dofs)
                    cls.dofmap = np.array([dofmap[d] for d in cls.dofs], dtype=int)
        return cls
    

class ABCFiniteElement(metaclass=MetaFiniteElement):
    """
    Helper class that provides a standard way to create an ABC using
    inheritance.
    """
    __slots__ = ()
    dofs = ()
    dofmap = ()  
