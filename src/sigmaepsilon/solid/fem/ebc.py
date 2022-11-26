# -*- coding: utf-8 -*-
from typing import Iterable, Union, Tuple
from abc import abstractmethod
import numpy as np
from numpy import ndarray, concatenate as conc
from scipy.sparse import coo_matrix as coo

from neumann.linalg import ReferenceFrame
from neumann.array import atleast1d, atleast2d, repeat_diagonal_2d
from polymesh.space import index_of_closest_point

from .mesh import FemMesh
from .dofmap import DOF
from .constants import DEFAULT_DIRICHLET_PENALTY


__all__ = ['NodalSupport']


class EssentialBoundaryCondition:
    """
    Base class for Dirichlet boundary conditions accounted for
    using Courant-type penalization. 
    """
    
    @abstractmethod
    def assemble(self, mesh:FemMesh) -> Tuple[coo, ndarray]:
        ...


class NodalSupport(EssentialBoundaryCondition):
    """
    A class to handle nodal supports.
    
    Parameters
    ----------
    x : Iterable, Optional
        An iterable expressing the location of one or more points. 
        Use it if you don't know the index of the point of application 
        in advance. Default is None.
        
    i : Iterable[int] or int, Optional
        The index of one or more points of in a pointcloud. 
        Default is None.
            
    values : dict, Optional
        Dictionary to define the prescribed values. Valid keys are
        'UX', 'UY', 'UZ', 'ROTX', 'ROTY', 'ROTZ'. Default is None.
    
    frame : numpy.ndarray or ReferenceFrame, Optional
        Use it do define inclined supports. If it is a NumPy array, 
        it must be a 3x3 matrix representing a set of orthonormal base 
        vectors. A None value means that the prescribed values are understood
        in the frame of the mesh the constraints are applied to. 
        Default is None.
        
    penalty : float, Optional
        Penalty value for Courant-type penalization. 
        Default is `sigmaepsilon.solid.fem.constants.DEFAULT_DIRICHLET_PENALTY`.
        
    **kwargs : dict, Optional
        Keyword arguments used for elementwise definition of 'values'.
        
    Examples
    --------
    To define a support at one node
    
    >>> from sigmaepsilon.solid.fem import NodalSupport
    >>> support = NodalSupport(x=[10, 0, 0], UZ=1.)
    
    To define an inclined support with nonzero prescribed displacements:
    
    >>> from polymesh.space import StandardFrame
    >>> GlobalFrame = StandardFrame(dim=3)
    >>> frame = GlobalFrame.rotate_new('Body', [0, angle, 0], 'XYZ')
    >>> support = NodalSupport(x=[10, 0, 0], frame=frame, UX=1., UZ=1.)
    
    To apply the same constrain at multiple locations:
    
    >>> support = NodalSupport(x=[[0, 0, 0], [L, 0, 0]], UX = 0., UY=0.)
    
    To define a support with a custom penalty:
    
    >>> support = NodalSupport(x=[[0, 0, 0], [L, 0, 0]], 
    >>>                        UX = 0., UY=0., penalty=1e12)
            
    """

    def __init__(self, data:dict=None, *, x: Iterable = None, i: int = None,
                 frame: Union[ndarray, ReferenceFrame] = None, 
                 penalty : float = DEFAULT_DIRICHLET_PENALTY, **kwargs):
        self.x = x if x is None else np.array(x, dtype=float)
        self.i = i if i is None else np.array(i, dtype=int)
        self.penalty = penalty
        self.data = data
        self.frame = frame
        if data is None:
            self.data = kwargs
        self.dofmap = DOF.dofmap(self.data.keys())
        if not isinstance(self.i, int):
            assert self.x is not None, "Index or position must be defined!"
        if self.frame is not None:
            if not isinstance(self.frame, ReferenceFrame):
                raise TypeError("Invalid frame type. Read the " + \
                    "docs of the NodalSupport class.")
        else:
            self.frame = ReferenceFrame(dim=3)
            
    def assemble(self, mesh:FemMesh) -> Tuple[coo, ndarray]:
        """
        Returns the penalty stiffness matrix and the penalty load matrix.
        """
        i = None
        if self.i is None:
            coords = mesh.coords()
            x = atleast2d(self.x, front=True)
            i = index_of_closest_point(coords, x)
        else:
            i = atleast1d(self.i)
        nI = len(i)
        nDOF = mesh.NDOFN
        nN = len(mesh.pointdata)
        N = nDOF * nN
        c, r = divmod(nDOF, 3)
        assert r == 0, "The number of deegrees of freedom per" \
            + " node must be a multiple of 3."
        dcm = self.frame.dcm(source=mesh.frame)
        nodal_dcm = repeat_diagonal_2d(dcm, c)[self.dofmap]
        factors = repeat_diagonal_2d(nodal_dcm, nI)
        dofs = np.arange(nDOF)
        inds = conc([dofs + (i_ * nDOF) for i_ in i])
        kdata = (factors.T @ factors).flatten()
        krows = np.repeat(inds, len(inds))
        kcols = np.tile(inds, len(inds))
        Kp = self.penalty * coo((kdata, (krows, kcols)), shape=(N, N))
        fp = np.zeros(N, dtype=float)
        fdata = factors.T @ np.tile(list(self.data.values()), nI)
        fp[inds] = self.penalty * fdata
        Kp.eliminate_zeros()
        return Kp, fp
