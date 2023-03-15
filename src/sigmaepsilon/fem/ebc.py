from typing import Iterable, Union, Tuple, Callable
from abc import abstractmethod
import numpy as np
from numpy import ndarray, concatenate as conc
from scipy.sparse import coo_matrix as coo

from neumann.linalg import ReferenceFrame
from neumann import atleast1d, atleast2d, repeat_diagonal_2d

from polymesh.space import PointCloud
from polymesh.cell import PolyCell3d
from polymesh.utils.space import index_of_closest_point
from polymesh.utils.tet import _pip_tet_bulk_knn_, _pip_tet_bulk_
from polymesh.utils import (
    points_of_cells, 
    cell_centers_bulk2, 
    k_nearest_neighbours,
)
from polymesh.utils.cells.utils import (
    _find_first_hits_, _find_first_hits_knn_
)

from .mesh import FemMesh
from .dofmap import DOF
from .constants import DEFAULT_DIRICHLET_PENALTY
from ..utils.fem.ebc import link_points_to_points


__all__ = ["NodalSupport", "NodeToNode", "FaceToFace", "BodyToBody"]


class EssentialBoundaryCondition:
    """
    Base class for Dirichlet boundary conditions accounted for
    using Courant-type penalization.
    """

    @abstractmethod
    def assemble(self, mesh: FemMesh) -> Tuple[coo, ndarray]:
        ...


class BodyToBody(EssentialBoundaryCondition):
    """
    Constrains the dofs of two touching bodies by gluing them together.
    
    Parameters
    ----------
    source: PolyCell
        The source body.
    target: PolyCell
        The target body.
    dofs: Iterable, Optional
        The global degrees of freedom to glue together.
    penalty: float, Optional
        The penalty value.
    lazy: bool, Optional
        Default is True.
    tol: float, Optional
        Floating point tolerance for detecting point in polygons. Default is 1e-12.
    k: int, Optional
        THe number of neighbours.
    """
    def __init__(
        self,
        source: PolyCell3d,
        target: PolyCell3d,
        dofs: Iterable = None,
        penalty: float = DEFAULT_DIRICHLET_PENALTY,
        lazy: bool = True,
        tol: float = 1e-12,
        k: int = 4
    ):
        assert source.NDIM == 3, "Source must be a 3 dimensional body!"
        assert target.NDIM == 3, "Source must be a 3 dimensional body!"
        assert source.container.source() is target.container.source(), (
            "The source and the target must belong to the same pointcloud!"
        )
        assert source.container.source() is source.container.root(), (
            "The mesh must be brought to a standard form!"
        )
        assert isinstance(target.__class__.monomsfnc, Callable), (
            "The class is not equipped with the tools for this operation."
        )
        self.source = source
        self.target = target
        self.penalty = penalty
        self.lazy = lazy
        self.k = k
        self.tol = tol
        
        if isinstance(dofs, Iterable):
            self.dofmap = DOF.dofmap(dofs)
        else:
            self.dofmap = None
            
    def assemble(self, mesh: FemMesh) -> Tuple[coo, ndarray]:        
        S: PolyCell3d = self.source
        T: PolyCell3d = self.target
               
        coords = S.source_coords()
        
        inds_S = S.unique_indices()
        coords_S = coords[inds_S]
        
        tetra_T = T.to_tetrahedra(flatten=True)
        ecoords_T = points_of_cells(coords, tetra_T, centralize=False)

        if self.lazy:
            centers_T = cell_centers_bulk2(ecoords_T)
            k = min(self.k, len(centers_T))
            neighbours = k_nearest_neighbours(centers_T, coords_S, k=k)
            pips = _pip_tet_bulk_knn_(coords_S, ecoords_T, neighbours, self.tol)
        else:
            pips = _pip_tet_bulk_(coords_S, ecoords_T, self.tol)
        # pips : (nP, nE)
        
        pip = np.squeeze(np.any(pips, axis=1))  # (nP,)
        id_S = np.where(pip)[0]  # indices of coords_S that are on the surface of T
        pips = pips[id_S]
        if self.lazy:
            points_to_cells = _find_first_hits_knn_(pips, neighbours)
        else:
            points_to_cells = _find_first_hits_(pips)
        # points_to_cells : (nP,)
        
        tetmap_T = T.__class__.tetmap()
        i_T = np.floor(np.arange(len(tetra_T)) / len(tetmap_T)).astype(int)
        id_T = i_T[points_to_cells]  # indices of T that contain any points of S
        
        """monomsfnc = T.__class__.monomsfnc
        lc_T = T.lcoords()
        centers_T = T.centers()[id_T]
        ecoords_T = T.points_of_cells()[id_T]
        ecoords_T = _centralize_cells_coords_(ecoords_T)
        coords_S = coords_S[id_S]
        monoms_glob_T = monomsfnc(ecoords_T)
        monoms_glob_S = monomsfnc(coords_S - centers_T)
        loc_T = _glob_to_loc_bulk_2_c_(lc_T, monoms_glob_T, monoms_glob_S, centers_T)
        return loc_T"""


class NodeToNode(EssentialBoundaryCondition):
    """
    Constrains relative motion of nodes.

    Parameters
    ----------
    imap : Union[dict, ndarray, list]
        An iterable describing pairs of nodes.
    penalty : float, Optional
        Penalty value for Courant-type penalization.
        Default is `~sigmaepsilon.fem.constants.DEFAULT_DIRICHLET_PENALTY`.

    Example
    -------
    The following lines tie together all DOFs of nodes 1 with node 2 and node 3 with 4.
    The large penalty value means that the tied nodes should have the same displacements.

    >>> from sigmaepsilon.fem import NodeToNode
    >>> n2n = NodeToNode([[1, 2], [3, 4]], penalty=1e12)

    To tie only DOFs 'UX' and 'ROTZ':

    >>> n2n = NodeToNode([[1, 2], [3, 4]], dofs=['UX', 'ROTZ'], penalty=1e12)
    """

    def __init__(
        self,
        imap: Union[dict, ndarray, list] = None,
        *,
        source: PointCloud = None,
        target: PointCloud = None,
        dofs: Iterable = None,
        penalty: float = DEFAULT_DIRICHLET_PENALTY,
    ):
        if imap is None:
            if isinstance(source, PointCloud) and isinstance(target, PointCloud):
                imap = link_points_to_points(source, target)
        self.imap = imap
        self.penalty = penalty
        if isinstance(dofs, Iterable):
            self.dofmap = DOF.dofmap(dofs)
        else:
            self.dofmap = None

    def assemble(self, mesh: FemMesh) -> Tuple[coo, ndarray]:
        """
        Returns the penalty stiffness matrix and the penalty load matrix.
        """
        imap = None
        if isinstance(self.imap, dict):
            imap = np.stack([list(self.imap.keys()), list(self.imap.values())], axis=1)
        else:
            imap = np.array(self.imap).astype(int)
        nI = len(imap)
        nDOF = mesh.NDOFN
        nN = len(mesh.pointdata)
        N = nDOF * nN
        if self.dofmap is None:
            dmap = np.arange(nDOF)
        else:
            dmap = self.dofmap
        dmap = np.array(dmap, dtype=int)
        nF = nI * len(dmap)
        fdata = np.tile([1, -1], nF)
        frows = np.repeat(np.arange(nF), 2)
        i_source = conc([dmap + (i_ * nDOF) for i_ in imap[:, 0]])
        i_target = conc([dmap + (i_ * nDOF) for i_ in imap[:, 1]])
        fcols = np.stack([i_source, i_target], axis=1).flatten()
        factors = coo((fdata, (frows, fcols)), shape=(nF, N))
        Kp = self.penalty * (factors.T @ factors)
        fp = np.zeros(N, dtype=float)
        Kp.eliminate_zeros()
        return Kp, fp


class NodalSupport(EssentialBoundaryCondition):
    """
    A class to handle nodal supports.

    Parameters
    ----------
    data : dict, Optional
        Dictionary to define the prescribed values. Valid keys are
        'UX', 'UY', 'UZ', 'ROTX', 'ROTY', 'ROTZ'. Default is None.
    x : Iterable, Optional
        An iterable expressing the location of one or more points.
        Use it if you don't know the index of the point of application
        in advance. Default is None.
    i : Iterable[int] or int, Optional
        The index of one or more points of in a pointcloud.
        Default is None.
    frame : numpy.ndarray or ReferenceFrame, Optional
        Use it do define inclined supports. If it is a NumPy array,
        it must be a 3x3 matrix representing a set of orthonormal base
        vectors. A None value means that the prescribed values are understood
        in the frame of the mesh the constraints are applied to.
        Default is None.
    penalty : float, Optional
        Penalty value for Courant-type penalization.
        Default is `~sigmaepsilon.fem.constants.DEFAULT_DIRICHLET_PENALTY`.
    **kwargs : dict, Optional
        Keyword arguments used for elementwise definition of 'data'.

    Examples
    --------
    To define a support at one node

    >>> from sigmaepsilon.fem import NodalSupport
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

    The prescribed values can also be provided with a dictionary as the first
    positional argument:

    >>> support = NodalSupport({'UX' : 0., 'UY' : 0.}, x=[[0, 0, 0], [L, 0, 0]],
    >>>                         penalty=1e12)
    """

    def __init__(
        self,
        data: dict = None,
        *,
        x: Iterable = None,
        i: Union[int, Iterable[int]] = None,
        frame: Union[ndarray, ReferenceFrame] = None,
        penalty: float = DEFAULT_DIRICHLET_PENALTY,
        **kwargs,
    ):
        self.x = x if x is None else np.array(x, dtype=float)
        self.i = i if i is None else np.array(i, dtype=int)
        self.penalty = penalty
        self.data = data
        self.frame = frame
        if data is None:
            self.data = kwargs
        self.dofmap = DOF.dofmap(self.data.keys())
        if not isinstance(self.i, (int, Iterable)):
            assert self.x is not None, "Index or position must be defined!"
        if self.frame is not None:
            if not isinstance(self.frame, ReferenceFrame):
                raise TypeError(
                    "Invalid frame type. Read the " + "docs of the NodalSupport class."
                )
        else:
            self.frame = ReferenceFrame(dim=3)

    def assemble(self, mesh: FemMesh) -> Tuple[coo, ndarray]:
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
        assert r == 0, (
            "The number of deegrees of freedom per" + " node must be a multiple of 3."
        )
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
