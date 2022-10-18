# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray

from neumann.array import atleastnd
from neumann.array import atleastnd, isboolarray, is1dfloatarray
from polymesh.celldata import CellData as MeshCellData


class CellData(MeshCellData): 
    """
    A subclass of :class:`polymesh.celldata.CellData`to handle data related to the cells 
    of a finite element mesh. This class does not need to be used directly, but rather serves 
    as a base class for finite element classes.
    
    Parameters
    ----------
    activity : :class:`numpy.ndarray`, Optional
        1d boolean array describing the activity of the elements.
        
    density : :class:`numpy.ndarray`, Optional
        1d float array describing the density as mass per unit volume if areas
        (or thickness) are provided, mass per unit length (or area) othervise. 
        See the Notes. Default is None.
        
    loads : :class:`numpy.ndarray`, Optional
        3d (for a single load case) or 4d (for multiple load cases) float array 
        of body loads for each load component of each node of each cell.
        Default is None.
        
    strain_loads : :class:`numpy.ndarray`, Optional
        2d float array of body strain loads for each cell and strain component.
        Default is None.
        
    t or thickness : :class:`numpy.ndarray`, Optional
        1d float array of thicknesses. Only for 2d cells.
        Default is None.
        
    fixity : :class:`numpy.ndarray`, Optional
        3d boolean array of element connectivity, same as 'connectivity'.
        Default is None.
    
    connectivity : :class:`numpy.ndarray`, Optional
        3d boolean array of element connectivity.
        Default is None.
        
    areas : :class:`numpy.ndarray`, Optional
        1d float array of cross sectional areas. Only for 1d cells.
        Default is None.
        
    fields : `dict`, Optional
        Every value of this dictionary is added to the dataset. Default is `None`.
        
    **kwargs : dict, Optional
        For every key and value pair where the value is a numpy array
        with a matching shape (has entries for all points), the key
        is considered as a field and the value is added to the database.
        
    Notes
    -----
    For 1d and 2d cells, it is the user's responsibility to ensure that the input data makes sense.
    For example, 'areas' can be omitted, but then 'density' must be 'mass per unit length'.
    The same way, 'thickness' can be omitted for 2d cells if 'density' is 'mass per unit area'.
    Otherwise 'density' is expected as 'mass per unit volume'.
    
    See also
    --------
    :class:`polymesh.celldata.CellData`
    :class:`sigmaepsilon.solid.fem.pointdata.PointData`
        
    Example
    -------
    All cell types are subclasses of :class:`CellData`
    
    >>> from sigmaepsilon.solid.fem.cells import B2 as Beam
    >>> import numpy as np
    >>> random_topo = np.zeros((10, 2), dtype=int)  # random erroneous topology
    >>> random_data = np.random.rand(10, 3)
    >>> celldata = Beam(topo=random_topo, random_field = random_data)
    
    Then, the attached data is available as either
    
    >>> celldata['random_data']
    
    or 
    
    >>> celldata.random_data
        
    """

    _attr_map_ = {
        'loads': 'loads',
        'strain-loads': 'strain-loads',
        'density': 'density',
        'activity' : 'activity',
        'connectivity' : 'connectivity',
        'areas' : 'areas',
    }

    def __init__(self, *args, model=None, activity=None, density=None,
                 loads=None, strain_loads=None, t=None, fields=None, 
                 thickness=None, connectivity=None, areas=None, 
                 tight=True, **kwargs):
        amap = self.__class__._attr_map_

        t = t if thickness is None else thickness
        if fields is not None:
            activity = fields.pop(amap['activity'], activity)
            loads = fields.pop(amap['loads'], loads)
            strain_loads = fields.pop(amap['strain-loads'], strain_loads)
            density = fields.pop(amap['density'], density)
            t = fields.pop(amap['t'], t)

        super().__init__(*args, fields=fields, activity=activity, **kwargs)

        if self.db is not None:
            topo = self.topology()
            nE, nNE = topo.shape
            NDOFN = self.__class__.NDOFN
            NSTRE = self.__class__.NSTRE

            if activity is None:
                activity = np.ones(nE, dtype=bool)
            else:
                assert isboolarray(activity) and len(activity.shape) == 1, \
                    "'activity' must be a 1d boolean numpy array!"
            self.activity = activity
            
            # connectivity
            if isinstance(connectivity, np.ndarray):
                self.connectivity = connectivity
                
            # areas
            if isinstance(areas, np.ndarray):
                self._wrapped[self.__class__._attr_map_['areas']] = areas
            
            # densities
            if isinstance(density, np.ndarray):
                assert len(density.shape) == 1, \
                    "'densities' must be a 1d float or integer numpy array!"
                self.density = density
            else:
                if density is not None and 'density' not in fields:
                    raise TypeError("'density' must be a 1d float" + \
                                    " or integer numpy array!")

            # body loads
            if loads is None and not tight:
                loads = np.zeros((nE, nNE, NDOFN, 1))
            if loads is not None:
                assert isinstance(loads, np.ndarray)
                if loads.shape[0] == nE and loads.shape[1] == nNE:
                    self.loads = loads
                elif loads.shape[0] == nE and loads.shape[1] == nNE * NDOFN:
                    loads = atleastnd(loads, 3, back=True)
                    self.loads = loads.reshape(
                        nE, nNE, NDOFN, loads.shape[-1])

            # strain loads
            if strain_loads is None and not tight:
                strain_loads = np.zeros((nE, NSTRE, 1))
            if strain_loads is not None:
                assert isinstance(strain_loads, np.ndarray)
                assert strain_loads.shape[0] == nE
                self.db[amap['strain-loads']] = strain_loads

            if self.__class__.NDIM == 2:
                if t is None:
                    t = np.ones(nE, dtype=float)
                else:
                    if isinstance(t, float):
                        t = np.full(nE, t)
                    else:
                        assert is1dfloatarray(t), \
                            "'t' must be a 1d numpy array or a float!"
                self.db[amap['t']] = t

        self._model = model
        
    @property
    def nodes(self) -> ndarray:
        """Returns the topology of the cells."""
        return self._wrapped[self.__class__._attr_map_['nodes']].to_numpy()

    @nodes.setter
    def nodes(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['nodes']] = value
    
    @property
    def strain_loads(self) -> ndarray:
        """Returns strain loads."""
        return self._wrapped[self.__class__._attr_map_['strain-loads']].to_numpy()

    @strain_loads.setter
    def strain_loads(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['strain-loads']] = value    
    
    @property
    def loads(self) -> ndarray:
        """Returns body loads."""
        return self._wrapped[self.__class__._attr_map_['loads']].to_numpy()

    @loads.setter
    def loads(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['loads']] = value
                
    @property
    def density(self) -> ndarray:
        """Returns densities."""
        return self._wrapped[self.__class__._attr_map_['density']].to_numpy()

    @density.setter
    def density(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['density']] = value
        
    @property
    def activity(self) -> ndarray:
        """Returns activity of the cells."""
        key = self.__class__._attr_map_['activity']
        if key in self._wrapped.fields:
            return self._wrapped[key].to_numpy()
        else:
            return None

    @activity.setter
    def activity(self, value: ndarray):
        """Sets the activity of the cells."""
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['activity']] = value
    
    @property
    def connectivity(self) -> ndarray:
        """Returns the connectivity of the cells."""
        key = self.__class__._attr_map_['connectivity']
        if key in self._wrapped.fields:
            return self._wrapped[key].to_numpy()
        else:
            return None

    @connectivity.setter
    def connectivity(self, value: ndarray):
        """Sets the connectivity of the cells."""
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['connectivity']] = value
