from typing import Union

import numpy as np
from numpy import ndarray

from dewloosh.core import classproperty
from neumann import atleastnd, repeat
from neumann.logical import is1dfloatarray, isboolarray
from polymesh.celldata import CellData as MeshCellData


from ...core.material import MaterialLike


class CellData(MeshCellData):
    """
    A subclass of :class:`polymesh.celldata.CellData`to handle data related
    to the cells of a finite element mesh. This class does not need to be used
    directly, but rather serves as a base class for finite element classes.

    Parameters
    ----------
    activity : numpy.ndarray, Optional
        1d boolean array describing the activity of the elements.
    density : numpy.ndarray, Optional
        1d float array describing the density as mass per unit volume if areas
        (or thickness) are provided, mass per unit length (or area) othervise.
        See the Notes. Default is None.
    loads : numpy.ndarray, Optional
        3d (for a single load case) or 4d (for multiple load cases) float
        array of body loads for each load component of each node of each cell.
        Default is None.
    strain_loads : numpy.ndarray, Optional
        2d float array of body strain loads for each cell and strain
        component. Default is None.
    t or thickness : numpy.ndarray, Optional
        1d float array of thicknesses. Only for 2d cells.
        Default is None.
    fixity : numpy.ndarray, Optional
        3d boolean array of element fixity. Default is None.
    areas : numpy.ndarray, Optional
        1d float array of cross sectional areas. Only for 1d cells.
        Default is None.
    fields : dict, Optional
        Every value of this dictionary is added to the dataset.
        Default is `None`.
    **kwargs : dict, Optional
        For every key and value pair where the value is a numpy array
        with a matching shape (has entries for all points), the key
        is considered as a field and the value is added to the database.

    Notes
    -----
    For 1d and 2d cells, it is the user's responsibility to ensure that the
    input data makes sense. For example, 'areas' can be omitted, but then
    'density' must be 'mass per unit length'. The same way, 'thickness' can
    be omitted for 2d cells if 'density' is 'mass per unit area'. Otherwise
    'density' is expected as 'mass per unit volume'.

    See also
    --------
    :class:`~polymesh.celldata.CellData`
    :class:`~sigmaepsilon.fem.pointdata.PointData`

    Example
    -------
    All cell types are subclasses of :class:`CellData`

    >>> from sigmaepsilon.fem.cells import B2 as Beam
    >>> import numpy as np
    >>> random_topo = np.zeros((10, 2), dtype=int)  # random erroneous topology
    >>> random_data = np.random.rand(10, 3)
    >>> celldata = Beam(topo=random_topo, random_field = random_data)

    Then, the attached data is available as either

    >>> celldata['random_field']

    or

    >>> celldata.random_field
    """

    _attr_map_ = {
        "body-loads": "_body-loads_",
        "strain-loads": "_strain-loads_",
        "density": "_density_",
        "activity": "_activity_",
        "fixity": "_fixity_",
        "K": "_K_",  # elastic FEM stiffness matrix
        "C": "_C_",  # elastic material stiffness matrix
        "M": "_M_",  # mass matrix
        "B": "_B_",  # strain displacement matrix
        "f": "_f_",  # nodal load vector
    }

    def __init__(
        self,
        *args,
        material: Union[MaterialLike, ndarray] = None,
        activity: ndarray = None,
        density: ndarray = None,
        loads: ndarray = None,
        fields: dict = None,
        strain_loads: ndarray = None,
        t: ndarray = None,
        thickness: ndarray = None,
        fixity: ndarray = None,
        **kwargs
    ):
        amap = self.__class__._attr_map_

        t = t if thickness is None else thickness
        if fields is not None:
            activity = fields.pop(amap["activity"], activity)
            loads = fields.pop(amap["body-loads"], loads)
            strain_loads = fields.pop(amap["strain-loads"], strain_loads)
            density = fields.pop(amap["density"], density)
            t = fields.pop(amap["t"], t)

        super().__init__(*args, fields=fields, activity=activity, **kwargs)

        if self.db is not None:
            topo = self.topology()
            nE, nNE = topo.shape
            NDOFN = self.__class__.NDOFN

            if activity is None:
                activity = np.ones(nE, dtype=bool)
            else:
                assert (
                    isboolarray(activity) and len(activity.shape) == 1
                ), "'activity' must be a 1d boolean numpy array!"
            self.activity = activity

            # fixity
            if isinstance(fixity, np.ndarray):
                assert isboolarray(fixity), "Fixity must be a boolean array."
                self.fixity = fixity

            # densities
            if isinstance(density, np.ndarray):
                assert (
                    len(density.shape) == 1
                ), "'densities' must be a 1d float or integer numpy array!"
                self.density = density
            else:
                if density is not None and "density" not in fields:
                    raise TypeError(
                        "'density' must be a 1d float" + " or integer numpy array!"
                    )

            # body loads
            if loads is not None:
                assert isinstance(loads, np.ndarray)
                if loads.shape[0] == nE and loads.shape[1] == nNE:
                    self.loads = loads
                elif loads.shape[0] == nE and loads.shape[1] == nNE * NDOFN:
                    loads = atleastnd(loads, 3, back=True)
                    self.loads = loads.reshape(nE, nNE, NDOFN, loads.shape[-1])

            # strain loads
            if strain_loads is not None:
                assert isinstance(strain_loads, np.ndarray)
                assert strain_loads.shape[0] == nE
                self.strain_loads = strain_loads

            if self.__class__.NDIM == 2:
                if t is None:
                    t = np.ones(nE, dtype=float)
                else:
                    if isinstance(t, float):
                        t = np.full(nE, t)
                    else:
                        assert is1dfloatarray(
                            t
                        ), "'t' must be a 1d numpy array or a float!"
                self.db[self._dbkey_thickness_] = t

        self._material = None
        if isinstance(material, MaterialLike):
            self._material = material
            self.material_stiffness = material.elastic_stiffness_matrix()
        elif isinstance(material, ndarray):
            self.material_stiffness = material

    @classproperty
    def _dbkey_stiffness_matrix_(cls) -> str:
        return cls._attr_map_["K"]

    @classproperty
    def _dbkey_material_stiffness_matrix_(cls) -> str:
        return cls._attr_map_["C"]

    @classproperty
    def _dbkey_nodal_load_vector_(cls) -> str:
        return cls._attr_map_["f"]

    @classproperty
    def _dbkey_strain_displacement_matrix_(cls) -> str:
        return cls._attr_map_["B"]

    @classproperty
    def _dbkey_mass_matrix_(cls) -> str:
        return cls._attr_map_["M"]

    @classproperty
    def _dbkey_fixity_(cls) -> str:
        return cls._attr_map_["fixity"]

    @classproperty
    def _dbkey_density_(cls) -> str:
        return cls._attr_map_["density"]

    @classproperty
    def _dbkey_activity_(cls) -> str:
        return cls._attr_map_["activity"]

    @classproperty
    def _dbkey_body_loads_(cls) -> str:
        return cls._attr_map_["body-loads"]

    @classproperty
    def _dbkey_strain_loads_(cls) -> str:
        return cls._attr_map_["strain-loads"]

    # methods to check if a field is available

    @property
    def has_fixity(self):
        return self._dbkey_fixity_ in self._wrapped.fields

    @property
    def has_body_loads(self):
        return self._dbkey_body_loads_ in self._wrapped.fields

    @property
    def has_strain_loads(self):
        return self._dbkey_strain_loads_ in self._wrapped.fields

    @property
    def has_nodal_load_vector(self):
        return self._dbkey_nodal_load_vector_ in self._wrapped.fields

    @property
    def has_material_stiffness(self):
        return self._dbkey_material_stiffness_matrix_ in self._wrapped.fields

    # methods to access and set fields

    @property
    def nodes(self) -> ndarray:
        """Returns the topology of the cells."""
        return self._wrapped[self._dbkey_nodes_].to_numpy()

    @nodes.setter
    def nodes(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self._dbkey_nodes_] = value

    @property
    def strain_loads(self) -> ndarray:
        """Returns strain loads."""
        return self._wrapped[self._dbkey_strain_loads_].to_numpy()

    @strain_loads.setter
    def strain_loads(self, value: ndarray):
        if value is None:
            if self.has_strain_loads:
                del self._wrapped[self._dbkey_strain_loads_]
        else:
            assert isinstance(value, ndarray)
            self._wrapped[self._dbkey_strain_loads_] = value

    @property
    def loads(self) -> ndarray:
        """Returns body loads."""
        return self._wrapped[self._dbkey_body_loads_].to_numpy()

    @loads.setter
    def loads(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self._dbkey_body_loads_] = value

    @property
    def density(self) -> ndarray:
        """Returns densities."""
        return self._wrapped[self._dbkey_density_].to_numpy()

    @density.setter
    def density(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self._dbkey_density_] = value

    @property
    def activity(self) -> ndarray:
        """Returns activity of the cells."""
        return self._wrapped[self._dbkey_activity_].to_numpy()

    @activity.setter
    def activity(self, value: ndarray):
        """Sets the activity of the cells."""
        assert isinstance(value, ndarray)
        self._wrapped[self._dbkey_activity_] = value

    @property
    def fixity(self) -> ndarray:
        """Returns the fixity of the cells."""
        return self._wrapped[self._dbkey_fixity_].to_numpy()

    @fixity.setter
    def fixity(self, value: ndarray):
        """
        Sets the fixity of the cells.
        """
        assert isinstance(value, ndarray)
        self._wrapped[self._dbkey_fixity_] = value

    @property
    def material_stiffness(self) -> ndarray:
        """Returns the material stiffness matrices of the cells."""
        return self._wrapped[self._dbkey_material_stiffness_matrix_].to_numpy()

    @material_stiffness.setter
    def material_stiffness(self, value: ndarray):
        assert isinstance(value, ndarray)
        dbkey = self._dbkey_material_stiffness_matrix_
        if len(value.shape) == 2:
            self._wrapped[dbkey] = repeat(value, len(self))
        else:
            assert len(value.shape) == 3
            self.db[dbkey] = value
