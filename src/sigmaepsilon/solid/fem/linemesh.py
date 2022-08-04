# -*- coding: utf-8 -*-
import numpy as np

from sectionproperties.analysis.section import Section

from ...mesh.config import __hasplotly__, __hasmatplotlib__
from ...mesh.plotting.plotly import plot_lines_3d

from ..model.bernoulli.section import BeamSection

from .mesh import FemMesh
from .cells.bernoulli2 import Bernoulli2
from .cells.bernoulli3 import Bernoulli3


class LineMesh(FemMesh):
    """
    A data class dedicated to 1d cells. It handles sections and other line 
    related information, presets for plotting, etc.
    
    """
    
    _cell_classes_ = {
        2: Bernoulli2,
        3: Bernoulli3,
    }

    def __init__(self, *args, areas=None, connectivity=None, model=None, 
                 section=None, **kwargs):
        
        if connectivity is not None:
            if isinstance(connectivity, np.ndarray):
                assert len(connectivity.shape) == 3
                assert connectivity.shape[0] == nE
                assert connectivity.shape[1] == 2
                assert connectivity.shape[2] == self.__class__.NDOFN          
        
        if section is None:
            if isinstance(model, Section):
                section = BeamSection(wrap=model)
        if isinstance(section, BeamSection):
            section.calculate_geometric_properties()
            model = section.model_stiffness_matrix()
        self._section = section
                
        super().__init__(*args, connectivity=connectivity, model=model, **kwargs)
        
        if self.celldata is not None:
            nE = len(self.celldata)
            if areas is None:
                if isinstance(self.section, BeamSection):
                    areas = np.full(nE, self.section.A)
                else:
                    areas = np.ones(nE)
            else:
                assert len(areas.shape) == 1, \
                    "'areas' must be a 1d float or integer numpy array!"
            dbkey = self.celldata.__class__._attr_map_['areas']
            self.celldata.db[dbkey] = areas
            
    @property
    def section(self) -> BeamSection:
        if self._section is not None:
            return self._section
        else:
            if self.is_root():
                return self._section
            else:
                return self.parent.section
                                  
    def plot(self, *args, scalars=None, backend='plotly', scalar_labels=None, **kwargs):
        if backend == 'vtk':
            return self.pvplot(*args, scalars=scalars, scalar_labels=scalar_labels, 
                               **kwargs)
        elif backend == 'plotly':
            assert __hasplotly__
            coords = self.coords()
            topo = self.topology()
            return plot_lines_3d(coords, topo)
        elif backend == 'mpl':
            raise NotImplementedError
            assert __hasmatplotlib__
        else:
            msg = "No implementation for backend '{}'".format(backend)
            raise NotImplementedError(msg)
        
    def plot_dof_solution(self, *args, scalars=None, backend='plotly', case=0, **kwargs):
        if backend == 'vtk':
            return self.pvplot(*args, **kwargs)
        elif backend == 'plotly':
            scalar_labels = ['U', 'V', 'W']
            coords = self.coords()
            topo = self.topology()
            dofsol = self.nodal_dof_solution()[:, :3, case]
            return plot_lines_3d(coords, topo, scalars=dofsol, scalar_labels=scalar_labels)
        else:
            msg = "No implementation for backend '{}'".format(backend)
            raise NotImplementedError(msg)
    
    def _init_config_(self):
        super()._init_config_()
        key = self.__class__._pv_config_key_
        self.config[key]['color'] = 'k'
        self.config[key]['line_width'] = 10
        self.config[key]['render_lines_as_tubes'] = True 


class BernoulliFrame(LineMesh):
    
    NDOFN = 6
    