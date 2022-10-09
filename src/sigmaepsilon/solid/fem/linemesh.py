# -*- coding: utf-8 -*-
from typing import Any, Iterable

import numpy as np

from sectionproperties.analysis.section import Section
from polymesh.config import __hasplotly__, __hasmatplotlib__
if __hasplotly__:
    from dewloosh.plotly import plot_lines_3d
    
from ..model.bernoulli.section import BeamSection
from .mesh import FemMesh
from .cells import B2, B3


class LineMesh(FemMesh):
    """
    A data class dedicated to 1d cells. It handles sections and other line 
    related information, plotting, etc.    
    """
    
    _cell_classes_ = {
        2: B2,
        3: B3,
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
            
    def simplify(self, inplace=True) -> 'LineMesh':
        pass
            
    @property
    def section(self) -> BeamSection:
        """
        Returns the section of the cells or None if there is no associated data.

        Returns
        -------
        :class:`BeamSection`
            The section instance associated with the beams of the block.
            
        """
        if self._section is not None:
            return self._section
        else:
            if self.is_root():
                return self._section
            else:
                return self.parent.section
                                  
    def plot(self, *args, scalars=None, backend='plotly', scalar_labels=None, **kwargs):
        """
        Plots the line elements using one of the supported backends.
        
        Parameters
        ----------
        scalars : :class:`numpy.ndarray`, Optional
            Data to plot. Default is None.
            
        backend : str, Optional
            The backend to use for plotting. Available options are 'plotly' and 'vtk'.
            Default is 'plotly'.
            
        scalar_labels : Iterable, Optional
            Labels of the scalars in 'scalars'. Only if Plotly is selected as the backend.
            Defaeult is None.
        
        Returns
        -------
        Any
            A PyVista or a Plotly object.
         
        """
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
        
    def plot_dof_solution(self, *args, backend:str='plotly', case:int=0, component:int=0, 
                          labels:Iterable=None, **kwargs) -> Any:
        """
        Plots degrees of freedom solution using 'vtk' or 'plotly'.
        
        Parameters
        ----------
        scalars : :class:`numpy.ndarray`, Optional
            Data to plot. Default is None.
            
        case : int, Optional
            The index of the load case. Default is 0.
            
        component : int, Optional
            The index of the DOF component. Default is 0.
            
        labels : Iterable, Optional
            Labels of the DOFs. Only if Plotly is selected as the backend.
            Defaeult is None.
            
        **kwargs : dict, Optional
            Keyqord arguments forwarded to :func:`pvplot`.
            
        Returns
        -------
        Any
            A figure object or None, depending on the selected backend.
            
        """
        if backend == 'vtk':
            scalars = self.nodal_dof_solution()[:, component, case]
            return self.pvplot(*args, scalars=scalars, **kwargs)
        elif backend == 'plotly':
            scalar_labels = labels if labels is not None else ['U', 'V', 'W']
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
        