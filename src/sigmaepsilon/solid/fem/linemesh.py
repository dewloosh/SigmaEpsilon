# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Any, Iterable, Tuple

import numpy as np
from randomdict import RandomDict

from sectionproperties.analysis.section import Section
from polymesh.space import StandardFrame, frames_of_lines
from polymesh.config import __hasplotly__, __hasmatplotlib__
if __hasplotly__:
    from dewloosh.plotly import plot_lines_3d
    
from ..model.bernoulli.section import BeamSection, get_section
from .mesh import FemMesh
from .pointdata import PointData
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

    @classmethod
    def from_dict(cls, d_in:dict) -> Tuple[dict, 'LineMesh']:
        """
        Reads a mesh form a dictionary. Returns a decorated version
        of the input dictionary and a `LineMesh` instance.
        
        .. highlight:: python
        .. code-block:: python

            {
                "_version_": "0.0.1",
    
                "materials": {
                    "Material": {
                        "E": 210000000.0,
                        "nu": 0.3,
                        "fy": 235000.0,
                        "type": "steel",
                        "density": 7850.0,
                        "thermal-coefficient": 1.2e-05
                    }
                },
                
                "sections": {
                    "Section": {
                        "type": "CHS",
                        "d": 1.0,
                        "t": 0.1
                    }
                },
                
                "frames": {
                    "0": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]
                    ]
                },
                
                "fixity": {
                    "0": [True, True, True, True, True, True]
                },
                
                "loads": {
                    "LC1": {
                        "nodal": {
                            "3": [0.0, 0.0, -100.0, 0.0, 0.0, 0.0]
                        }
                    }
                },
                
                "nodes": {
                    "0": [0.0, 0.0, 0.0],
                    "1": {
                        "x": [3.0, 0.0, 0.0],
                        "fixity": [False, False, False, False, False, False],
                        "load": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    },
                    "2": {
                        "x": [7.0, 0.0, 0.0],
                    },
                    "3": {
                        "x": [10.0, 0.0, 0.0],
                        "fixity": [True, True, True, True, True, True],
                        "mass": 1.0
                    }
                },
                
                "cells": {
                    "0": {
                        "nodes": [0, 1],
                        "material": "Material",
                        "section": "Section",
                        "density": 0.1,
                        "frame": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]
                        ]
                    },
                    "1": {
                        "nodes": [1, 2],
                        "material": "Material",
                        "section": "Section",
                        "density": 0.1,
                        "frame": "auto"
                    },
                    "2": {
                        "nodes": [2, 3],
                        "material": {
                            "E": 210000000.0,
                            "nu": 0.3,
                            "fy": 235000.0,
                            "type": "steel",
                            "density": 7850.0,
                            "thermal-coefficient": 1.2e-05
                        },
                        "section": {
                            "type": "CHS",
                            "d": 1.0,
                            "t": 0.1
                        },
                        "density": 0.1,
                        "frame": "auto"
                    }
                },
                
                "data": {
                    "nodes": {},
                    "cells": {}
                },
                
                "options": {
                    "stat": {
                        "solver": "scipy"
                    },
                    "dyn": {
                        "n_mode": "all"
                    }
                }
            }
        
        """
        d_out, data = _get_metadata_(d_in)
        
        # space
        GlobalFrame = StandardFrame(dim=3)
        
        # pointdata
        pd = PointData(coords=data['coords'], frame=GlobalFrame,
                    loads=data['loads'], fixity=data['fixity'])
        # celldata
        topo=data['topo']
        if isinstance(topo, np.ndarray):
            if topo.shape[-1] == 2:
                cd = B2(topo=topo, frames=data['frames'])
            elif topo.shape[-1] == 2:
                cd = B3(topo=topo, frames=data['frames'])
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        # set up mesh
        mesh = LineMesh(pd, cd, model=data['model'], frame=GlobalFrame)
        
        # return decorated input dictionary and the mesh object
        return d_out, mesh


class BernoulliFrame(LineMesh):
    
    NDOFN = 6
    

def _get_metadata_(data):
    
    dofs =('UX', 'UY', 'UZ', 'ROTX', 'ROTY', 'ROTZ')
    dofmap = {key : i for i, key in enumerate(dofs)}

    d = deepcopy(data)

    # ------------------------------ SECTIONS -------------------------------
    # get sections and add some data to the records
    index_sec = 0
    for sectionkey, section in d['sections'].items():
        s = get_section(section['type'], **section)
        if s is not None:
            bs = BeamSection(wrap=s)
            bs.calculate_section_properties()
            props = bs.get_section_properties()
            props['obj'] = bs  # store section object
            props['id'] = index_sec  # add an index
            props['cells'] = []  # we fill this up later
            d['sections'][sectionkey].update(**props)
        else:
            raise NotImplementedError("Section type not supported.")
        index_sec += 1

    # ------------------------------ MATERIALS -------------------------------
    # get materials and add some data to the records
    index_mat = 0
    for materialkey in d['materials'].keys():
        d['materials'][materialkey]['id'] = index_mat  # add index
        d['materials'][materialkey]['cells'] = []  # we fill this up later
        index_mat += 1
        
    # collect nodal data and create indices
    nN = len(d['nodes'])
    coords = np.zeros((nN, 3), dtype=float)
    mass = np.zeros((nN,), dtype=float)
    index_node = 0
    for nodekey, node in d['nodes'].items():
        if isinstance(node, list):
            coords[index_node] = node
            d['nodes'] = {
                "x": node,
                "id" : index_node,
                "mass" : 0.0
            }
        elif isinstance(node, dict):
            d['nodes']['id'] = index_node            
            coords[index_node] = node['x']
            if 'mass' in node:
                mass[index_node] = node['mass']
        index_node += 1

    # ------------------------------ CELLS -------------------------------
    cells = RandomDict(**d['cells'])
    nE = len(cells)
    nNE = len(cells.random_value()['nodes'])
    topo = np.zeros((nE, nNE), dtype=int)  # topology
    conn = np.zeros((nE, nNE, 6), dtype=int)  # connectivity
    Hooke = np.zeros((nE, 4, 4), dtype=float)  # material model
    index_cell = 0
    for cellkey, cell in cells.items():
        # material
        if isinstance(cell['material'], dict):
            m = cell['material']
            m['id'] = index_mat
            index_mat += 1
        elif isinstance(cell['material'], str):
            m = d['materials'][cell['material']]
        # section
        if isinstance(cell['section'], dict):
            s = cell['section']
            section = get_section(s['type'], **s)
            if section is not None:
                bs = BeamSection(wrap=section)
                bs.calculate_section_properties()
                props = bs.get_section_properties()
                props['obj'] = bs  # store section object
                props['id'] = index_sec  # add an index
                props['cells'] = []  # we fill this up later
                cell['section'].update(**props)
                cell['section']['id'] = index_sec
                index_sec += 1
            else:
                raise NotImplementedError("Section type not supported.")
        elif isinstance(cell['section'], str):
            s = d['sections'][cell['section']]
            d['sections'][cell['section']]['cells'].append(index_cell)
            d['materials'][cell['material']]['cells'].append(index_cell)
        # register indices
        d['cells'][cellkey]['id'] = index_cell
        d['cells'][cellkey]['material_id'] = m['id']
        d['cells'][cellkey]['section_id'] = s['id']
        # build Hook's model
        E, nu = m['E'], m['nu']
        A, Ix, Iy, Iz = s['A'], s['Ix'], s['Iy'], s['Iz']
        G = E / (2 * (1 + nu))
        Hooke[index_cell] = np.array([
            [E*A, 0, 0, 0],
            [0, G*Ix, 0, 0],
            [0, 0, E*Iy, 0],
            [0, 0, 0, E*Iz]
        ])
        topo[index_cell] = cell['nodes']
        # connectivity
        if 'connectivity' in d['cells'][cellkey]:
            conn[index_cell] = d['cells'][cellkey]['connectivity']
        index_cell += 1
    
    # ------------------------------ FRAMES -------------------------------
    # create coordinate frames
    frames_auto = frames_of_lines(coords, topo)
    frames = np.zeros_like(frames_auto)
    index = 0
    for cellkey, cell in d['cells'].items():
        if 'frame' in cell:
            if isinstance(cell['frame'], str):
                if cell['frame'] == 'auto':
                    frame = frames_auto[index]
            elif isinstance(cell['frame'], dict):
                frame = np.array(
                    [
                        cell['frame']['i'],
                        cell['frame']['j'],
                        cell['frame']['k']
                    ]
                )
            elif isinstance(cell['frame'], list):
                frame = np.array(cell['frame'])
            else:
                raise NotImplementedError
            frames[index] = frame
        else:
            frames[index] = frames_auto[index]
        index += 1

    # set finite element type
    if nNE == 2:
        celltype = B2
    elif nNE == 3:
        celltype = B3
    else:
        raise NotImplementedError

    # ------------------------------ LOADS -------------------------------
    # natural boundary conditions (external load constrants)
    # and strain loads
    nC = len(d['loads'])
    nodal_loads = np.zeros((nN, 6, nC), dtype=float)
    body_loads = np.zeros((nE, nNE, 6, nC), dtype=float)
    strain_loads = np.zeros((nE, 4, nC), dtype=float)
    strinds = list(map(lambda i: str(i), range(6)))
    index = 0
    for loadkey, load in d['loads'].items():
        d['loads'][loadkey]['id'] = index
        if "nodal" in load:
            for nodekey, nodal_load in load["nodal"].items():
                nodekey = int(nodekey)
                nodal_loads[nodekey, :, index] = nodal_load
        if "body" in load:
            for cellkey, body_load in load["body"].items():
                cellkey = int(cellkey)
                for iNE in range(nNE):
                    body_loads[cellkey, iNE, :, index] = \
                        body_load[strinds[iNE]]
        if "heat" in load:
            for cellkey, heat_load in load["heat"].items():
                mkey = data['cells'][cellkey]['material']
                coeff = data['materials'][mkey]['thermal-coefficient']
                strain_loads[int(cellkey), 0, index] = heat_load * coeff
        index += 1

    #----------------------------- SUPPORTS -------------------------------
    # essential boundary conditions (displacement constraints)
    fixity = np.zeros((nN, 6)).astype(bool)
    for nodekey, fixvals in d['fixity'].items():
        fixity[int(nodekey)] = fixvals

    return d, dict(coords=coords, topo=topo, loads=nodal_loads, fixity=fixity,
                   celltype=celltype, model=Hooke, body_loads=body_loads,
                   strain_loads=strain_loads, frames=frames, mass=mass)