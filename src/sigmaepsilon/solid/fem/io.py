# -*- coding: utf-8 -*-
from copy import deepcopy
from functools import partial

import numpy as np
from randomdict import RandomDict

from linkeddeepdict import LinkedDeepDict
from polymesh.space import frames_of_lines

from ..model.bernoulli.section import BeamSection, get_section
from .cells import B2, B3


__all__ = ['_get_bernoulli_metadata_']


def _dofdict2list(d:dict, dofs : list, default=0.0) -> list:
    missing_keys = filter(lambda k : k not in d, dofs)
    d.update({k : default for k in missing_keys})
    return [d[k] for k in dofs]


def _get_nodal_dof_vector(*args, **kwargs):
    if isinstance(args[0], list):
        return args[0]
    if isinstance(args[0], dict):
        return _dofdict2list(*args, **kwargs)
    raise NotImplementedError
    
    
def _get_bernoulli_metadata_(data, *args, dofs=None, return_imap=False):
    
    if dofs is None:
        dofs = ('UX', 'UY', 'UZ', 'ROTX', 'ROTY', 'ROTZ')
    
    get_V6 = partial(_get_nodal_dof_vector, dofs=dofs)
    
    d = deepcopy(data)
    
    imap = LinkedDeepDict()
    # This is going to be filled up with inverse maps of indices
    # and keys. This allows for arbitrary keys in the input. 
    # You might need this to translate between input data and the output.
    
    # ---------------------------------------------------------------
    # --------------------------- POINTS ----------------------------
    # ---------------------------------------------------------------
    
    # allocate arrays for point-like data
    pkey = 'points' if 'points' in d else 'nodes'
    assert pkey in d
    # allocate data related to points
    # these objects are going to be filled up in the next for loop
    nP = len(d[pkey])
    coords = np.zeros((nP, 3), dtype=float)  # coordinates
    fixity = np.zeros((nP, 6), dtype=bool)  # essential bc
    mass = np.zeros((nP,), dtype=float)  # nodal masses
    loads = LinkedDeepDict()
    # collect data and create inverse mapping
    i_p = 0
    id_to_key = {}
    for key, p in d[pkey].items():
        id_to_key[i_p] = key
        if isinstance(p, list):
            # only coordinates
            coords[i_p] = p
            d[pkey] = {
                "x": p,
                "id" : i_p,
            }
        elif isinstance(p, dict):
            # several data
            d[pkey]['id'] = i_p            
            coords[i_p] = p['x']
            if 'mass' in p:
                mass[i_p] = p['mass']
            if 'load' in p:
                for case, value in p['load'].items():
                    loads[case, 'conc', i_p] = get_V6(value, default=0.0)
            if 'fixity' in p:                
                fixity[i_p] = get_V6(p['fixity'], default=False)
        i_p += 1
    imap['points'] = deepcopy(id_to_key)
    id_to_key = {}
        
    # ---------------------------------------------------------------
    # --------------- FRAMES, MATERIALS AND SECTIONS ----------------
    # ---------------------------------------------------------------
    
    # get materials and add some data to the records
    if 'frames' in d:
        i_f = 0
        id_to_key = {}
        for framekey in d['frames'].keys():
            id_to_key[i_f] = framekey
            i_f += 1
        imap['frames'] = deepcopy(id_to_key)
        id_to_key = {}
    
    # get materials and add some data to the records
    if 'materials' in d:
        i_m = 0
        id_to_key = {}
        for materialkey in d['materials'].keys():
            id_to_key[i_m] = materialkey
            d['materials'][materialkey]['cells'] = []  # we fill this up later
            i_m += 1
        imap['materials'] = deepcopy(id_to_key)
        id_to_key = {}
    
    # get sections and add some data to the records
    if 'sections' in d:
        i_s = 0
        id_to_key = {}
        for sectionkey, section in d['sections'].items():
            s = get_section(section['type'], **section)
            if s is not None:
                bs = BeamSection(wrap=s)
                bs.calculate_section_properties()
                props = bs.get_section_properties()
                props['obj'] = bs  # store section object
                props['id'] = i_s  # add an index
                props['cells'] = []  # we fill this up later
                d['sections'][sectionkey].update(**props)
            else:
                raise NotImplementedError("Section type not supported.")
            id_to_key[i_s] = sectionkey
            i_s += 1
        imap['sections'] = deepcopy(id_to_key)
        id_to_key = {}
    
    # ---------------------------------------------------------------
    # ---------------------------- CELLS ----------------------------
    # ---------------------------------------------------------------
    
    # FIXME This assumes a reguar input at the moment
    
    cells = RandomDict(**d['cells'])
    nE = len(cells)
    random_cell = cells.random_value()
    tkey = 'nodes' if 'nodes' in random_cell else 'topology'
    nNE = len(random_cell[tkey])
    
    if nNE == 2:
        celltype = B2
    elif nNE == 3:
        celltype = B3
    else:
        raise NotImplementedError
    
    topo = np.zeros((nE, nNE), dtype=int)  # topology
    conn = np.zeros((nE, nNE, 6), dtype=int)  # connectivity
    Hooke = np.zeros((nE, 4, 4), dtype=float)  # material model
    frames = np.zeros((nE, 3, 3), dtype=float)  # coordinate frames
    i_c = 0
    id_to_key = {}
    for cellkey, cell in cells.items():
        id_to_key[i_c] = key
        # topology
        topo[i_c] = cell[tkey]
        # connectivity
        if 'connectivity' in d['cells'][cellkey]:
            conn[i_c] = d['cells'][cellkey]['connectivity']
        # material
        if 'material' in cell:
            mid = cell['material']
        else:
            # material is defined for the section
            assert 'section' in cell, "A section must be defined"
            assert 'material' in cell['section'], "A material must be defined for the section." 
            mid = d['sections'][cell['section']]['material']
            m = d['materials'][mid]
        try:
            m = d['materials'][mid]
        except KeyError:
            if 'materials' in d:
                raise KeyError("Material id {} npt found.".format(mid))
            else:
                raise KeyError("Materials are not defined.")
        # section
        if 'section' in cell:
            s = d['sections'][cell['section']]
            d['sections'][cell['section']]['cells'].append(i_c)
            d['materials'][mid]['cells'].append(i_c)
        # build Hook's model
        E, nu = m['E'], m['nu']
        A, Ix, Iy, Iz = s['A'], s['Ix'], s['Iy'], s['Iz']
        G = E / (2 * (1 + nu))
        Hooke[i_c] = np.array([
            [E*A, 0, 0, 0],
            [0, G*Ix, 0, 0],
            [0, 0, E*Iy, 0],
            [0, 0, 0, E*Iz]
        ])
        
        # coordinate frame
        if 'frame' in cell:
            frames[i_c] = d['frames'][cell['frame']]
        
        # body loads
        if 'load' in cell:
            # nDOF values per element
            for case, value in cell['load'].items():
                loads[case, 'body', i_c] = get_V6(value, default=0.0)
        
        # temperature load
        if 'heat' in cell:
            # 1 value per element
            for case, value in cell['heat'].items():
                loads[case, 'heat', i_c] = value
            
        # register indices
        d['cells'][cellkey]['id'] = i_c
        d['cells'][cellkey]['material_id'] = m['id']
        d['cells'][cellkey]['section_id'] = s['id']
        i_c += 1
    imap['cells'] = deepcopy(id_to_key)
    id_to_key = {}
    # create coordinate frames
    # topology must be known for this implementation
    # FIXME This could be integrated into the block above, by
    # creating a single evaluation version of 'frames_of_lines'
    # -> 'frame_of_line'
    frames_auto = frames_of_lines(coords, topo)
    frames = np.zeros_like(frames_auto)
    i_c = 0
    for cellkey, cell in d['cells'].items():
        if 'frame' in cell:
            if isinstance(cell['frame'], str):
                if cell['frame'] == 'auto':
                    frame = frames_auto[i_c]
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
            frames[i_c] = frame
        else:
            frames[i_c] = frames_auto[i_c]
        i_c += 1

    # ---------------------------------------------------------------
    # ---------------------------- LOADS ----------------------------
    # ---------------------------------------------------------------
    # assembly of the load vectors
    nLC = len(loads)
    
    # vector of nodal loads
    nodal_loads = np.zeros((nP, 6, nLC))  
    
    # body load vector - for every dof for every node for element
    body_loads = np.zeros((nE, nNE, 6, nLC), dtype=float)  # vector of 
    
    # strain load vector - for every strain component for every element
    strain_loads = np.zeros((nE, 4, nLC), dtype=float)    
    i_lc = 0
    cid_to_ckey = imap['cells']
    for loadkey, lc in loads.items():
        if 'conc' in lc:
            for i, values in lc['conc'].items():
                nodal_loads[i, :, i_lc] = values
        if "body" in lc:
            for i, values in lc['body'].items():
                for iNE in range(nNE):
                    body_loads[i, iNE, :, i_lc] = values
        if "heat" in lc:
            for cid, heat in lc['heat'].items():
                ckey = cid_to_ckey[cid]
                mkey = data['cells'][ckey]['material']
                coeff = data['materials'][mkey]['thermal-coefficient']
                strain_loads[int(cellkey), 0, i_lc] = heat * coeff
        d['loads'][loadkey]['id'] = i_lc
        i_lc += 1

    return d, dict(coords=coords, topo=topo, loads=nodal_loads, fixity=fixity,
                   celltype=celltype, model=Hooke, body_loads=body_loads,
                   strain_loads=strain_loads, frames=frames, mass=mass)