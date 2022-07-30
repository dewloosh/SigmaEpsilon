# -*- coding: utf-8 -*-
from copy import deepcopy
from ctypes import ArgumentError

import numpy as np

from randomdict import RandomDict

from .mesh.space import frames_of_lines
from .solid.fem.cells import B2, B3

from .solid.model.bernoulli.section import BeamSection, get_section


def structure_from_json(path):
    pass


def structure_from_dict(data):
    pass


def calculate_section_properties(d: dict):
    res = deepcopy(d)
    index = 0
    for sectionkey, section in d['sections'].items():
        s  = get_section(sectionkey, section)
        if s is not None:
            bs = BeamSection(wrap=s)
            props = bs.get_section_properties()
            props['obj'] = bs
            props['id'] = index
            props['cells'] = []
            res['sections'][sectionkey].update(**props)
        else:
            raise NotImplementedError("Section type not supported.")
        index += 1
    return res


def get_structure_metadata(data):
    
    d = calculate_section_properties(data)

    for i, materialkey in enumerate(d['materials'].keys()):
        d['materials'][materialkey]['id'] = i
        d['materials'][materialkey]['cells'] = []

    nN = len(d['nodes'])
    coords = np.zeros((nN, 3), dtype=float)
    for nodekey, node in d['nodes'].items():
        coords[int(nodekey)] = node

    cells = RandomDict(**data['cells'])
    # TODO : check if cells are nested or not, now assume they are not
    celltype = cells.pop('__type__', None)
    if celltype is None:
        raise ArgumentError("Missing type specification for cells!")
    assert isinstance(celltype, str)
    if celltype == 'beam':    
        nE = len(cells)
        nNE = len(cells.random_value()['nodes'])
        topo = np.zeros((nE, nNE), dtype=int)
        Hooke = np.zeros((nE, 4, 4), dtype=float)
        for cellkey, cell in cells.items():
            cellid = int(cellkey)
            m = d['materials'][cell['material']]
            s = d['sections'][cell['section']]
            d['cells'][cellkey]['material_id'] = m['id']
            d['cells'][cellkey]['section_id'] = s['id']
            d['sections'][cell['section']]['cells'].append(cellid)
            d['materials'][cell['material']]['cells'].append(cellid)
            E, nu = m['E'], m['nu']
            A, Ix, Iy, Iz = s['A'], s['Ix'], s['Iy'], s['Iz']
            G = E / (2 * (1 + nu))
            Hooke[cellid] = np.array([
                [E*A, 0, 0, 0],
                [0, G*Ix, 0, 0],
                [0, 0, E*Iy, 0],
                [0, 0, 0, E*Iz]
            ])
            topo[cellid] = cell['nodes']

        frames_auto = frames_of_lines(coords, topo)
        frames = np.zeros_like(frames_auto)
        for cellkey, cell in d['cells'].items():
            cellkey = int(cellkey)
            if 'frame' in cell:
                if isinstance(cell['frame'], str):
                    if cell['frame'] == 'auto':
                        frame = frames_auto[cellkey]
                else:
                    frame = np.array(
                        [
                            cell['frame']['i'],
                            cell['frame']['j'],
                            cell['frame']['k']
                        ]
                    )
                frames[cellkey] = frame
            else:
                frames[cellkey] = frames_auto[cellkey]

        if nNE == 2:
            celltype = B2
        elif nNE == 3:
            celltype = B3
        else:
            raise NotImplementedError

    nC = len(d['loads'])
    nodal_loads = np.zeros((nN, 6, nC), dtype=float)
    body_loads = np.zeros((nE, nNE, 6, nC), dtype=float)
    strain_loads = np.zeros((nE, 4, nC), dtype=float)
    strinds = list(map(lambda i: str(i), range(6)))
    for loadkey, load in d['loads'].items():
        loadkey = int(loadkey)
        if "nodal" in load:
            for nodekey, nodal_load in load["nodal"].items():
                nodekey = int(nodekey)
                nodal_loads[nodekey, :, loadkey] = nodal_load
        if "body" in load:
            for cellkey, body_load in load["body"].items():
                cellkey = int(cellkey)
                for iNE in range(nNE):
                    body_loads[cellkey, iNE, :, loadkey] = \
                        body_load[strinds[iNE]]
        if "heat" in load:
            for cellkey, heat_load in load["heat"].items():
                mkey = data['cells'][cellkey]['material']
                coeff = data['materials'][mkey]['thermal-coefficient']
                strain_loads[int(cellkey), 0, loadkey] = heat_load * coeff

    fixity = np.zeros((nN, 6)).astype(bool)
    for nodekey, fixvals in d['fixity'].items():
        fixity[int(nodekey)] = fixvals

    return d, dict(coords=coords, topo=topo, loads=nodal_loads, fixity=fixity,
                   celltype=celltype, model=Hooke, body_loads=body_loads,
                   strain_loads=strain_loads, frames=frames)