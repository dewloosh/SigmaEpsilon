# -*- coding: utf-8 -*-
import json
from copy import deepcopy

import numpy as np

from randomdict import RandomDict

from polymesh.space import frames_of_lines
from .solid.fem.cells import B2, B3

from .solid.model.bernoulli.section import BeamSection, get_section


def json2structure(path):
    pass


def dict2structure(data):
    pass


def json2dict(jsonpath: str):
    with open(jsonpath, "r") as jsonfile:
        return json.loads(jsonfile.read())


def dict2json(jsonpath: str, d: dict):
    with open(jsonpath, "w") as outfile:
        json.dump(d, outfile)


def get_structure_metadata(data):

    d = deepcopy(data)

    # get sections and add some data to the records
    index = 0
    for sectionkey, section in d['sections'].items():
        s = get_section(sectionkey, section)
        if s is not None:
            bs = BeamSection(wrap=s)
            props = bs.get_section_properties()
            props['obj'] = bs  # store section object
            props['id'] = index  # add an index
            props['cells'] = []  # we fill this up later
            d['sections'][sectionkey].update(**props)
        else:
            raise NotImplementedError("Section type not supported.")
        index += 1

    # get materials and add some data to the records
    for i, materialkey in enumerate(d['materials'].keys()):
        d['materials'][materialkey]['id'] = i  # add index
        d['materials'][materialkey]['cells'] = []  # we fill this up later

    # collect nodal data and create indices
    nN = len(d['nodes'])
    coords = np.zeros((nN, 3), dtype=float)
    mass = np.zeros((nN,), dtype=float)
    index = 0
    for nodekey, node in d['nodes'].items():
        if isinstance(node, list):
            coords[index] = node
            d['nodes'] = {
                "x": node,
                "id" : index,
                "mass" : 0.0
            }
        elif isinstance(node, dict):
            d['nodes'][nodekey]['id'] = index
            coords[index] = node['x']
            if 'mass' in node:
                mass[index] = node['mass']
        index += 1

    # collect cell data and create indices
    cells = RandomDict(**data['cells'])
    nE = len(cells)
    nNE = len(cells.random_value()['nodes'])
    topo = np.zeros((nE, nNE), dtype=int)  # topology
    conn = np.zeros((nE, nNE, 6), dtype=int)  # connectivity
    Hooke = np.zeros((nE, 4, 4), dtype=float)  # material model
    index = 0
    for cellkey, cell in cells.items():
        m = d['materials'][cell['material']]
        s = d['sections'][cell['section']]
        d['cells'][cellkey]['id'] = index
        d['cells'][cellkey]['material_id'] = m['id']
        d['cells'][cellkey]['section_id'] = s['id']
        d['sections'][cell['section']]['cells'].append(index)
        d['materials'][cell['material']]['cells'].append(index)
        E, nu = m['E'], m['nu']
        A, Ix, Iy, Iz = s['A'], s['Ix'], s['Iy'], s['Iz']
        G = E / (2 * (1 + nu))
        Hooke[index] = np.array([
            [E*A, 0, 0, 0],
            [0, G*Ix, 0, 0],
            [0, 0, E*Iy, 0],
            [0, 0, 0, E*Iz]
        ])
        topo[index] = cell['nodes']
        index += 1
        
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

    # natural boundary conditions (external load constrants)
    # and strain loads
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

    # essential boundary conditions (displacement constraints)
    fixity = np.zeros((nN, 6)).astype(bool)
    for nodekey, fixvals in d['fixity'].items():
        fixity[int(nodekey)] = fixvals

    return d, dict(coords=coords, topo=topo, loads=nodal_loads, fixity=fixity,
                   celltype=celltype, model=Hooke, body_loads=body_loads,
                   strain_loads=strain_loads, frames=frames, mass=mass)
