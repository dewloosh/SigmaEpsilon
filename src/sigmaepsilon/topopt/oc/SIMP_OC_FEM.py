# -*- coding: utf-8 -*-
import numpy as np
from collections import namedtuple
from copy import deepcopy

from sigmaepsilon.math.linalg.sparse.csr import csr_matrix as csr
from sigmaepsilon.mesh.utils import cells_around
from sigmaepsilon.solid.fem.structure import Structure

from .filter import sensitivity_filter, sensitivity_filter_csr
from .utils import (compliances_bulk, get_filter_factors,
                    get_filter_factors_csr,
                    weighted_stiffness_bulk as weighted_stiffness)

OptRes = namedtuple('OptimizationResult', 'x obj vol pen n')


def OC_SIMP_COMP(structure: Structure, *args,
                 miniter=50, maxiter=100, p_start=1.0, p_stop=4.0,
                 p_inc=0.2, p_step=5, q=0.5, vfrac=0.6, dtol=0.1,
                 r_max=None, summary=True, penalty=None, nostop=True,
                 neighbours=None, guess=None, i_start=0, **kwargs):

    do_filter = r_max is not None

    # if i_start==0:
    if structure.Solver is None:
        structure.preprocess()
    femsolver = structure.Solver.core
    assert femsolver.regular
    gnum = femsolver.gnum
    K_bulk_0 = np.copy(femsolver.K)
    vols = structure.volumes()
    centers = structure.centers()

    # initial solution to set up parameters
    dens = np.zeros_like(vols)
    dens_tmp = np.zeros_like(dens)
    dens_tmp_ = np.zeros_like(dens)
    dCdx = np.zeros_like(dens)
    comps = np.zeros_like(dens)

    def get_dof_solution():
        U = femsolver.u
        dU = len(U.shape)
        if dU == 2:
            # multiple load cases
            return U[:,  0]
        return U

    def compliance(update_stiffness=False):
        """
        Init == True means that we are in the initialization phase.
        """
        if update_stiffness:
            femsolver.update_stiffness(weighted_stiffness(K_bulk_0, dens))
        femsolver.proc()
        U = get_dof_solution()
        comps[:] = compliances_bulk(K_bulk_0, U, gnum)
        np.clip(comps, 1e-7, None, out=comps)
        return np.sum(comps)

    # ------------------ INITIAL SOLUTION ---------------
    comp = compliance()
    vol = np.sum(vols)
    vol_start = vol
    vol_min = vfrac * vol_start

    # initialite filter
    if do_filter:
        if neighbours is None:
            neighbours = cells_around(centers, r_max, frmt='dict')
        if isinstance(neighbours, csr):
            factors = get_filter_factors_csr(centers, neighbours, r_max)
            fltr = sensitivity_filter_csr
        else:
            factors = get_filter_factors(centers, neighbours, r_max)
            fltr = sensitivity_filter

    # initialize penalty parameters
    if penalty is not None:
        p_start = penalty
        p_stop = penalty + 1
        p_step = maxiter + 1

    # ------------- INITIAL FEASIBLE SOLUTION ------------
    if guess is None:
        dens[:] = vfrac
    else:
        dens = deepcopy(guess)
    vol = np.sum(dens * vols)
    comp = compliance(update_stiffness=True)
    yield OptRes(dens, comp, vol, p_start, -1)

    # ------------------- ITERATION -------------------
    p = p_start
    cIter = -1 + i_start
    dt = 0
    terminate = False
    while not terminate:
        if (p < p_stop) and (np.mod(cIter, p_step) == 0):
            p += p_inc
        cIter += 1

        # estimate lagrangian
        lagr = p * comp / vol

        # set up boundaries of change
        _dens = dens * (1 - dtol)
        np.clip(_dens, 1e-5, 1.0, out=_dens)
        dens_ = dens * (1 + dtol)
        np.clip(dens_, 1e-5, 1.0, out=dens_)

        # sensitivity [*(-1)]
        dCdx[:] = p * comps * dens ** (p-1)

        # sensitivity filter
        if do_filter:
            dCdx[:] = fltr(dCdx, dens, neighbours, factors)

        # calculate new densities and lagrangian
        dens_tmp_[:] = dens * (dCdx / vols) ** q
        dens_tmp[:] = dens_tmp_
        _lagr = 0
        lagr_ = 2 * lagr
        _maxtries = 200
        _ntries = 0
        while (lagr_ - _lagr) > 1e-3:
            if _ntries == _maxtries:
                raise RuntimeError("Couldn't find multiplier :(.")
            _lagr_ = (_lagr + lagr_) / 2
            dens_tmp[:] = dens_tmp_ / (_lagr_ ** q)
            np.clip(dens_tmp, _dens, dens_, out=dens_tmp)
            vol_tmp = np.sum(dens_tmp * vols)
            if vol_tmp < vol_min:
                lagr_ = _lagr_
            else:
                _lagr = _lagr_
            _ntries += 1
        lagr = lagr_
        dens[:] = dens_tmp

        # resolve equilibrium equations and calculate compliance
        comp = compliance(update_stiffness=True)
        dt += femsolver._summary['proc', 'time']
        vol = np.sum(dens * vols)
        res = OptRes(dens, comp, vol, p, cIter)
        yield res

        if nostop:
            terminate = False
        else:
            if cIter < miniter:
                terminate = False
            elif cIter >= maxiter:
                terminate = True
            else:
                terminate = (p >= p_stop)

    if summary:
        structure.summary['topopt'] = {
            'avg. time': dt / cIter,
            'niter': cIter
        }
    femsolver.K[:, :, :] = K_bulk_0
    # structure.postprocess()
    return OptRes(dens, comp, vol, p, cIter)
