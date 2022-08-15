from typing import Callable
from time import time

from sigmaepsilon.solid.fem.utils import irows_icols_bulk
from sigmaepsilon.solid.fem.mesh import FemMesh
from sigmaepsilon.solid.fem.imap import (index_mappers, box_spmatrix,
                                         box_rhs, box_dof_numbering)
from neumann.array import matrixform
from neumann.linalg.sparse.csr import csr_matrix as csr

from scipy.sparse import coo_matrix as npcoo, csc_matrix as npcsc
from scipy.sparse.linalg import spsolve
import numpy as np


from .utils import compliances_bulk, cells_around, \
    get_filter_factors, get_filter_factors_csr, weighted_stiffness_bulk, \
    filter_stiffness
from .filter import sensitivity_filter, sensitivity_filter_csr


def OC_SIMP_COMP(*args, sparsify=False, **kwargs):
    if (len(args) > 0) and isinstance(args[0], FemMesh):
        mesh = args[0]
        centers = mesh.centers()
        vols = mesh.areas()
        K_bulk = mesh.stiffness_matrix(sparse=False)
        gnum = mesh.element_dof_numbering()

        # mapping dofs
        loc_to_glob, glob_to_loc = index_mappers(gnum, return_inverse=True)
        gnum = box_dof_numbering(gnum, glob_to_loc)

        # essential boundary conditions
        Kp_coo = mesh.penalty_matrix_coo()
        Kp_coo = box_spmatrix(Kp_coo, glob_to_loc)

        # natural boundary conditions
        F = box_rhs(matrixform(mesh.load_vector()), loc_to_glob)
        if sparsify:
            F = npcsc(F)

        return _OC_SIMP_COMP_VOL_SENS_(K_bulk, Kp_coo, F, gnum, vols, centers, **kwargs)
    else:
        return _OC_SIMP_COMP_VOL_SENS_(*args, **kwargs)


def _OC_SIMP_COMP_VOL_SENS_(
        K_bulk: np.ndarray, Kp_coo: npcoo, RHS: np.ndarray,
        gnum: np.ndarray, vols: np.ndarray, centers: np.ndarray, *args,
        miniter=10, maxiter=100, p_start=1.0, p_stop=4.0,
        p_inc=0.2, p_step=5, q=0.5, vfrac=0.6, dtol=0.1,
        r_min=None, fltr_stiff=False, summary=True,
        callback: Callable = None, **kwargs):
    do_filter = r_min is not None

    # initial solution to set up parameters
    krows, kcols = irows_icols_bulk(gnum)
    krows = krows.flatten()
    kcols = kcols.flatten()
    nTOTV = gnum.max() + 1
    full_shape = (nTOTV, nTOTV)
    #
    dens = np.zeros_like(vols)
    dens_tmp = np.zeros_like(vols)
    dens_tmp_ = np.zeros_like(vols)
    dCdx = np.zeros_like(vols)

    # ------------------ INITIAL SOLUTION ---------------

    Kr_coo = npcoo((K_bulk.flatten(), (krows, kcols)),
                   shape=full_shape, dtype=K_bulk.dtype)
    K_coo = Kr_coo + Kp_coo
    U = spsolve(K_coo, RHS, permc_spec='NATURAL', use_umfpack=True)
    comps = compliances_bulk(K_bulk, U, gnum)
    np.clip(comps, 1e-7, None, out=comps)
    comp = np.sum(comps)
    vol = np.sum(vols)

    # initialization
    vol_start = vol
    vol_min = vfrac * vol_start
    if do_filter:
        neighbours = cells_around(centers, r_min, as_csr=False)
        if isinstance(neighbours, csr):
            factors = get_filter_factors_csr(centers, neighbours, r_min)
            fltr = sensitivity_filter_csr
        else:
            factors = get_filter_factors(centers, neighbours, r_min)
            fltr = sensitivity_filter

    # ------------- INITIAL FEASIBLE SOLUTION ------------

    dens[:] = vfrac
    K_bulk_w = weighted_stiffness_bulk(K_bulk, dens)
    Kr_coo = npcoo((K_bulk_w.flatten(), (krows, kcols)),
                   shape=full_shape, dtype=K_bulk.dtype)
    K_coo = Kr_coo + Kp_coo
    U = spsolve(K_coo, RHS, permc_spec='NATURAL', use_umfpack=True)
    comps[:] = compliances_bulk(K_bulk, U, gnum)
    np.clip(comps, 1e-7, None, out=comps)
    comp = np.sum(comps)
    if callback is not None:
        callback(0, comp, vol, dens)

    # ------------------- ITERATION -------------------

    p = p_start
    cIter = -1
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
        while (lagr_ - _lagr) > 1e-3:
            _lagr_ = (_lagr + lagr_) / 2
            dens_tmp[:] = dens_tmp_ / (_lagr_ ** q)
            np.clip(dens_tmp, _dens, dens_, out=dens_tmp)
            vol_tmp = np.sum(dens_tmp * vols)
            if vol_tmp < vol_min:
                lagr_ = _lagr_
            else:
                _lagr = _lagr_
        lagr = lagr_
        dens[:] = dens_tmp

        # resolve equilibrium equations
        K_bulk_w = weighted_stiffness_bulk(K_bulk, dens)
        if fltr_stiff:
            # TODO : SELECT WORST <X> PERCENT
            Kr_coo = npcoo(filter_stiffness(K_bulk_w, gnum, dens, tol=0.001),
                           shape=full_shape, dtype=K_bulk.dtype)
        else:
            Kr_coo = npcoo((K_bulk_w.flatten(), (krows, kcols)),
                           shape=full_shape, dtype=K_bulk.dtype)
        K_coo = Kr_coo + Kp_coo
        t0 = time()
        U = spsolve(K_coo, RHS, permc_spec='NATURAL', use_umfpack=True)
        dt += time() - t0
        comps[:] = compliances_bulk(K_bulk, U, gnum)
        np.clip(comps, 1e-7, None, out=comps)
        comp = np.sum(comps)
        vol = np.sum(dens * vols)

        if callback is not None:
            callback(cIter, comp, vol, dens)

        if cIter < miniter:
            terminate = False
        elif cIter >= maxiter:
            terminate = True
        else:
            terminate = (p >= p_stop)

    if summary:
        dsum = {
            'avg. time [ms]': 1000 * dt / cIter,
            'niter': cIter,
            'filter_stiff': fltr_stiff
        }
        return dens, dsum
    return dens


if __name__ == '__main__':
    pass
