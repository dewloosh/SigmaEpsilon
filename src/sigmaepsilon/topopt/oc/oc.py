import numpy as np
from numpy import ndarray
from copy import deepcopy
from typing import NamedTuple, Iterable

from neumann.linalg.sparse import csr_matrix as csr
from polymesh.utils import cells_around
from sigmaepsilon.fem.structure import Structure
from sigmaepsilon.fem.femsolver import StaticSolver

from .filter import sensitivity_filter, sensitivity_filter_csr
from .utils import (
    get_filter_factors,
    get_filter_factors_csr,
    weighted_stiffness_flat as weighted_stiffness,
    element_stiffness_ranges,
)
from ...utils.fem.postproc import element_compliances_flat as element_compliances


class OptRes(NamedTuple):
    x: Iterable  # the design variables
    obj: float  # the actual value of the objective function
    vol: float  # the actual volume
    pen: float  # the actual value of the penalty
    n: int  # the number of iterations completed


def maximize_stiffness(
    structure: Structure,
    *_,
    miniter: int = 50,
    maxiter: int = 100,
    p_start: float = 1.0,
    p_stop: float = 3.0,
    p_inc: float = 0.2,
    p_step: int = 5,
    q: float = 0.5,
    vfrac: float = 0.6,
    dtol: float = 0.1,
    r_max: float = None,
    penalty: float = None,
    nostop: bool = True,
    neighbours: Iterable = None,
    guess: Iterable = None,
    i_start: int = 0,
    **__
) -> OptRes:
    """
    Performs topology optimization using an Optimality Criteria Method to
    maximize the stiffness of a structure, given a design space and a certain 
    amount of material to distribute.
    
    .. math::
        :nowrap:
        
        \\begin{equation}
            \\begin{array}{rrclcl}
            \\displaystyle \\min_{\\boldsymbol{\\rho}} & \\mathbf{u(\\boldsymbol{\\rho})}^T \\mathbf{f} \\\\
            \\textrm{s. t.} \\\\
            & \\mathbf{K}(\\boldsymbol{\\rho}) \\mathbf{u(\\boldsymbol{\\rho})} & = & \\mathbf{f} \\\\
            & V(\\boldsymbol{\\rho}) - \\eta V_0 & \\leq & 0 & & \\\\
            & \\rho_i \\in \\{0, 1\\} & & & & \\forall i \\in N
            \\end{array}
        \\end{equation}
    
    Parameters
    ----------
    structure : Structure
        An instance of sigmaepsilon.fem.Structure.
    miniter : int, Optional
        The minimum number of iterations to perform. Default is 50.
    maxiter : int, Optional
        The maximum number of iterations to perform. Default is 100.
    p_start : float, Optional
        Initial value of the penalty on intermediate densities. Default is 1.
    p_stop : float, Optional
        Final value of the penalty on intermediate densities. Default is 3.
    p_inc : float, Optional
        Increment of the penalty on intermediate densities. Default is 0.2
    p_step : int, Optional
        The number of interations it takes to increment the penalty on intermediate
        density values. Default is 5.
    q : float, Optional
        Smoothing factor. Defaul is 0.5. 
    vfrac : float, Optional
        The fraction of available volume and the volume of the virgin structure.
        Default is 0.6. 
    dtol : float, Optional
        This controls the maximum change in the value of a design variable. 
        Default is 0.1. 
    r_max : float, Optional
        Radius for filtering. Default is None.
    neighbours : float, Optional
        The neighbours of the cells for filtering. Default is None.
    guess : numpy.ndarray, Optional
        A guess on the solution. This parameter can be used to contiue
        a workflow. Default is None.  
    i_start : int, Optional
        Starting index for iterations. This parameter can be used to contiue
        a workflow. Default is 0.
    summary : bool, Optional
        If True, a short summary about execution time and the number of iterations
        is available after execution as `structure.summary['topopt']`. Default is False.
    nostop : bool, Optional
        If True, iterations neglect all stopping criteria, those govern by 
        mniniter and maxiter included.
        Default is False.
    
    Yields
    ------
    OptRes
        The results of the actual iteration.
        
    Notes
    -----
    - The function returns a generator expression.
    - This function can be used for both size and topology optimization,
      depending on the inputs.
    """
    do_filter = r_max is not None

    if structure._static_solver_ is None:
        structure.linear_static_analysis()
    femsolver: StaticSolver = structure._static_solver_.core
    assert femsolver.regular
    krows, kcols = femsolver.krows, femsolver.kcols
    kshape = femsolver.kshape
    kranges = element_stiffness_ranges(kshape)
    K_virgin = np.copy(femsolver.K_bulk.flatten())
    mesh = structure.mesh
    vols = mesh.volumes()
    centers = mesh.centers()

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
            return U[:, 0]
        return U

    def compliance(update_stiffness: bool = False) -> ndarray:
        if update_stiffness:
            femsolver.update_stiffness(weighted_stiffness(K_virgin, dens, kranges))
        femsolver._proc_()
        U = get_dof_solution()
        comps[:] = element_compliances(K_virgin, U, krows, kcols, kranges)
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
            neighbours = cells_around(centers, r_max, frmt="dict")
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
    cIter = i_start
    p = p_start
    yield OptRes(dens, comp, vol, p, cIter)

    # ------------------- ITERATION -------------------
    terminate = False
    while not terminate:
        if (p < p_stop) and (np.mod(cIter, p_step) == 0):
            p += p_inc

        # estimate lagrangian
        lagr = p * comp / vol

        # set up boundaries of change
        _dens = dens * (1 - dtol)
        np.clip(_dens, 1e-5, 1.0, out=_dens)
        dens_ = dens * (1 + dtol)
        np.clip(dens_, 1e-5, 1.0, out=dens_)

        # sensitivity [*(-1)]
        dCdx[:] = p * comps * dens ** (p - 1)

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
            dens_tmp[:] = dens_tmp_ / (_lagr_**q)
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
        vol = np.sum(dens * vols)
        cIter += 1
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
                terminate = p >= p_stop

    femsolver.update_stiffness(K_virgin)
    return OptRes(dens, comp, vol, p, cIter)
