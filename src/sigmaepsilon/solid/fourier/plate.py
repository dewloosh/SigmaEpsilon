# -*- coding: utf-8 -*-
import numpy as np
from numpy import swapaxes as swap
from time import time

from .loads import LoadGroup
from .preproc import lhs_Navier
from .postproc import postproc
from .proc import linsolve


class RectangularPlate:

    def __init__(self, size: tuple, shape: tuple, *args,
                 D: np.ndarray = None, S: np.ndarray = None,
                 Winkler=None, Pasternak=None, loads=None,
                 Hetenyi=None, model='mindlin', mesh=None,
                 Bernoulli: bool = None, Navier=True, **kwargs):
        assert Navier is True, "Only Navier boundary conditions are covered at the moment."
        self.size = np.array(size, dtype=float)
        self.shape = np.array(shape, dtype=int)
        self.D = D
        self.S = S
        self.model = model.lower()
        self.mesh = mesh
        if isinstance(loads, LoadGroup):
            self.loads = loads
        elif isinstance(loads, dict):
            self.add_loads_from_dict(loads)
        elif isinstance(loads, np.ndarray):
            raise NotImplementedError
        else:
            self.loads = LoadGroup(Navier=self)
        self._summary = {}
        self.Bernoulli = not self.model in [
            'm', 'mindlin'] if Bernoulli is None else Bernoulli
        assert isinstance(
            self.Bernoulli, bool), "The keyword argument `Bernoulli` must be `True` or `False`."

    def add_loads_from_dict(self, d: dict, *args, **kwargs):
        self.loads = LoadGroup.from_dict(d)
        self.loads._Navier = self
        return self.loads

    def solve(self, *args, key='_coeffs', postproc=False, **kwargs):
        self._summary = {}
        self._key_coeff = key
        LC = list(self.loads.load_cases())
        D = self.D
        if self.Bernoulli and self.model in ['m', 'mindlin']:
            D = np.full(self.D.shape, 1e12)
        LHS = lhs_Navier(self.size, self.shape, D=D, S=self.S,
                         model=self.model, squeeze=False)
        RHS = list(lc.rhs() for lc in LC)
        [setattr(lc, '_rhs', rhs) for lc, rhs in zip(LC, RHS)]
        if self.model in ['k', 'kirchhoff']:
            RHS = [np.sum(rhs, axis=-1) for rhs in RHS]
        t0 = time()
        coeffs = linsolve(LHS, np.stack(RHS), squeeze=False)
        self._summary['dt'] = time() - t0
        nRHS, nLHS, nMN = coeffs.shape[:3]
        self._summary['nMN'] = nMN
        self._summary['nRHS'] = nRHS
        self._summary['nLHS'] = nLHS
        [setattr(lc, self._key_coeff, c) for lc, c in zip(LC, coeffs)]

    def postproc(self, points: np.ndarray, *args, cleanup=True,
                 key='res2d', **kwargs):
        self._key_res2d = key
        size = self.size
        shape = self.shape
        LC = list(self.loads.load_cases())
        RHS = np.stack(list(lc._rhs for lc in LC))
        coeffs = np.stack([getattr(lc, self._key_coeff) for lc in LC])
        ABDS = np.zeros((5, 5))
        ABDS[:3, :3] = self.D
        if self.S is not None:
            ABDS[3:, 3:] = self.S
        else:
            ABDS[3:, 3:] = 1e12
        res = postproc(ABDS, points, size=size, shape=shape, loads=RHS,
                       solution=coeffs, model=self.model)
        if len(LC) == 1:
            setattr(LC[0], self._key_res2d, np.squeeze(swap(res, 0, 1)))
        else:
            [setattr(lc, self._key_res2d, np.squeeze(swap(r2d, 0, 1)))
                for lc, r2d in zip(LC, res)]
        res = np.squeeze(res)
        if cleanup:
            [delattr(lc, self._key_coeff) for lc in LC]
