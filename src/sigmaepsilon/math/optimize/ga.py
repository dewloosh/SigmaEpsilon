# -*- coding: utf-8 -*-
from abc import abstractmethod
from typing import Iterable, Callable
import numpy as np


__all__ = ['GeneticAlgorithm', 'BinaryGeneticAlgorithm']


def even(n): return n % 2 == 0
def odd(n): return not even(n)


class GeneticAlgorithm:
    """Base class for Genetic Algorithms (GA). Use this as a base 
    class to your custom implementation of a GA.
                
    """

    def __init__(self, fnc: Callable, ranges: Iterable, *args, 
                 length=5, p_c=1, p_m=0.2, nPop=100, **kwargs):
        super().__init__()
        self.fnc = fnc
        self.ranges = np.array(ranges)
        self.dim = getattr(fnc, 'dimension', self.ranges.shape[0])
        self.length = length
        self.p_c = p_c
        self.p_m = p_m

        # Second half of the population is used as a pool to make parents.
        # This assumes that population size is a multiple of 4.
        if odd(nPop):
            nPop += 1
        if odd(int(nPop/2)):
            nPop += 2
        assert nPop % 4 == 0
        assert nPop >= 4

        self.nPop = nPop
        self._genotypes = None
        self._fittness = None
        self.reset()
        self._set_solution_params(**kwargs)

    @property
    def genotypes(self):
        return self._genotypes

    @genotypes.setter
    def genotypes(self, value):
        self._genotypes = value
        self.phenotypes = self.decode(self._genotypes)

    def reset(self):
        self._evolver = self.evolver()
        self._evolver.send(None)

    def _set_solution_params(self, tol=1e-12, maxiter=200, miniter=100,
                             elitism=1, **kwargs):
        self.tol = tol
        self.maxiter = np.max([miniter, maxiter])
        self.miniter = np.min([miniter, maxiter])
        self.elitism = elitism

    def evolver(self):
        self.genotypes = self.populate()
        _ = yield
        while True:
            self.genotypes = self.populate(
                self.select(self._genotypes, self.phenotypes))
            yield self._genotypes

    def evolve(self, cycles=1):
        for _ in range(cycles):
            next(self._evolver)
        return self.genotypes

    def criteria(self) -> bool:
        value = yield
        while True:
            _value = yield
            yield abs(value - _value) < self.tol
            value = _value

    def solve(self, reset=False, returnlast=False, **kwargs):
        """Returns the best phenotype."""
        if reset:
            self.reset()
        self._set_solution_params(**kwargs)
        criteria = self.criteria()
        criteria.send(None)
        criteria.send(self.fnc(self.best_phenotype()))
        finished = False
        nIter = 0
        while (not finished and nIter < self.maxiter) or \
                (nIter < self.miniter):
            next(self._evolver)
            finished = criteria.send(self.fnc(self.best_phenotype()))
            next(criteria)
            nIter += 1
        self.nIter = nIter
        return self.best_phenotype(lastknown=returnlast)

    def fittness(self, phenotypes=None):
        if phenotypes is not None:
            self._fittness = np.array([self.fnc(x) for x in phenotypes],
                                      dtype=float)
        return self._fittness

    def best_phenotype(self, lastknown=False):
        if lastknown:
            fittness = self._fittness
        else:
            fittness = self.fittness(self.phenotypes)
        best = np.argmin(fittness)
        return self.phenotypes[best]

    def divide(self, fittness=None):
        """Divides population to elit and others, and returns the corresponding 
        index arrays.

        Parameters
        ----------
        fittness : Iterable, optional
            Fittness values. If not provided, values from the latest
            evaluation are used.

        Returns
        -------
        list
            Indices of the members of the elite.

        list
            Indices of the members of the others.

        """
        fittness = self.fittness() if fittness is None else fittness
        assert fittness is not None, "No available fittness data detected."
        if self.elitism < 1:
            argsort = np.argsort(fittness)
            elit = argsort[:int(self.nPop*self.elitism)]
            others = argsort[int(self.nPop*self.elitism):]
        elif self.elitism > 1.1:
            argsort = np.argsort(fittness)
            elit = argsort[:self.elitism]
            others = argsort[self.elitism:]
        else:
            elit = []
            others = list(range(self.nPop))
        return list(elit), others

    @classmethod
    def random_parents_generator(cls, genotypes=None):
        """Yields random pairs from a list of genotypes.

        The implamantation assumes theat the length of the input array 
        is a multiple of 2.

        Parameters
        ----------
        genotypes : Iterable
            Genotypes of the parents as a 2d integer array.

        """
        n = len(genotypes)
        assert n % 2 == 0
        pool = np.full(n, True)
        nPool = n
        while nPool > 2:
            where = np.argwhere(pool == True).flatten()
            nPool = len(where)
            pair = np.random.choice(where, 2, replace=False)
            parent1 = genotypes[pair[0]]
            parent2 = genotypes[pair[1]]
            pool[pair] = False
            yield parent1, parent2

    @abstractmethod
    def populate(self, genotypes=None):
        """To be implemented"""
        ...

    @abstractmethod
    def decode(self, genotypes=None):
        """To be implemented"""
        ...

    @abstractmethod
    def crossover(self, parent1=None, parent2=None):
        """To be implemented"""
        ...

    @abstractmethod
    def mutate(self, child=None):
        """To be implemented"""
        ...

    @abstractmethod
    def select(self, genotypes=None, phenotypes=None):
        """To be implemented"""
        ...


class BinaryGeneticAlgorithm(GeneticAlgorithm):
    """
    An implementation of a Binary Genetic Algorithm (BGA) for finding 
    minimums of real valued unconstrained problems of continuous variables 
    in n-dimensional vector spaces. 
    
    In other words, it solves the following problem:
    
    .. math::
            :nowrap:

            \\begin{eqnarray}
                & minimize&  \quad  f(\mathbf{x}) \quad in \quad \mathbf{x} \in \mathbf{R}^n. 
            \\end{eqnarray}
    
    Parameters
    ----------
        fnc : Callable
            The fittness function.

        ranges : Iterable
            sequence of pairs of limits for each variable

        length : int, Optional.
            Chromosome length (string length). Default is 5.

        p_c : float, Optional. 
            Probability of crossover, 0 <= p_c <= 1. Default is 1.

        p_m : float, Optional.
            Probability of mutation, 0 <= p_m <= 1. Default is 0.2.

        nPop : int, Optional
            Number of members in the population. Default is 100.

        elitism : float or integer, Optional.
            Value to control elitism. Default is 1.

    Examples
    --------
    Find the minimizer of the Rosenbrock function. 
    The exact value of the solution is x = [1.0, 1.0].

    >>> def Rosenbrock(x, y):
    >>>     a = 1, b = 100
    >>>     return (a-x)**2 + b*(y-x**2)**2
    >>> ranges = [
    >>>     [-10, 10],
    >>>     [-10, 10]
    >>> ]
    >>> BGA = BinaryGeneticAlgorithm(Rosenbrock, ranges, length=12, nPop=200)
    >>> BGA.solve()
    array([0.99389553, 0.98901176])
    
    The following code prints the history using the `evolve` generator of 
    the object
    
    >>> import matplotlib.pyplot as plt
    >>> BGA = BinaryGeneticAlgorithm(f, ranges, length=12, nPop=200)
    >>> history = [f(BGA.best_phenotype())]
    >>> for _ in range(100):
    >>>     BGA.evolve(1)
    >>>     history.append(f(BGA.best_phenotype()))
    >>> plt.plot(history)
    >>> plt.show()
    >>> x = BGA.best_phenotype()
    >>> fx = f(x)
    >>> print('min {} @ {}'.format(fx,x))

    """

    def populate(self, genotypes=None):
        """
        Populates the model from a list of genotypes as seeds.
        
        """
        nPop = self.nPop
        if genotypes is None:
            poolshape = (int(nPop / 2), self.dim * self.length)
            genotypes = np.random.randint(2, size=poolshape)
        else:
            poolshape = genotypes.shape
        nParent = poolshape[0]
        if nParent < nPop:
            offspring = []
            g = self.random_parents_generator(genotypes)
            try:
                while (len(offspring) + nParent) < nPop:
                    parent1, parent2 = next(g)
                    offspring.extend(self.crossover(parent1, parent2))
                genotypes = np.vstack([genotypes, offspring])
            except Exception:
                raise RuntimeError
        return genotypes

    def decode(self, genotypes: np.ndarray = None) -> np.ndarray:
        """
        Decodes the genotypes to phenotypes.
        
        """
        span = (2**self.length - 2**0)
        genotypes = genotypes.reshape((self.nPop, self.dim, self.length))
        precisions = [(self.ranges[d, -1] - self.ranges[d, 0]) / span
                      for d in range(self.dim)]
        phenotypes = \
            np.sum([genotypes[:, :, i]*2**i
                    for i in range(self.length)], axis=0).astype(float)
        for d in range(self.dim):
            phenotypes[:, d] *= precisions[d]
            phenotypes[:, d] += self.ranges[d, 0]
        return phenotypes

    def crossover(self, parent1=None, parent2=None, nCut=None):
        """
        Performs crossover on the parents `parent1` and `parent2`, 
        using an `nCut` number of cuts.
        
        """
        if np.random.rand() > self.p_c:
            return parent1, parent2

        if nCut is None:
            nCut = np.random.randint(1, self.dim*self.length-1)

        cuts = [0, self.dim * self.length]
        p = np.random.choice(range(1, self.length * self.dim - 1),
                             nCut, replace=False)
        cuts.extend(p)
        cuts = np.sort(cuts)

        child1 = np.zeros(self.dim*self.length, dtype=int)
        child2 = np.zeros(self.dim*self.length, dtype=int)

        randBool = np.random.rand() > 0.5
        for i in range(nCut+1):
            if (i % 2 == 0) == randBool:
                child1[cuts[i]:cuts[i+1]] = parent1[cuts[i]:cuts[i+1]]
                child2[cuts[i]:cuts[i+1]] = parent2[cuts[i]:cuts[i+1]]
            else:
                child1[cuts[i]:cuts[i+1]] = parent2[cuts[i]:cuts[i+1]]
                child2[cuts[i]:cuts[i+1]] = parent1[cuts[i]:cuts[i+1]]

        return self.mutate(child1), self.mutate(child2)

    def mutate(self, child=None):
        """
        Mutates a child.
        
        """
        p = np.random.rand(self.dim*self.length)
        return np.where(p > self.p_m, child, 1-child)

    def select(self, genotypes=None, phenotypes=None):
        """
        Organizes a tournament and returns the winners.
        
        """
        fittness = self.fittness(phenotypes)
        winners, others = self.divide(fittness)
        while len(winners) < int(self.nPop / 2):
            candidates = np.random.choice(others, 3, replace=False)
            winner = np.argsort([fittness[ID] for ID in candidates])[0]
            winners.append(candidates[winner])
        return np.array([genotypes[w] for w in winners], dtype=float)