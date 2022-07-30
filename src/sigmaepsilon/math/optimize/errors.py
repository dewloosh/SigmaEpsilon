# -*- coding: utf-8 -*-


__all__ = ['DegenerateProblemError', 'NoSolutionError', 'PreProcError', 'PostProcError']


class SolutionError(Exception):
    """
    Raises during the solution stage of a solution.
    
    """
    pass


class DegenerateProblemError(SolutionError):
    """
    The problem is degenerate. 
    
    The objective could be decreased, but only on the expense
    of violating positivity of the standard variables.
    
    """
    pass


class NoSolutionError(SolutionError):
    """
    There is no solution to this problem.
    
    Step size could be indefinitely increased in a
    direction without violating feasibility.
    
    """
    pass


class PreProcError(Exception):
    """
    Raises during the postprocessing stage of a solution.
    
    """
    pass


class PostProcError(Exception):
    """
    Raises during the postprocessing stage of a solution.
    
    """
    pass