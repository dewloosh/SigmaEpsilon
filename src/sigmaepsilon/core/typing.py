# -*- coding: utf-8 -*-
from typing import Hashable
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import six


__all__ = ["issequence", "ishashable"]


def issequence(arg) -> bool:
    """
    Returns `True` if `arg` is any kind of iterable, but not a string,
    returns `False` otherwise.
    
    Examples
    --------
    The formatter to use to print a floating point number with 4 digits:
    
    >>> from dewloosh.core.tools import issequence
    >>> issequence([1, 2])
    True
    
    To print the actual value as a string:
    
    >>> issequence('lorem ipsum')
    False    
    """
    return (
        isinstance(arg, Iterable)
        and not isinstance(arg, six.string_types)
    )
    

def ishashable(obj):
    """
    Returns `True` if `obj` is hashable.
    """
    return isinstance(obj, Hashable) 