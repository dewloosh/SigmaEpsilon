# -*- coding: utf-8 -*-
from .meta import *


__all__ = ["ABC_Strong", "ABC_Weak", "ABC_Safe"]


class ABC_Weak(metaclass=ABCMeta_Weak):
    """
    Helper class that provides a standard way to create an ABC using
    inheritance.
    """
    __slots__ = ()


class ABC_Strong(metaclass=ABCMeta_Strong):
    """Helper class that provides a standard way to create an ABC using
    inheritance.
    """
    __slots__ = ()


class ABC_Safe(metaclass=ABCMeta_Safe):
    """
    Helper class that provides a standard way to create an ABC using
    inheritance.
    """
    __slots__ = ()

