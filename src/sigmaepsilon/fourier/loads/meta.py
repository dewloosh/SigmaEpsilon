from dewloosh.core.meta import ABCMeta_Weak
from linkeddeepdict import LinkedDeepDict

__all__ = ["ABC_LoadGroup"]


_string_to_dtype_ = {}


class ABCMeta_LoadGroup(ABCMeta_Weak):
    """
    Meta class for PointData and CellData classes.

    It merges attribute maps with those of the parent classes.
    """

    def __init__(self, name, bases, namespace, *args, **kwargs):
        super().__init__(name, bases, namespace, *args, **kwargs)

    def __new__(metaclass, name, bases, namespace, *args, **kwargs):
        cls = super().__new__(metaclass, name, bases, namespace, *args, **kwargs)

        # merge database fields
        _attr_map_ = namespace.get("_attr_map_", {})
        for base in bases:
            _attr_map_.update(base.__dict__.get("_attr_map_", {}))
        cls._attr_map_ = _attr_map_

        # add class to helpers
        tag = getattr(cls, "_typestr_", None)
        if isinstance(tag, str):
            _string_to_dtype_[tag] = cls
        return cls


class ABC_LoadGroup(LinkedDeepDict, metaclass=ABCMeta_LoadGroup):
    """
    Helper class that provides a standard way to create an ABC using
    inheritance.
    """

    __slots__ = ()

    @staticmethod
    def _string_to_dtype_(tag: str = None):
        return _string_to_dtype_[tag]
