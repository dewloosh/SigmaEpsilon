# -*- coding: utf-8 -*-
from abc import ABCMeta


__all__ = ["ABCMeta_Weak", "ABCMeta_Strong", "ABCMeta_Safe"]


def _is_callable(n, v): return callable(v) and ('__' not in n)


class ABCMeta_Weak(ABCMeta):
    """
    Standard python metaclass. It follows weak abstraction in the meaning, that
    it is enough to implement the abstarcts in the instance, they impose no
    restriction on the inherited class itself. Therefore, error only occurs at
    runtime.
    """

    @staticmethod
    def _get_cls_methods(namespace, nomagic=False):
        """
        Returns the callable items in the namespace of the class.
        If `nomagic=False`, magic functions with trailing double 
        underscores are not returned.
        """
        if not nomagic:
            return {name for name, value in namespace.items()
                    if callable(value)}
        else:
            return {name for name, value in namespace.items()
                    if _is_callable(name, value)}
            
    @staticmethod
    def _get_cls_abstracts(namespace):
        """
        Returns the callable items in the namespace of the class.
        If `nomagic=False`, magic functions with trailing double 
        underscores are not returned.
        """            
        return getattr(namespace, "__abstractmethods__", set())
            
    @staticmethod
    def _get_base_methods(bases):
        """
        Returns the callable items in the namespace of the class.
        If `nomagic=False`, magic functions with trailing double 
        underscores are not returned.
        """
        base_methods = {}
        for base in bases:
            base_methods[base.__name__] = {}
            for k, v in base.__dict__.items():
                if _is_callable(k, v):
                    base_methods[base.__name__][k] = v
        return base_methods
                
    @staticmethod
    def _get_base_abstracts(bases):
        """
        Returns the callable items in the namespace of the class.
        If `nomagic=False`, magic functions with trailing double 
        underscores are not returned.
        """
        base_abc_methods = {}
        for base in bases:
            base_abc_methods[base.__name__] = \
                getattr(base, "__abstractmethods__", set())
        return base_abc_methods


class ABCMeta_Strong(ABCMeta_Weak):
    """
    Strong Python metaclass. It follows strong abstraction in the meaning, that
    an abstract function of any of the base classes must be implemented or
    delayed with the use of another @abstractmethod decorator.
    Error occurs at compilation time.
    """

    def __init__(self, name, bases, namespace, *args, **kwargs):
        super().__init__(name, bases, namespace, *args, **kwargs)

    def __new__(metaclass, name, bases, namespace, *args, **kwargs):
        cls = super().__new__(metaclass, name, bases, namespace, *args,
                              **kwargs)
        cls_methods = metaclass._get_cls_methods(namespace)
        for base in bases:
            base_abstracts = set()
            for name in getattr(base, "__abstractmethods__", set()):
                value = getattr(cls, name, None)
                if getattr(value, "__isabstractmethod__", False):
                    base_abstracts.add(name)
            for abstract in base_abstracts:
                if abstract not in cls_methods:
                    err_str = """Can't create abstract class {classname}!
                    {classname} must implement abstract method {method} of
                    class {base_class}!""".format(
                        classname=name,
                        method=abstract,
                        base_class=base.__name__)
                    raise TypeError(err_str)
        return cls


class ABCMeta_Safe(ABCMeta_Weak):
    """
    Python metaclass for safe inheritance. Throws a TypeError
    if a method tries to shadow a definition in any of the base
    classes.
    """

    def __init__(self, name, bases, namespace, *args, **kwargs):
        super().__init__(name, bases, namespace, *args, **kwargs)

    def __new__(metaclass, name, bases, namespace, *args, **kwargs):
        cls = super().__new__(metaclass, name, bases, namespace, *args,
                              **kwargs)
        cls_methods = metaclass._get_cls_methods(namespace, nomagic=True)
        for base in bases:
            for method in cls_methods:
                if hasattr(base, method):
                    err_str = """Can't create abstract class {classname}!
                        Method {method} is already implemented in class
                        {base_class}!""".format(
                        classname=name,
                        method=method,
                        base_class=base.__name__)
                    raise TypeError(err_str)
        return cls