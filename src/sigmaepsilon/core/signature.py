# -*- coding: utf-8 -*-
from inspect import signature, Parameter
from typing import Any
from typing import Generic as _GenericAlias


__all__ = ["Signature"]


class Signature(dict):
    """
    A class to differentiate between function declarations using their
    type signatures. It helps to decide if an implementation satisfies
    some requirements imposed on a class.
    """
    
    __abckey__ = 'isabstractoperation'

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        setattr(self, 'isabstract', 'abstract' in args)

    @classmethod
    def from_function(cls, funcobj, *attrs, **kwargs):

        if getattr(funcobj, cls.__abckey__, False):
            sig = Signature('abstract', **kwargs)
        else:
            sig = Signature(**kwargs)

        sig['name'] = funcobj.__name__
        if len(attrs) > 0:
            sig['attrs'] = frozenset(attrs)
        else:
            sig['attrs'] = frozenset()

        params = signature(funcobj).parameters
        if 'args' in params or 'kwargs' in params:
            raise TypeError(
                'Operation can only have a finite number of arguments!')
        if 'self' in params or 'cls' in params:
            sig['arity'] = len(params)-1
        else:
            sig['arity'] = len(params)

        if sig['arity'] == 0:
            sig['dtype'] = [Any]
        else:
            sig['dtype'] = []
            for pname, param in params.items():
                if pname not in ['self', 'cls']:
                    if param.annotation is not Parameter.empty:
                        sig['dtype'].append(param.annotation)
                    else:
                        sig['dtype'].append(Any)

        annotations = funcobj.__annotations__
        if 'return' in annotations:
            sig['rtype'] = annotations['return']
        else:
            sig['rtype'] = Any

        return sig

    @classmethod
    def from_property(cls, funcobj, **kwargs):

        if funcobj.isabstractattribute:
            sig = Signature('abstract', **kwargs)
        else:
            sig = Signature(**kwargs)

        sig['name'] = funcobj.__name__

        annotations = funcobj.__annotations__
        if 'return' in annotations:
            sig['rtype'] = annotations['return']
        else:
            sig['rtype'] = Any

        return sig

    def compatible_function(self, other: 'Signature') -> bool:
        """
        Returns True if two instances are compatible in terms of
        domain type and result type.
        """
        if not isinstance(other, Signature):
            return False

        # check domain types
        for type1, type2 in zip(self['dtype'], other['dtype']):
            if type1 == Any:
                continue
            elif isinstance(type1, _GenericAlias):
                if not isinstance(type2, _GenericAlias):
                    typeargs = type1.__dict__['__args__']
                    if type2 not in typeargs:
                        return False
                else:
                    if type1 != type2:
                        return False
            else:
                if type1 != type2:
                    return False

        # check result type
        if isinstance(self['rtype'], _GenericAlias):
            if not isinstance(other['rtype'], _GenericAlias):
                typeargs = self['rtype'].__dict__['__args__']
                if not other['rtype'] in typeargs:
                    return False
            else:
                if self['rtype'] != other['rtype']:
                    return False
        elif self['rtype'] != Any:
            if self['rtype'] != other['rtype']:
                return False

        return True

    def accepts_property(self, other: 'Signature') -> bool:
        """
        Returns True if other (implemented operation's signature) is
        compatible to self (abstract operation's signature).
        Every value to every key of self must be present in other.
        For example, here is where axioms are forced to match
        between two signatures.
        """
        if any([not self.isabstract, other.isabstract]):
            return False
        if not self.compatible_op(other):
            return False
        if not self['name'] == other['name']:
            return False

        for key, value in self.items():
            if key not in ['rtype', 'dtype']:
                if key in other:
                    if isinstance(value, frozenset) and \
                            isinstance(other[key], frozenset):
                        if not value.issubset(other[key]):
                            return False
                    else:
                        if not value == other[key]:
                            return False
                else:
                    return False
        return True

    def update_function(self, other: 'Signature'):
        """
        Updates self with the content of other. Both must be abstracts.
        """
        assert isinstance(other, Signature)
        assert self.isabstract and other.isabstract
        assert self.compatible_op(other)
        for key, value in other.items():
            if key not in ['rtype', 'dtype']:
                if isinstance(value, frozenset):
                    if key in self and isinstance(self[key], frozenset):
                        s = set(self[key])
                        s.update(set(value))
                        self[key] = frozenset(s)
                    else:
                        self[key] = value
                else:
                    self[key] = value
        return

    def accepts_property(self, value: Any = None) -> bool:
        """
        Returns True if other (implemented attribute's signature) is
        compatible to self (abstract attribute's signature).
        Name and result type must match.
        """
        if not self.isabstract:
            return False
        if value is None:
            return False

        # check result type
        if isinstance(self['rtype'], _GenericAlias):
            if not isinstance(type(value), _GenericAlias):
                typeargs = self['rtype'].__dict__['__args__']
                if not type(value) in typeargs:
                    return False
            else:
                if self['rtype'] != type(value):
                    return False
        elif self['rtype'] != Any:
            if self['rtype'] != type(value):
                return False
        return True

    def accepts_parameters(self, *args):
        raise NotImplementedError
