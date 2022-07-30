# -*- coding: utf-8 -*-
from types import FunctionType
from abc import abstractmethod


def attributor(*attrs: str) -> FunctionType:
    """
    It renders a decorator a default behaviour. If a decorator
    is called with a None argument, it returns the attribute, otherwise it 
    returns the decorated function.
    """
    abstract = '__isabstractmethod__' in attrs
    if abstract:
        attributes = [x for x in attrs if x != '__isabstractmethod__']
    else:
        attributes = attrs

    def decorator(fnc):
        if fnc is None:
            return attrs
        else:
            for attr in attributes:
                setattr(fnc, attr, True)
        if abstract:
            return abstractmethod(fnc)
        return fnc
    return decorator


if __name__ == '__main__':

    axiom = attributor('__isaxiom__')
    abstractaxiom = attributor('__isaxiom__', '__isabstractmethod__')

    @axiom
    def foo(a, b):
        return 'an axiom'
