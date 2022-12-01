# -*- coding: utf-8 -*-
from enum import Enum
from typing import Iterable, List


class DOF(Enum):
    """
    Degrees of freedom.
    
    Examples
    --------
    >>> from sigmaepsilon.solid.fem.config import DOF
    >>> DOF['UXX']
    <DOF.ROTX: 3>
    >>> DOF.UXX
    <DOF.ROTX: 3>
    >>> DOF.UXX.value
    3
    >>> DOF.UXX.name
    'ROTX'
    
    """
    UX = 0
    UY = 1
    UZ = 2
    ROTX = UXX = UYZ = 3
    ROTY = UYY = UXZ = 4
    ROTZ = UZZ = UXY = 5
    
    @classmethod
    def dofmap(cls, dofs:Iterable) -> List:
        return [cls[d].value for d in dofs]
    

if __name__=='__main__':

    DOF.UXX, DOF['UXX']
    type(DOF['UXX'].value)
