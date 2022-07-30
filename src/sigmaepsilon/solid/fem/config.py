# -*- coding: utf-8 -*-
from enum import Enum, auto, unique

@unique
class DOF(Enum):
    UX = auto()
    UY = auto()
    UZ = auto()
    UYZ = auto()
    UXZ = auto()
    UXY = auto()


print(list(DOF))
print(list(map(lambda dof : dof.value, DOF)))