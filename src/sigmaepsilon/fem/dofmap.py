from enum import Enum
from typing import Iterable, List


class DOF(Enum):
    """
    An enumeration class to handle degrees of freedom.
    It acceprs several keys for the same DOF to help dissolve
    ambiguities between different notations.

    Examples
    --------
    >>> from sigmaepsilon.fem.config import DOF
    >>> DOF['UXX']
    <DOF.ROTX: 3>
    >>> DOF.UXX
    <DOF.ROTX: 3>
    >>> DOF.UXX.value
    3
    >>> DOF.UXX.name
    'ROTX'

    The keys 'UX', 'U1' and 'U' all refer to the same thing:

    >>> DOF.dofmap(['U', 'UX', 'U1'])
    [0, 0, 0]

    Similarly, the rotation around X can be referenced using
    multiple keys (for rotations, a total of 7 versions are available):

    >>> DOF.dofmap(['ROTX', 'UXX', 'UYZ', 'U32', 'U23'])
    [3, 3, 3, 3, 3]

    """

    UX = U1 = U = 0  # displacement in X direction
    UY = U2 = V = 1  # displacement in Y direction
    UZ = U3 = W = 2  # displacement in Z direction
    ROTX = UXX = UYZ = UZY = U11 = U23 = U32 = 3  # rotation around X
    ROTY = UYY = UXZ = UZX = U22 = U13 = U31 = 4  # rotation around Y
    ROTZ = UZZ = UXY = UYX = U33 = U12 = U21 = 5  # rotation around Z

    @classmethod
    def dofmap(cls, dofs: Iterable) -> List:
        """
        Returns indices of dofs.

        Examples
        --------
        >>> from sigmaepsilon.fem.config import DOF
        >>> DOF.dofmap(['UX', 'U', 'UXX'])
        [0, 0, 3]
        """
        return [cls[d].value for d in dofs]


if __name__ == "__main__":
    DOF.UXX, DOF["UXX"]
    type(DOF["UXX"].value)
