from linkeddeepdict import LinkedDeepDict

from .elem import FiniteElement
from .meta import ABCFiniteElement
from .celldata import CellData

from .bernoulli2 import Bernoulli2 as B2
from .bernoulli3 import Bernoulli3 as B3

from .cst import CST_M, CST_P_MR, CST_S_MR
from .lst import LST_M, LST_P_MR, LST_S_MR

from .q4 import Q4_M, Q4_P_MR, Q4_S_MR
from .q9 import Q9_M, Q9_P_MR, Q9_S_MR

from .tet4 import TET4
from .tet10 import TET10

from .h8 import H8
from .h27 import H27

from .wedge import W6, W18

cells_dict = LinkedDeepDict(
    {
        "bernoulli": {"B2": B2, "B3": B3},
        "membrane": {
            "T3": CST_M,
            "T6": LST_M,
            "Q4": Q4_M,
            "Q9": Q9_M,
        },
        "mindlin": {"T3": CST_P_MR, "T6": LST_P_MR, "Q4": Q4_P_MR, "Q9": Q9_P_MR},
        "3d": {"H8": H8, "H27": H27, "TET4": TET4, "TET10": TET10},
    }
)
