# -*- coding: utf-8 -*-
from linkeddeepdict import LinkedDeepDict

from .elem import FiniteElement
from .meta import ABCFiniteElement
from .celldata import CellData

from .bernoulli2 import Bernoulli2 as B2
from .bernoulli3 import Bernoulli3 as B3

from .cst import CSTM, CSTP
from .lst import LSTM, LSTP

from .q4 import Q4M, Q4P
from .q9 import Q9M, Q9P

from .tet4 import TET4
from .tet10 import TET10

from .h8 import H8
from .h27 import H27


cells_dict = LinkedDeepDict({
    'bernoulli' : {
        'B2' : B2, 'B3' : B3
    },
    'membrane' : {
        'T3' : CSTM, 'T6' : LSTM,
        'Q4' : Q4M, 'Q9' : Q9M,
    },
    'mindlin' : {
        'T3' : CSTP, 'T6' : LSTP,
        'Q4' : Q4P, 'Q9' : Q9P
    },
    '3d' : {
        'H8' : H8, 'H27' : H27, 
        'TET4' : TET4, 'TET10' : TET10
    }
})