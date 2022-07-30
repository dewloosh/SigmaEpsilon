# -*- coding: utf-8 -*-
import unittest

import math

from sigmaepsilon.core.tools.tools import float_to_str_sig, issequence, \
    suppress, timeit
from sigmaepsilon.core.tools.alphabet import alphabet, ordrange, latinrange, \
    urange, greekrange, arabicrange
from sigmaepsilon.core.tools.kwargtools import isinkwargs, allinkwargs, anyinkwargs, \
    getfromkwargs, popfromkwargs, getallfromkwargs, getasany, countkwargs


class TestTools(unittest.TestCase):

    def test_tools(self):
        assert float_to_str_sig(math.pi, sig=4) == '3.142'
        seq = [math.pi, math.pi]
        assert issequence(seq)
        assert not issequence(seq[0])
        assert float_to_str_sig(seq, sig=4) == ['3.142', '3.142']
        assert issequence([1, 2, 3])
        assert not issequence('123')

    def test_kwargtools(self):
        kwargs = dict(a = 1, c = 2)
        assert isinkwargs('a', **kwargs)
        assert all(isinkwargs(['a', 'c'], **kwargs))
        assert not isinkwargs('b', **kwargs)
        assert not any(isinkwargs(['b', 'd'], **kwargs))
        assert any(isinkwargs(['a', 'b'], **kwargs))
        assert any(isinkwargs(['c', 'd'], **kwargs))
        assert allinkwargs(['a', 'c'], **kwargs)
        assert allinkwargs('a', **kwargs)
        assert anyinkwargs(['d', 'c'], **kwargs)
        assert anyinkwargs('c', **kwargs)
        assert getfromkwargs(['a'], None, int, **kwargs) == [1]
        assert getfromkwargs(['b'], None, None, **kwargs) == [None]
        assert getallfromkwargs(['a', 'c'], None, **kwargs) == [1, 2]
        assert getasany(['a', 'b'], None, **kwargs) == 1
        
        d = {'E1': 1, 'E2': 2, 'G12': 12, 'NU23': 0}
        nE = countkwargs(lambda s: s[0] == 'E', **d)
        nG = countkwargs(lambda s: s[0] == 'G', **d)
        nNU = countkwargs(lambda s: s[0:2] == 'NU', **d)
        assert nE == 2
        assert nG == 1
        assert nNU == 1
        popfromkwargs(['E1'], d)
        assert isinstance(popfromkwargs(['E2'], d, astype=float)[0], float)
        assert 'E1' not in d
        
    def test_alphabet(self):
        for abctype in ['ord', 'latin', 'u', 'greek']:
            g = alphabet(abctype)
            [next(g) for i in range(5)]
        
        ordrange(5)
        latinrange(5)
        urange(5)
        greekrange(5)
        arabicrange(5)
                
        lrange = latinrange(5, start='i')
        grange = greekrange(5)
        orange = ordrange(5)
        arange = arabicrange(5, start=1)

        abc = alphabet('latin', start='i')
        next(abc)

        abc = alphabet(start='u')
        next(abc)
            
        abc = alphabet('u', start='\x03')
        next(abc)
        
        abc = alphabet('u', start='\x03')
        pokerstr = [next(abc) for _ in range(4)]
        
        
    def test_misc(self):
        
        @timeit
        def foo(): return None
        
        @suppress
        def foo(): return None
            


if __name__ == "__main__":
    unittest.main()
