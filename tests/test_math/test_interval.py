# -*- coding: utf-8 -*-
import unittest
import numpy as np
#from numba import jit
#from hypothesis import settings

from sigmaepsilon.math.interval import Interval


"""settings.register_profile(
    "interval",
    max_examples=10,
    deadline=None,
)"""


#@jit(nopython=True, cache=True)
def inside_interval(interval, x):
    return interval.lo <= x < interval.hi


#@jit(nopython=True, cache=True)
def interval_width(interval):
    return interval.width


#@jit(nopython=True, cache=True)
def interval_data(interval):
    return interval.data


#@jit(nopython=True, cache=True)
def interval_getitem(interval, i):
    return interval[i]


#@jit(nopython=True, cache=True)
def new_interval(lo, hi, data):
    return Interval(lo, hi, data)


"""class TestInterval(unittest.TestCase):

    # @given(st.floats(min_value=1., max_value=3.),
    #       st.floats(min_value=1., max_value=3.))
    # @settings(settings.load_profile("interval"))
    def test_interval(self, lo=1.0, hi=3.0):
        data = np.array([1.1, 3.1, 2.1])
        new_interval(lo, hi, data)._arr
        interval_data(new_interval(lo, hi, data))
        interval_getitem(new_interval(lo, hi, data), 0)"""


if __name__ == "__main__":

    # small example to compile functions
    """lo = 1.0
    hi = 3.0
    data = np.array([1.1, 3.1, 2.1])
    new_interval(lo, hi, data)._arr
    interval_data(new_interval(lo, hi, data))
    interval_getitem(new_interval(lo, hi, data), 0)"""

    unittest.main()
