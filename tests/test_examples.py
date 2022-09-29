# -*- coding: utf-8 -*-
import unittest

from sigmaepsilon.examples import console_grid_bernoulli

class TestExamples(unittest.TestCase):
    
    def test_example_console_bernoulli_grid(self):
        structure = console_grid_bernoulli()
        structure.linsolve()
                                
    
if __name__ == "__main__":
    
    unittest.main()