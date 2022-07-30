# -*- coding: utf-8 -*-
import unittest
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix as csr_scipy

from sigmaepsilon.math.linalg.sparse.csr import csr_matrix, csr_matrix_nb


class TestCSR(unittest.TestCase):

    def test_csr(self):
        @njit
        def csr_row(csr: csr_matrix, i: int):
            return csr.row(i)

        @njit
        def csr_data(csr: csr_matrix):
            return csr.data

        @njit
        def csr_m(dtype=np.float64):
            return csr_matrix_nb(dtype)

        np.random.seed = 0
        mat = csr_scipy(np.random.rand(10, 12) > 0.8, dtype=int)
        print(mat.A)

        csr = csr_matrix(mat)
        sdata, scols = csr_row(csr, 0)
        print(sdata, scols)

        e = csr_matrix.eye(3)

        print(csr_data(csr))


if __name__ == "__main__":

    unittest.main()
