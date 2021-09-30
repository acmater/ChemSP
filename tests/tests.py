import unittest
import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import RBF

from chemsp.graphs import *

import matplotlib
import matplotlib.pyplot as plt


X = np.array([[0.1,0.1,0.1],
              [0.1,0.2,0.3],
              [0.2,0.4,0.3]])

class TestAdjacency(unittest.TestCase):
    def test_gen_adj(self):
        adj = adjacency(X,RBF(1))
        assert adj is not None, "Adjacency not successfully generated."

class TestDegree(unittest.TestCase):
    def test_degree(self):
        deg = degree(adjacency(X, RBF(1)))

class TestLaplacian(unittest.TestCase):
    def test_laplacian(self):
        lap = laplacian(adjacency(X,RBF(1)))

if __name__ == "__main__":
    unittest.main()
