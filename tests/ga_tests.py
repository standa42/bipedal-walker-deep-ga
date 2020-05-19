import unittest
import numpy as np
import tensorflow as tf
import random

from ga import GeneticAlgorithm


class GeneticAlgorithmTestCase(unittest.TestCase):
    def setUp(self):
        self.ga = GeneticAlgorithm(threads=1)

    def test_mutate(self):
        sigma = 0.002
        ind = self.ga.init_population(1)[0]

        mutated = self.ga.mutate(ind, sigma)
        self.assertEqual(True, False)


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    unittest.main()
