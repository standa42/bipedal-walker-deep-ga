import unittest

from ga import GeneticAlgorithm


class GeneticAlgorithmTestCase(unittest.TestCase):
    def setUp(self):
        self.ga = GeneticAlgorithm()

    def test_mutate(self):
        sigma = 0.002
        ind = self.ga.init_population(1)[0]

        mutated = self.ga.mutate(ind, sigma)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
