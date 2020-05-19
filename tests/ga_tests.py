import asyncio
import unittest
import numpy as np
import tensorflow as tf
import random

from ga import GeneticAlgorithm


def async_test(f):
    def wrapper(*args, **kwargs):
        coro = asyncio.coroutine(f)
        future = coro(*args, **kwargs)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(future)
    return wrapper


class GeneticAlgorithmTestCase(unittest.TestCase):
    def setUp(self):
        self.ga = GeneticAlgorithm(threads=1, env_name="BipedalWalker-v3", max_episode_len=100, seed=42)

    def test_smoke_mutate(self):
        sigma = 0.002
        ind = self.ga.init_population(1)[0]

        mutated = self.ga.mutate(ind, sigma)

    def test_smoke_evalute_fitness(self):
        ind = self.ga.init_population(1)[0]
        fitness = self.ga.evaluate_fitness(ind)

    @async_test
    async def test_smoke_async_evaluate_fitness(self):
        ind = self.ga.init_population(1)[0]
        task = asyncio.gather(
            asyncio.ensure_future(self.ga.evaluate_fitness(ind)),
            asyncio.ensure_future(self.ga.evaluate_fitness(ind)))

        await task




if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    unittest.main()
