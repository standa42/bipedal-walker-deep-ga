import random

import numpy as np

from gym_evaluator import GymEnvironment
from inidividual import Individual
from network import Network
import tensorflow as tf

from utils import batch


class GeneticAlgorithm:
    
    def __init__(self, threads, env_name: str, max_episode_len: int, seed: int = 42):
        self._seed = seed
        gym = GymEnvironment(env_name)
        self._input_shape = gym.state_shape
        self._output_shape = gym.action_shape
        self._threads = threads
        self._env_name = env_name
        self._max_episode_len = max_episode_len

    def fit(self, generation_count, population_size, sigma, truncation_size, elitism_evaluations):
        """main ga cycle"""
        population = self.init_population(population_size)
        elite = None

        for g in range(generation_count):
            print(f"Generation {g} started \r")
            new_population = []

            # TODO roman
            # paralelize
            parents = population[:truncation_size]
            for _ in range(population_size):
                offspring = self.generate_offspring(parents, sigma)
                new_population.append(offspring)


            # descending sort
            #new_population.sort(key=lambda x: x.fitness, reverse=True)

            elite = self.get_elite(elite, new_population, elitism_evaluations)

            # append elite
            new_population.remove(elite)
            new_population = [elite] + new_population

            print(f"Generation {g} has elite fitness: {elite.fitness}")

    def generate_offspring(self, parents, sigma):
        """
        Generates offspring from selected list of parents adding sample from
         normal distribution with zero mean and sigma parameter.
        :param parents: Parents from which offspring will be generated.
        :param sigma: Sigma of normal distribution
        """
        chosen_parent: Individual = random.choice(parents)
        offspring = chosen_parent.clone()
        self.mutate(offspring, sigma)
        offspring.fitness = self.evaluate_fitness(offspring)
        return offspring

    def init_population(self, population_size):
        max_int = 2 ** 63 - 1
        population = []
        for _ in range(population_size):
            seed = random.randint(0, max_int)
            network = Network(self._input_shape, self._output_shape, seed=seed)

            ind = Individual(network)
            population.append(ind)
        return population

    async def evaluate_fitness(self, individual):
        """
        Evaluates fitness of specified individual.
        :param individual: Individual
        :return: Fitness of the individual
        """
        gym = GymEnvironment(self._env_name, seed=self._seed)
        state, done = gym.reset(), False

        network = individual.network

        step = 0
        total_rewards = 0
        while not done:
            gym.render()

            state = np.expand_dims(state, 0)
            action = network.predict(state)[0]
            next_state, reward, done, _ = gym.step(action)
            total_rewards += reward

            state = next_state
            step += 1

            if step >= self._max_episode_len:
                done = True
        return total_rewards

    def mutate(self, individual, sigma):
        """
        Mutates specified individual. It performs modifications on existing instance, doesn't create new one.
        :param individual:
        :param sigma:
        :return:
        """
        network = individual.network
        weights = network.get_weights()

        modified_weights = []
        for w in weights:
            update = tf.random.normal(shape=w.shape, stddev=sigma)
            modified_weights.append(w + update)
        network.set_weights(modified_weights)

    def get_elite(self, elite, population, elitism_evaluations):
        # TODO roman
        # elitism
        # candidates - 10 best + last gen elite
        choose_best_count = 10
        best_from_population = population[:choose_best_count]
        candidates = best_from_population
        if elite is not None:
            candidates.append(elite)
        # choose best candidate according to mean in -elitism_evaluations- evals
        from statistics import mean
        for candidate in candidates:
            candidate_fitnesses = []
            for _ in range(elitism_evaluations):
                # TODO: evaluate n-times
                candidate_fitnesses.append(self.evaluate_fitness(candidate))
                pass
            candidate.fitness = mean(candidate_fitnesses)

        import operator
        new_elite = max(candidates, key=operator.attrgetter('fitness'))

