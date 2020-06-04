import operator
import os
import random
from time import time

import numpy as np

from gym_evaluator import GymEnvironment
from inidividual import Individual
from network import Network

from multiprocessing import Pool
import tensorflow as tf
import gc

import gym
import csv
gym.logger.set_level(40)


class GeneticAlgorithm:
    def __init__(self, threads, env_name: str, max_episode_len: int, elite_choose_best_count: int, render_each: int, logdir, nn_width: int, seed: int = 42 ):
        self._seed = seed
        gym = GymEnvironment(env_name)
        self._input_shape = gym.state_shape
        self._output_shape = gym.action_shape
        self._threads = threads
        self._env_name = env_name
        self._max_episode_len = max_episode_len
        self._elite_choose_best_count = elite_choose_best_count
        self._render_each = render_each
        self._logdir = logdir
        self.nn_width = nn_width

    def fit(self, generation_count, population_size, sigma, truncation_size, elitism_evaluations, sigma_final=None):
        """main ga cycle"""
        population = self.init_population(population_size)
        elite = None
        output_csv_path = os.path.join(self._logdir, "metrics.csv")
        if not sigma_final:
            sigma_final = sigma
        sigmas = np.linspace(sigma, sigma_final, generation_count)

        for g in range(1, generation_count + 1):
            sigma = sigmas[g]
            generation_start_time = time()
            print(f"GENERATION {g}", flush=True)
            new_population = []

            # paralelize
            parents = population[-truncation_size:]
            for _ in range(population_size):
                offspring = self.generate_offspring(parents, sigma)
                new_population.append(offspring)
            print(f"Offspring generation ({time() - generation_start_time:.2f}s)", end="", flush=True)

            start_time = time()
            max_int = 2 ** 63 - 1
            # fitnesses = [self.evaluate_fitness(
            #     (ind.network.get_weights(), random.randint(0, max_int))) for ind in new_population]
            with Pool(self._threads) as pool:
                fitnesses = pool.map(self.evaluate_fitness, [
                    (ind.network.get_weights(), random.randint(0, max_int)) for ind in new_population])
            print(f"\rFitness computation ({time() - start_time:.2f}s)", end="", flush=True)

            for index in range(len(population)):
                new_population[index].fitness = fitnesses[index]

            # descending sort
            new_population.sort(key=lambda x: x.fitness)

            start_time = time()
            elite = self.get_elite(elite, new_population, elitism_evaluations)
            print(f"\rElite chosen ({time() - start_time:.2f}s)", end="", flush=True)

            # remove elite (if it exists) and readd it
            try:
                new_population.remove(elite)
            except ValueError:
                # if there's no elite, exclude first member (with worst fitness)
                new_population.pop(0)
            new_population.append(elite)
            population = new_population

            # specify and log metrics
            fitnesses = np.array([ind.fitness for ind in population])

            mean = np.mean(fitnesses)
            std = np.std(fitnesses, ddof=1)
            quantiles = np.quantile(fitnesses, [0.25, 0.5, 0.75])
            best_fitness = elite.fitness
            with open(output_csv_path, "a") as file:
                writer = csv.writer(file, delimiter=",")
                writer.writerow([best_fitness, mean, std, quantiles[0], quantiles[1], quantiles[2]])
            # save network weights of the elite
            elite.network.save_weights(os.path.join(self._logdir, f"cp-{g}.h5"))

            output_str = f"best: {best_fitness:.4f}, mean: {mean:.4f}, std: {std:.4f}, q1: {quantiles[0]:.4f}, " \
                         f"q2(med): {quantiles[1]:.4f}, q3: {quantiles[2]:.4f}"
            print(f"\rGeneration {g} ({time() - generation_start_time:.2f}s): ", output_str, flush=True)
            print()

            gc.collect()
            tf.keras.backend.clear_session()

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
        return offspring

    def init_population(self, population_size):
        max_int = 2 ** 63 - 1
        population = []
        for _ in range(population_size):
            seed = random.randint(0, max_int)
            network = Network(self._input_shape, self._output_shape, seed=seed, nn_width=self.nn_width)

            ind = Individual(network)
            population.append(ind)
        return population

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
        # elitism
        # candidates - 10 best + last gen elite
        max_int = 2 ** 63 - 1
        choose_best_count = self._elite_choose_best_count
        best_from_population = population[-choose_best_count:]
        candidates = best_from_population
        if elite is not None:
            candidates.append(elite)
        # choose best candidate according to mean in -elitism_evaluations- evals
        for candidate in candidates:
            # candidate_fitnesses = [self.evaluate_fitness((ind.network.get_weights(), random.randint(0, max_int)))
            #                        for ind in [candidate] * elitism_evaluations]
            with Pool(self._threads) as pool:
                candidate_fitnesses = pool.map(self.evaluate_fitness, [
                    (ind.network.get_weights(), random.randint(0, max_int)) for ind in [candidate] * elitism_evaluations])
            candidate_fitnesses.append(candidate.fitness)
            candidate.fitness = np.mean(candidate_fitnesses)

        new_elite = max(candidates, key=operator.attrgetter('fitness'))
        return new_elite

    def evaluate_fitness(self, params):
        """
        Evaluates fitness of specified individual.
        :return: Fitness of the individual
        """
        network_weights = params[0]
        seed = params[1]
        network = Network(self._input_shape, self._output_shape, seed, nn_width=self.nn_width, initializer="zeros")
        network.set_weights(network_weights)

        gym = GymEnvironment(self._env_name, seed=seed)
        state, done = gym.reset(), False

        # start_time = time()

        step = 0
        equal_steps = 0
        rewards = []
        min_equal_steps = 5
        while not done:
            if self._render_each and step % self._render_each == 0:
                gym.render()

            state = np.expand_dims(state, 0)
            action = network(state).numpy()[0]
            next_state, reward, done, _ = gym.step(action)
            if np.allclose(state, next_state):
                equal_steps += 1
            else:
                equal_steps = 0
            rewards.append(reward)

            state = next_state
            step += 1

            if step >= self._max_episode_len:
                done = True
            elif equal_steps >= min_equal_steps:
                done = True
                # add expected reward if we waited till the episode would end
                rewards.append((self._max_episode_len - step) * np.mean(rewards[-min_equal_steps:]))
        # print(f"Total steps {step}: {time() - start_time:.4f}")

        total_reward = np.sum(rewards)

        return total_reward