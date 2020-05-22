import operator
import random
from time import time

import numpy as np

from gym_evaluator import GymEnvironment
from inidividual import Individual
from network import Network

from multiprocessing import Pool

import gym
gym.logger.set_level(40)


class GeneticAlgorithm:
    def __init__(self, threads, env_name: str, max_episode_len: int, render_each: int, seed: int = 42):
        self._seed = seed
        gym = GymEnvironment(env_name)
        self._input_shape = gym.state_shape
        self._output_shape = gym.action_shape
        self._threads = threads
        self._env_name = env_name
        self._max_episode_len = max_episode_len
        self._render_each = render_each

    def fit(self, generation_count, population_size, sigma, truncation_size, elitism_evaluations):
        """main ga cycle"""
        population = self.init_population(population_size)
        elite = None

        for g in range(generation_count):
            generation_start_time = time()
            print(f"Generation {g} started ")
            new_population = []

            # paralelize
            parents = population[:truncation_size]
            for _ in range(population_size):
                offspring = self.generate_offspring(parents, sigma)
                new_population.append(offspring)
            print(f"Generation {g}: offspring generation, time elapsed: {time() - generation_start_time:.2f}")

            start_time = time()
            # fitnesses = [self.evaluate_fitness(ind.network.get_weights()) for ind in new_population]
            with Pool(self._threads) as pool:
                fitnesses = pool.map(self.evaluate_fitness, [ind.network.get_weights() for ind in new_population])
            print(f"Generation {g}: fitness computation, time elapsed: {time() - start_time:.2f}")

            for index in range(len(population)):
                new_population[index].fitness = fitnesses[index]

            # descending sort
            new_population.sort(key=lambda x: x.fitness)

            start_time = time()
            elite = self.get_elite(elite, new_population, elitism_evaluations)
            print(f"\rGeneration {g}: elite chosen, time elapsed: {time() - start_time:.2f}")

            try:
                new_population.remove(elite)
            except ValueError:
                new_population = new_population[1:]
            new_population = new_population + [elite]
            population = new_population

            print(f"\rGeneration {g} finished, total time {time() - generation_start_time:.2f}: elite fitness:{elite.fitness}")

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
            network = Network(self._input_shape, self._output_shape, seed=seed)

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
            import tensorflow as tf
            update = tf.random.normal(shape=w.shape, stddev=sigma)
            modified_weights.append(w + update)
        network.set_weights(modified_weights)

    def get_elite(self, elite, population, elitism_evaluations):
        # TODO roman
        # elitism
        # candidates - 10 best + last gen elite
        choose_best_count = 10
        best_from_population = population[-choose_best_count:]
        candidates = best_from_population
        if elite is not None:
            candidates.append(elite)
        # choose best candidate according to mean in -elitism_evaluations- evals
        for candidate in candidates:
            with Pool(self._threads) as pool:
                candidate_fitnesses = pool.map(self.evaluate_fitness, [ind.network.get_weights() for ind in [candidate] * elitism_evaluations])
            candidate.fitness = np.mean(candidate_fitnesses)

        new_elite = max(candidates, key=operator.attrgetter('fitness'))
        return new_elite

    def evaluate_fitness(self, network_weights):
        """
        Evaluates fitness of specified individual.
        :param individual: Individual
        :return: Fitness of the individual
        """

        network = Network(self._input_shape, self._output_shape, self._seed)
        network.set_weights(network_weights)

        gym = GymEnvironment(self._env_name)
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
        del gym
        del network
        del network_weights

        return total_reward

    # def evaluate_fitness_parallel(self, network_weights, evaluation_count):
    #     network = Network(self._input_shape, self._output_shape, self._seed)
    #     network.set_weights(network_weights)
    # 
    #     with GymEnvironment(self._env_name) as gym:
    #         states, done = gym.parallel_init(evaluation_count), False
    # 
    #         # start_time = time()
    # 
    #         step = 0
    #         equal_steps = 0
    #         all_rewards = []
    #         all_dones = []
    #         min_equal_steps = 5
    #         while not done:
    #             if self._render_each and step % self._render_each == 0:
    #                 gym.render()
    # 
    #             states = np.array(states)
    #             actions = network(states).numpy()
    #             next_states, rewards, dones, _ = zip(*gym.parallel_step(actions))
    #             if np.allclose(states, next_states):
    #                 equal_steps += 1
    #             else:
    #                 equal_steps = 0
    #             all_rewards.append(rewards)
    #             all_dones.append(dones)
    # 
    #             states = next_states
    #             step += 1
    # 
    #             done = np.all(dones)
    # 
    #             if step >= self._max_episode_len:
    #                 done = True
    #             elif equal_steps >= min_equal_steps:
    #                 done = True
    #                 # add expected reward if we waited till the episode would end
    #                 all_rewards.append((self._max_episode_len - step) * np.mean(all_rewards[-min_equal_steps:], axis=0))
    #         # print(f"Total steps {step}: {time() - start_time:.4f}")
    # 
    #         total_rewards = np.sum(all_rewards, axis=0)
    #     del network
    #     del network_weights
    # 
    #     return total_rewards