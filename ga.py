import random

from inidividual import Individual
from network import Network
import tensorflow as tf

class GeneticAlgorithm:
    
    def __init__(self, threads):
        self._input_shape = (10, )
        self._outputs = 4
        self._threads = threads

    def fit(self, generation_count, population_size, sigma, truncation_size, elitism_evaluations):
        """main ga cycle"""
        population = self.init_population(population_size)

        for g in range(generation_count):
            print(f"Generation {g} started \r")
            new_population = []

            # TODO roman
            # paralelize
            parents = population[:truncation_size]
            for _ in range(population_size):
                offspring = self.generate_offspring(parents, sigma)
                new_population.append(offspring)

            self.evaluate_pop_fitness(offspring)


            # descending sort
            new_population.sort(key=lambda x: x.fitness, reverse=True)

            elite = self.get_elite(elite, new_population, elitism_evaluations)

            # append elite
            new_population.remove(elite)
            new_population = [elite] + new_population

            print(f"Generation {g} has elite fitness: {elite.fitness}")

    def generate_offspring(self, parents, sigma):
        chosen_parent: Individual = random.choice(parents)
        offspring = chosen_parent.clone()
        self.mutate(offspring, sigma)
        return offspring


    def init_population(self, population_size):
        max_int = 2 ** 63 - 1
        population = []
        for _ in range(population_size):
            seed = random.randint(0, max_int)
            network = Network(self._input_shape, self._outputs, seed=seed)

            ind = Individual(network)
            population.append(ind)
        return population

    def evaluate_pop_fitness(self, population):
        """eval network"""
        # TODO roman
        return 42

    def mutate(self, individial, sigma):
        """generate offspring"""
        # TODO honza
        network = individial.network
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
        candidates = best_from_population + [elite]
        # choose best candidate according to mean in -elitism_evaluations- evals
        from statistics import mean
        for candidate in candidates:
            candidate_fitnesses = []
            for _ in elitism_evaluations:
                # TODO: evaluate n-times
                pass
            candidate.fitness = mean(candidate_fitnesses)

        import operator
        new_elite = max(candidates, key=operator.attrgetter('fitness'))

