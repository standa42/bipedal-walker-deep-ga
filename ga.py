from inidividual import Individual
from network import Network

class GeneticAlgorithm:
    
    def __init__(self):
        pass

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
                self.generate_offspring(parents, sigma)

            # TODO roman
            # descending sort

            elite = self.get_elite(elite, new_population, elitism_evaluations)

            # append elite
            new_population.remove(elite)
            new_population = [elite] + new_population

            print(f"Generation {g} has elite fitness: {elite.fitness}")

    def generate_offspring(self, parents, sigma):
        chosen_parent = None # TODO: choose one parent  # roman
        offspring = Individual(chosen_parent)
        offspring = self.mutate(offspring)
        offspring.fitness = self.evaluate_fitness(offspring)
        return offspring


    def init_population(self, population_size):
        # TODO honza
        return [None] * population_size

    def evaluate_fitness(self, individual):
        """eval network"""
        # TODO roman
        pass

    def mutate(self, individial, sigma):
        """generate offspring"""
        # TODO honza
        pass

    def get_elite(self, elite, new_population, elitism_evaluations):
        # TODO roman
        # elitism
        # candidates - 10 best + last gen elite
        # choose best candidate according to mean in -elitism_evaluations- evals
        pass

    









