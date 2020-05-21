import tensorflow as tf


class Individual:
    def __init__(self, network, fitness: float = 0):
        self.network = network
        self.fitness = fitness

    def clone(self):
        new_individual = Individual(tf.keras.models.clone_model(self.network), self.fitness)
        return new_individual
