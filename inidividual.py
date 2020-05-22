import tensorflow as tf


class Individual:
    def __init__(self, network, fitness: float = None):
        self.network = network
        self.fitness = fitness

    def clone(self):
        new_individual = Individual(tf.keras.models.clone_model(self.network), self.fitness)
        new_individual.network.set_weights(self.network.get_weights())
        return new_individual

    def __str__(self):
        return f"{self.fitness:.4f}"

    def __repr__(self):
        if self.fitness:
            return f"{self.fitness:.4f}"

        return "None"
