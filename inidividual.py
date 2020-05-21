import tensorflow as tf


class Individual:
    def __init__(self, network, fitness: float = None):
        self.network = network
        self.fitness = fitness

    def clone(self):
        new_individual = Individual(tf.keras.models.clone_model(self.network), self.fitness)
        new_individual.network.set_weights(self.network.get_weights())
        assert (new_individual.network.get_weights()[1] == self.network.get_weights()[1]).all()
        return new_individual
