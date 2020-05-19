import tensorflow as tf


class Network(tf.keras.Model):
    def __init__(self, input_shape, outputs_count, seed):
        input = tf.keras.layers.Input(input_shape)
        initializer = tf.keras.initializers.GlorotNormal(seed=seed)

        layer = input
        layer = tf.keras.layers.Dense(units=100, activation="relu",
                                      kernel_initializer=initializer)(layer)
        layer = tf.keras.layers.Dense(units=100, activation="relu",
                                      kernel_initializer=initializer)(layer)

        output = tf.keras.layers.Dense(units=outputs_count, activation="tanh",
                                       kernel_initializer=initializer)(layer)

        super(Network, self).__init__(inputs=input, outputs=output)

    def clone(self):
        new_network = tf.keras.models.clone_model(self)
        new_network.set_weights(self.get_weights())
        return new_network
