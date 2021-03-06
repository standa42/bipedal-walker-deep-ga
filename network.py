import tensorflow as tf


class Network(tf.keras.Model):
    def __init__(self, input_shape, output_shape, seed, nn_width, initializer=None):
        output_count = output_shape[0]

        input = tf.keras.layers.Input(input_shape)
        if not initializer:
            initializer = tf.keras.initializers.GlorotNormal(seed=seed)

        layer = input
        layer = tf.keras.layers.Dense(units=nn_width, activation="relu",
                                      kernel_initializer=initializer)(layer)
        layer = tf.keras.layers.Dense(units=nn_width, activation="relu",
                                      kernel_initializer=initializer)(layer)

        output = tf.keras.layers.Dense(units=output_count, activation="tanh",
                                       kernel_initializer=initializer)(layer)

        super(Network, self).__init__(inputs=input, outputs=output)
