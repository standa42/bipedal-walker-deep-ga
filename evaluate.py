import argparse

import numpy as np

from gym_evaluator import GymEnvironment
from network import Network

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment_name", default="BipedalWalker-v3", type=str,
                        help="Name of enviroment in gym")
    parser.add_argument("--render_each", default=1, type=int)
    parser.add_argument("--nn_width", default=50, type=int, help="Size of layer of neural network")
    parser.add_argument("--weights_file", default="model.h5", type=str)
    parser.add_argument("--seed", default=12, type=int)

    args = parser.parse_args()

    print(f"ARGS: {args}")
    print()

    gym = GymEnvironment(args.environment_name, seed=args.seed)
    input_shape = gym.state_shape
    output_shape = gym.action_shape


    network = Network(input_shape, output_shape, args.seed, nn_width=args.nn_width, initializer="zeros")
    network.load_weights(args.weights_file)

    state, done = gym.reset(), False

    step = 0
    rewards = []
    while not done:
        if args.render_each and step % args.render_each == 0:
            gym.render()

        state = np.expand_dims(state, 0)
        action = network(state).numpy()[0]
        next_state, reward, done, _ = gym.step(action)
        rewards.append(reward)

        state = next_state
        step += 1
    print(f"Rewards: {np.sum(rewards):.2f}")
