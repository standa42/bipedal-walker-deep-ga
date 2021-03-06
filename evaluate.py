import argparse

import numpy as np

from gym_evaluator import GymEnvironment
from network import Network
import tensorflow as tf
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment_name", default="BipedalWalker-v3", type=str,
                        help="Name of enviroment in gym.")
    parser.add_argument("--render_each", default=None, type=int, help="Specifies which frames are to be rendered. "
                                                                      "E.g. value is 4 means every 4th member will be rendered.")
    parser.add_argument("--nn_width", default=50, type=int, help="Size of layer of neural network")
    parser.add_argument("--weights_file", default="model.h5", type=str, help="Path to file with weights of neural network "
                                                                             "that is to be evaluated.")
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int, help="Number of iterations that will ")
    parser.add_argument("--out_video_dir", default=None, type=str, help="Specifies path to which out video is rendered. "
                                                                        "If value of this parameter is not None, "
                                                                        "render_each must also not be None.")

    args = parser.parse_args()

    assert not args.out_video_dir or args.render_each, "If out_video_dir is set, then render_each must be also set."

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    print(f"ARGS: {args}")
    print()

    gym = GymEnvironment(args.environment_name, seed=args.seed, out_video_dir=args.out_video_dir)
    input_shape = gym.state_shape
    output_shape = gym.action_shape

    network = Network(input_shape, output_shape, args.seed, nn_width=args.nn_width, initializer="zeros")
    network.load_weights(args.weights_file)

    all_returns = []
    for _ in range(args.iterations):
        state, done = gym.reset(True), False

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
        print(f"Return: {np.sum(rewards):.2f}, total steps: {step}")
        all_returns.append(np.sum(rewards))
    print(f"Avg return over {args.iterations} iterations: {np.mean(all_returns):.2f}")
