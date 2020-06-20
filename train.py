import uuid

import numpy as np
import tensorflow as tf
import random

from ga import GeneticAlgorithm
import argparse
import multiprocessing
from datetime import datetime

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "10"

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    parser.add_argument("--environment_name", default="BipedalWalker-v3", type=str,
                        help="Name of enviroment in gym")
    parser.add_argument("--max_episode_length", default=1600, type=int,
                        help="Maximal length of episode. Specifies number of steps after which evaluation (simulation) "
                             "will be cut off. In BipedalWalker maximum episode length is 1600.")
    parser.add_argument("--generations_count", default=2000, type=int,
                        help="Number of generations evolutionary algorithm will train.")
    parser.add_argument("--population_size", default=250, type=int, help="Size of the population.")
    parser.add_argument("--sigma", default=0.01, type=float, help="Sigma of normal distribution that is used to perform mutation.")
    parser.add_argument("--sigma_final", default=None, type=float)
    parser.add_argument("--truncation_size", default=20, type=int, help="Number of top individuals (by fitness) "
                                                                        "that are selected as parents and from which "
                                                                        "new population is generated.")
    parser.add_argument("--nn_width", default=50, type=int, help="Width of layers of neural network")

    parser.add_argument("--elitism_evaluations", default=12, type=int, help="How many times each elite candidate is evaluated. "
                                                                            "Larger number means more accurate estimation "
                                                                            "of his performance.")
    parser.add_argument("--elite_choose_best_count", default=10, type=int, help="Number of elite candidates from which "
                                                                                "elite individual is selected.")
    parser.add_argument("--threads", default=1, type=int, help="Number of threads used for the training.")
    parser.add_argument("--render_each", default=None, type=int)
    parser.add_argument("--min_equal_steps", default=5, type=int, help="Specifies number of equal states in evaluation "
                                                                        "after which it will be cut off and estimated.")
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()
    args.logdir = os.path.join("logs", f"train_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4()}")

    print(f"ARGS: {args}")
    print()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    run = GeneticAlgorithm(threads=args.threads, env_name=args.environment_name, max_episode_len=args.max_episode_length,
                           min_equal_steps=args.min_equal_steps, elite_choose_best_count=args.elite_choose_best_count,
                           render_each=args.render_each, logdir=args.logdir, nn_width=args.nn_width, seed=args.seed)
    run.fit(
        generation_count=args.generations_count,
        population_size=args.population_size,
        sigma=args.sigma,
        truncation_size=args.truncation_size,
        elitism_evaluations=args.elitism_evaluations,
        sigma_final=args.sigma_final
    )








