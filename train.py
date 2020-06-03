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
    parser.add_argument("--max_episode_length", default=1000, type=int,
                        help="Maximal length of episode")
    parser.add_argument("--generations_count", default=1000, type=int,
                        help="Number of generations for fit.")
    parser.add_argument("--population_size", default=500, type=int, help="Size of the population.")
    parser.add_argument("--sigma", default=0.002, type=float)
    parser.add_argument("--truncation_size", default=20, type=int)
    parser.add_argument("--nn_width", default=50, type=int, help="Size of layer of neural network")

    parser.add_argument("--elitism_evaluations", default=10, type=int)
    parser.add_argument("--elite_choose_best_count", default=10, type=int)
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--render_each", default=None, type=int)
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

    run = GeneticAlgorithm(args.threads, args.environment_name, args.max_episode_length, args.elite_choose_best_count,
                           args.render_each, args.logdir, args.nn_width, args.seed)
    run.fit(
        generation_count=args.generations_count,
        population_size=args.population_size,
        sigma=args.sigma,
        truncation_size=args.truncation_size,
        elitism_evaluations=args.elitism_evaluations
    )








