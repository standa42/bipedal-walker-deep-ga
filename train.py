import numpy as np
import tensorflow as tf
import random

from ga import GeneticAlgorithm
import argparse
import multiprocessing

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "10"

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    parser.add_argument("--enviroment_name", default="BipedalWalker-v3", type=str,
                        help="Name of enviroment in gym")
    parser.add_argument("--max_episode_length", default=400, type=int,
                        help="Maximal length of episode")
    parser.add_argument("--generations_count", default=1000, type=float,
                        help="Number of generations for fit.")
    parser.add_argument("--population_size", default=100
    , type=int,
                        help="Size of the population.")
    parser.add_argument("--sigma", default=0.002, type=float)
    parser.add_argument("--truncation_size", default=20, type=int)

    parser.add_argument("--elitism_evaluations", default=10, type=int)
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    run = GeneticAlgorithm(args.threads, args.enviroment_name, args.max_episode_length)
    run.fit(
        generation_count=args.generations_count,
        population_size=args.population_size,
        sigma=args.sigma,
        truncation_size=args.truncation_size,
        elitism_evaluations=args.elitism_evaluations
    )








