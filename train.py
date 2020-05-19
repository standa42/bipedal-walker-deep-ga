import numpy as np
import tensorflow as tf
import random

from ga import GeneticAlgorithm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--generations_count", default=1000, type=float,
                        help="Number of generations for fit.")
    parser.add_argument("--population_size", default=1000, type=int,
                        help="Size of the population.")
    parser.add_argument("--sigma", default=0.002, type=float)
    parser.add_argument("--truncation_size", default=20, type=int)
    parser.add_argument("--elitism_evaluations", default=30, type=int)
    parser.add_argument("--threads", default=42, type=int)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    run = GeneticAlgorithm(args.threads)
    run.fit(
        generation_count=args.generations_count,
        population_size=args.population_size,
        sigma=args.sigma,
        truncation_size=args.truncation_size,
        elitism_evaluations=args.elitism_evaluations
    )








