from ga import GeneticAlgorithm

if __name__ == "__main__":
    run = GeneticAlgorithm()
    run.fit(
        generation_count = 1000,
        population_size = 1000,
        sigma = 0.002,
        truncation_size = 20,
        elitism_evaluations = 30 
    )








