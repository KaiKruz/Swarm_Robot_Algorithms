import numpy as np
import random
from deap import base, creator, tools

# KPIs Configuration
COVERAGE_WEIGHT = 0.3
SURVIVOR_DETECTION_WEIGHT = 0.3
ENERGY_EFFICIENCY_WEIGHT = 0.2
FAULT_TOLERANCE_WEIGHT = 0.2

# Simulation Parameters
GRID_SIZE = 100  # Environment size
NUM_ROBOTS = 20  # Number of swarm robots
NUM_GENERATIONS = 100  # Number of generations
POPULATION_SIZE = 50  # Population size for GA
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.2

# Define KPIs evaluation

def evaluate(individual):
    """
    Evaluate the fitness of an individual based on KPIs.
    """
    coverage = calculate_coverage(individual)
    survivor_detection = calculate_survivor_detection(individual)
    energy_efficiency = calculate_energy_efficiency(individual)
    fault_tolerance = calculate_fault_tolerance(individual)

    fitness = (
        COVERAGE_WEIGHT * coverage
        + SURVIVOR_DETECTION_WEIGHT * survivor_detection
        + ENERGY_EFFICIENCY_WEIGHT * energy_efficiency
        + FAULT_TOLERANCE_WEIGHT * fault_tolerance
    )
    return fitness

def calculate_coverage(individual):
    """ Simulate and calculate coverage of the grid. """
    return random.uniform(0, 1)  # Placeholder for simulation result

def calculate_survivor_detection(individual):
    """ Simulate and calculate survivor detection effectiveness. """
    return random.uniform(0, 1)  # Placeholder for simulation result

def calculate_energy_efficiency(individual):
    """ Simulate and calculate energy efficiency. """
    return random.uniform(0, 1)  # Placeholder for simulation result

def calculate_fault_tolerance(individual):
    """ Simulate and calculate fault tolerance. """
    return random.uniform(0, 1)  # Placeholder for simulation result

# Define Genetic Algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Register tools
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=NUM_ROBOTS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    """ Main function to execute GA. """
    random.seed(42)

    # Initialize population
    population = toolbox.population(n=POPULATION_SIZE)

    # Statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else 0)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Evolutionary process
    for gen in range(NUM_GENERATIONS):
        print(f"Generation {gen}")

        # Evaluate fitness
        for ind in population:
            if not ind.fitness.valid:
                ind.fitness.values = (toolbox.evaluate(ind),)

        # Select and clone
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTATION_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Replace population with offspring
        population[:] = offspring

        # Gather statistics
        record = stats.compile(population)
        print(record)

    # Final population
    best_ind = tools.selBest(population, 1)[0]
    print("Best individual is", best_ind, "with fitness", best_ind.fitness.values)

if __name__ == "__main__":
    main()
