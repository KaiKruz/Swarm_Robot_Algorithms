import random
from deap import base, creator, tools

# Define Genetic Algorithm Parameters
NUM_GENERATIONS = 100
POPULATION_SIZE = 50
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.2

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=20)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", lambda ind: sum(ind))
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

if __name__ == "__main__":
    population = toolbox.population(n=POPULATION_SIZE)

    for gen in range(NUM_GENERATIONS):
        # Evaluate fitness
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind),

        # Select next generation
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

        # Replace population
        population[:] = offspring

    best_ind = tools.selBest(population, 1)[0]
    print(f"Best individual: {best_ind}, Fitness: {best_ind.fitness.values[0]}")

