import random  # Added missing import
from deap import tools  # Added missing import
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from algorithms.ga_algorithm import toolbox, NUM_GENERATIONS, POPULATION_SIZE

def run_simulation():
    population = toolbox.population(n=POPULATION_SIZE)

    for gen in range(NUM_GENERATIONS):
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind),
        
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        population[:] = offspring

    best_ind = tools.selBest(population, 1)[0]
    print(f"Simulation complete. Best individual: {best_ind}")

if __name__ == "__main__":
    run_simulation()