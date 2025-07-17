import unittest
import sys
import os

# Adjust the path to include the algorithms folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from algorithms.ga_algorithm import toolbox

class TestGA(unittest.TestCase):
    def test_population_initialization(self):
        population = toolbox.population(n=10)
        self.assertEqual(len(population), 10)

    def test_individual_fitness(self):
        ind = toolbox.individual()
        fitness = toolbox.evaluate(ind)
        self.assertTrue(isinstance(fitness, float) or isinstance(fitness, int))

if __name__ == "__main__":
    unittest.main()