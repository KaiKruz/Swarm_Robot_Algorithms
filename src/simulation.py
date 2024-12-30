### src/simulation.py

import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from agent import Agent

class Simulation:
    def __init__(self):
        self.environment = Environment(100, 100, 20, (5, 15), 10)
        self.agents = [Agent(self.environment, (np.random.uniform(0, 100), np.random.uniform(0, 100))) for _ in range(50)]

    def run(self, iterations=200):
        stats = []
        for iteration in range(iterations):
            for agent in self.agents:
                agent.update()
            self.environment.pheromone_map *= 0.95  # Evaporation
            if iteration % 20 == 0:
                detected = sum(len(agent.detected_survivors) for agent in self.agents)
                stats.append((iteration, detected))
                self.visualize(iteration)
        return stats

    def visualize(self, iteration):
        plt.figure(figsize=(12, 12))
        plt.imshow(self.environment.pheromone_map.T, origin='lower', cmap='hot', alpha=0.7)
        for survivor in self.environment.survivors:
            plt.scatter(survivor[0], survivor[1], color='green', s=100, label='Survivor')
        for obs in self.environment.obstacles:
            circle = plt.Circle(obs.position, obs.size, color="black", alpha=0.6)
            plt.gca().add_artist(circle)
        for agent in self.agents:
            plt.scatter(agent.position[0], agent.position[1], color='blue', s=20, label='Agent')
        plt.title(f"Iteration {iteration}")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.grid()
        plt.legend(loc="upper right")
        plt.show()

if __name__ == "__main__":
    simulation = Simulation()
    stats = simulation.run()
    print("Final Statistics:", stats)
