import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import KDTree
from collections import deque
import os
import csv

# Constants for Simulation
ENV_WIDTH = 100
ENV_HEIGHT = 100
NUM_AGENTS = 50
OBSTACLE_COUNT = 15
OBSTACLE_SIZE_RANGE = (5, 15)
SURVIVOR_COUNT = 5
EVAPORATION_RATE = 0.1
INITIAL_PHEROMONE = 1.0
ALPHA = 1.0  # Pheromone influence
BETA = 2.0   # Distance influence
ENERGY_LIMIT = 500  # Energy units per agent
PERCEPTION_RADIUS = 10.0
MAX_SPEED = 2.0

# Directories for results
RESULTS_DIR = "results"
DATA_DIR = os.path.join(RESULTS_DIR, "data")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

class Obstacle:
    def __init__(self, position, size):
        self.position = np.array(position)
        self.size = size

    def contains(self, point):
        return np.linalg.norm(point - self.position) <= self.size

class Environment:
    def __init__(self):
        self.width = ENV_WIDTH
        self.height = ENV_HEIGHT
        self.obstacles = self.generate_obstacles()
        self.survivors = self.generate_survivors()
        self.pheromone_map = np.ones((ENV_WIDTH, ENV_HEIGHT)) * INITIAL_PHEROMONE

    def generate_obstacles(self):
        obstacles = []
        for _ in range(OBSTACLE_COUNT):
            position = (
                random.uniform(OBSTACLE_SIZE_RANGE[1], ENV_WIDTH - OBSTACLE_SIZE_RANGE[1]),
                random.uniform(OBSTACLE_SIZE_RANGE[1], ENV_HEIGHT - OBSTACLE_SIZE_RANGE[1])
            )
            size = random.uniform(*OBSTACLE_SIZE_RANGE)
            obstacles.append(Obstacle(position, size))
        return obstacles

    def generate_survivors(self):
        survivors = []
        for _ in range(SURVIVOR_COUNT):
            position = (
                random.uniform(0, ENV_WIDTH),
                random.uniform(0, ENV_HEIGHT)
            )
            survivors.append(np.array(position))
        return survivors

    def is_collision(self, point):
        for obs in self.obstacles:
            if obs.contains(point):
                return True
        return False

class Agent:
    def __init__(self, environment, start_pos):
        self.environment = environment
        self.position = np.array(start_pos, dtype=float)
        self.energy = ENERGY_LIMIT
        self.path = deque()
        self.detected_survivors = set()
        self.completed = False

    def update(self):
        if self.energy <= 0 or self.completed:
            return
        self.move()
        self.check_survivors()
        self.leave_pheromone()

    def move(self):
        random_direction = (np.random.rand(2) - 0.5) * MAX_SPEED
        new_position = self.position + random_direction
        if (0 <= new_position[0] < ENV_WIDTH and
                0 <= new_position[1] < ENV_HEIGHT and
                not self.environment.is_collision(new_position)):
            self.position = new_position
            self.energy -= 1

    def check_survivors(self):
        for survivor in self.environment.survivors:
            if np.linalg.norm(self.position - survivor) < PERCEPTION_RADIUS:
                self.detected_survivors.add(tuple(survivor))

    def leave_pheromone(self):
        x, y = int(self.position[0]), int(self.position[1])
        if 0 <= x < ENV_WIDTH and 0 <= y < ENV_HEIGHT:
            self.environment.pheromone_map[x, y] += 1

    def get_statistics(self):
        return len(self.detected_survivors), self.energy

# Simulation Logic
def run_simulation():
    env = Environment()
    agents = [Agent(env, (random.uniform(0, ENV_WIDTH), random.uniform(0, ENV_HEIGHT))) for _ in range(NUM_AGENTS)]

    for iteration in range(100):  # Run for 100 iterations
        for agent in agents:
            agent.update()
        env.pheromone_map *= (1 - EVAPORATION_RATE)  # Pheromone evaporation

    # Collect Results
    total_detected = sum(len(agent.detected_survivors) for agent in agents)
    avg_energy_remaining = np.mean([agent.energy for agent in agents])
    print(f"Survivors detected: {total_detected}/{SURVIVOR_COUNT}")
    print(f"Average energy remaining: {avg_energy_remaining:.2f}")

    # Save Results
    with open(os.path.join(DATA_DIR, "results.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Survivors Detected", "Average Energy Remaining"])
        writer.writerow([total_detected, avg_energy_remaining])

    visualize_results(env, agents)

def visualize_results(env, agents):
    plt.figure(figsize=(10, 10))
    plt.imshow(env.pheromone_map.T, origin='lower', cmap='viridis', alpha=0.6)
    for survivor in env.survivors:
        plt.scatter(survivor[0], survivor[1], color='red', s=50, label='Survivor')
    for obs in env.obstacles:
        circle = plt.Circle(obs.position, obs.size, color="black", alpha=0.5)
        plt.gca().add_artist(circle)
    for agent in agents:
        plt.scatter(agent.position[0], agent.position[1], color='blue', s=10, label='Agent')
    plt.title("Survivor Detection and Pheromone Trails")
    plt.xlim(0, ENV_WIDTH)
    plt.ylim(0, ENV_HEIGHT)
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    run_simulation()
