import numpy as np
import matplotlib.pyplot as plt
import random
import os
from scipy.spatial import KDTree
from collections import deque

# Constants
ENV_WIDTH = 100
ENV_HEIGHT = 100
OBSTACLE_COUNT = 20
OBSTACLE_SIZE_RANGE = (5, 20)
START_POINTS = [(10, 10)]
GOAL_POINTS = [(90, 90)]
NUM_ANTS = 100
NUM_ITERATIONS = 500
EVAPORATION_RATE = 0.05
INITIAL_PHEROMONE = 1.0
PHEROMONE_DEPOSIT_AMOUNT = 100.0
ALPHA = 1.0  # Pheromone importance
BETA = 2.0   # Heuristic importance
SEPARATION_WEIGHT = 1.5
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 1.0
OBSTACLE_AVOIDANCE_WEIGHT = 3.0
GOAL_ATTRACTION_WEIGHT = 2.0
PERCEPTION_RADIUS = 10.0
MAX_SPEED = 2.0
MAX_FORCE = 0.1
TIME_STEP = 0.1
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
DATA_DIR = os.path.join(RESULTS_DIR, "data")

# Ensure directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Classes
class Obstacle:
    def __init__(self, position, size):
        self.position = np.array(position)
        self.size = size

class Environment:
    def __init__(self):
        self.width = ENV_WIDTH
        self.height = ENV_HEIGHT
        self.obstacles = self.generate_obstacles()
        self.obstacle_tree = KDTree([obs.position for obs in self.obstacles])

    def generate_obstacles(self):
        obstacles = []
        for _ in range(OBSTACLE_COUNT):
            position = (
                random.uniform(OBSTACLE_SIZE_RANGE[1], self.width - OBSTACLE_SIZE_RANGE[1]),
                random.uniform(OBSTACLE_SIZE_RANGE[1], self.height - OBSTACLE_SIZE_RANGE[1])
            )
            size = random.uniform(*OBSTACLE_SIZE_RANGE)
            obstacles.append(Obstacle(position, size))
        return obstacles

class Ant:
    def __init__(self, env, start_point, goal_point, pheromone_map):
        self.env = env
        self.position = np.array(start_point, dtype=float)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.goal_point = np.array(goal_point, dtype=float)
        self.pheromone_map = pheromone_map
        self.path = deque()
        self.path.append(start_point)
        self.completed = False

    def update(self, ants):
        if not self.completed:
            self.flocking(ants)
            self.apply_goal_attraction()
            self.apply_pheromone_influence()
            self.move()

    def move(self):
        if np.linalg.norm(self.acceleration) > MAX_FORCE:
            self.acceleration = (self.acceleration / np.linalg.norm(self.acceleration)) * MAX_FORCE
        self.velocity += self.acceleration
        if np.linalg.norm(self.velocity) > MAX_SPEED:
            self.velocity = (self.velocity / np.linalg.norm(self.velocity)) * MAX_SPEED
        self.position += self.velocity * TIME_STEP
        self.path.append(tuple(self.position))
        self.acceleration.fill(0)

    def flocking(self, ants):
        separation = np.zeros(2)
        alignment = np.zeros(2)
        cohesion = np.zeros(2)
        total = 0
        for other in ants:
            if other is not self:
                distance = np.linalg.norm(other.position - self.position)
                if distance < PERCEPTION_RADIUS:
                    separation += (self.position - other.position) / distance
                    alignment += other.velocity
                    cohesion += other.position
                    total += 1
        if total > 0:
            separation /= total
            alignment /= total
            cohesion = cohesion / total - self.position
            self.acceleration += SEPARATION_WEIGHT * separation
            self.acceleration += ALIGNMENT_WEIGHT * alignment
            self.acceleration += COHESION_WEIGHT * cohesion

    def apply_goal_attraction(self):
        desired = self.goal_point - self.position
        desired = (desired / np.linalg.norm(desired)) * MAX_SPEED
        steering = desired - self.velocity
        self.acceleration += GOAL_ATTRACTION_WEIGHT * steering

    def apply_pheromone_influence(self):
        grid_x, grid_y = int(self.position[0]), int(self.position[1])
        if 0 <= grid_x < self.pheromone_map.shape[0] and 0 <= grid_y < self.pheromone_map.shape[1]:
            pheromone_level = self.pheromone_map[grid_x, grid_y]
            gradient = np.zeros(2)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < self.pheromone_map.shape[0] and 0 <= ny < self.pheromone_map.shape[1]:
                        if self.pheromone_map[nx, ny] > pheromone_level:
                            gradient += np.array([dx, dy])
            if np.linalg.norm(gradient) > 0:
                gradient = (gradient / np.linalg.norm(gradient)) * MAX_SPEED
                steering = gradient - self.velocity
                self.acceleration += steering

# Visualization functions
def plot_convergence(best_lengths):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(best_lengths)), best_lengths, label="Path Length")
    plt.xlabel("Iteration")
    plt.ylabel("Best Path Length")
    plt.title("Convergence of Path Length Over Iterations")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, "convergence_graph.png"))
    plt.show()

def plot_pheromone_heatmap(pheromone_map, iteration):
    plt.figure(figsize=(10, 8))
    plt.imshow(pheromone_map, cmap="viridis", origin="lower")
    plt.colorbar(label="Pheromone Intensity")
    plt.title(f"Pheromone Heatmap at Iteration {iteration}")
    plt.savefig(os.path.join(FIGURES_DIR, f"pheromone_heatmap_iter_{iteration}.png"))
    plt.show()

def plot_behavior_snapshot(env, ants, iteration):
    plt.figure(figsize=(10, 8))
    for ant in ants:
        path_x, path_y = zip(*ant.path)
        plt.plot(path_x, path_y, alpha=0.6, linewidth=0.5)
        plt.scatter(*ant.position, color="blue", s=10)
    for obs in env.obstacles:
        circle = plt.Circle(obs.position, obs.size, color="black", alpha=0.5)
        plt.gca().add_artist(circle)
    plt.xlim(0, ENV_WIDTH)
    plt.ylim(0, ENV_HEIGHT)
    plt.title(f"Behavior Snapshot at Iteration {iteration}")
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, f"behavior_snapshot_iter_{iteration}.png"))
    plt.show()

# Simulation function
def run_aco_simulation():
    env = Environment()
    pheromone_map = np.ones((ENV_WIDTH, ENV_HEIGHT)) * INITIAL_PHEROMONE
    ants = [Ant(env, random.choice(START_POINTS), random.choice(GOAL_POINTS), pheromone_map) for _ in range(NUM_ANTS)]
    best_lengths = []

    for iteration in range(NUM_ITERATIONS):
        for ant in ants:
            ant.update(ants)
        pheromone_map *= (1 - EVAPORATION_RATE)
        for ant in ants:
            if np.linalg.norm(ant.position - ant.goal_point) < 2.0:
                ant.completed = True
        best_length = min([np.linalg.norm(ant.position - ant.goal_point) for ant in ants])
        best_lengths.append(best_length)
        if iteration % 10 == 0:
            plot_pheromone_heatmap(pheromone_map, iteration)
            plot_behavior_snapshot(env, ants, iteration)

    plot_convergence(best_lengths)

if __name__ == "__main__":
    run_aco_simulation()
