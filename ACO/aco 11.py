import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
from scipy.spatial import KDTree
from collections import deque
import time
import csv
import os

# Constants
ENV_WIDTH = 100
ENV_HEIGHT = 100
OBSTACLE_COUNT = 20
OBSTACLE_SIZE_RANGE = (5, 15)
START_POINTS = [(10, 10)]
GOAL_POINTS = [(90, 90)]
NUM_ANTS = 50
NUM_ITERATIONS = 200
EVAPORATION_RATE = 0.1
INITIAL_PHEROMONE = 1.0
ALPHA = 1.0  # Pheromone importance
BETA = 2.0   # Distance importance
Q = 100.0    # Pheromone deposit factor
SEPARATION_WEIGHT = 1.5
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 1.0
OBSTACLE_AVOIDANCE_WEIGHT = 3.0
GOAL_ATTRACTION_WEIGHT = 2.0
PERCEPTION_RADIUS = 10.0
MAX_SPEED = 2.0
MAX_FORCE = 0.1
TIME_STEP = 0.1
MAX_TIME = 100.0
VISUALIZATION_INTERVAL = 10  # Visualize every 10 iterations

# Environment and Visualization Classes
class Obstacle:
    """Represent a circular obstacle."""
    def __init__(self, position, size):
        self.position = np.array(position)
        self.size = size

    def contains(self, point):
        """Check if a point is within the obstacle."""
        return np.linalg.norm(point - self.position) <= self.size

class Environment:
    """Represent the simulation environment."""
    def __init__(self):
        self.width = ENV_WIDTH
        self.height = ENV_HEIGHT
        self.obstacles = self.generate_obstacles()
        self.obstacle_tree = KDTree([obs.position for obs in self.obstacles])

    def generate_obstacles(self):
        """Generate random obstacles."""
        obstacles = []
        for _ in range(OBSTACLE_COUNT):
            position = (
                random.uniform(OBSTACLE_SIZE_RANGE[1], self.width - OBSTACLE_SIZE_RANGE[1]),
                random.uniform(OBSTACLE_SIZE_RANGE[1], self.height - OBSTACLE_SIZE_RANGE[1])
            )
            size = random.uniform(*OBSTACLE_SIZE_RANGE)
            obstacles.append(Obstacle(position, size))
        return obstacles

    def is_collision(self, point):
        """Check if a point collides with any obstacle."""
        for obs in self.obstacles:
            if obs.contains(point):
                return True
        return False

    def get_nearby_obstacles(self, point):
        """Retrieve obstacles within a radius."""
        indices = self.obstacle_tree.query_ball_point(point, PERCEPTION_RADIUS)
        return [self.obstacles[i] for i in indices]

class Ant:
    """Represent an agent with ACO and flocking behavior."""
    def __init__(self, env, start_point, goal_point, pheromone_map):
        self.env = env
        self.position = np.array(start_point, dtype=float)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.goal_point = np.array(goal_point, dtype=float)
        self.pheromone_map = pheromone_map
        self.path = deque()
        self.path.append(start_point)
        self.visited = set()
        self.completed = False

    def update(self, ants):
        """Update the agent's position and interactions."""
        if not self.completed:
            self.flocking(ants)
            self.apply_goal_attraction()
            self.apply_obstacle_avoidance()
            self.move()

    def move(self):
        """Update the position based on velocity and acceleration."""
        if np.linalg.norm(self.acceleration) > MAX_FORCE:
            self.acceleration = (self.acceleration / np.linalg.norm(self.acceleration)) * MAX_FORCE
        self.velocity += self.acceleration
        if np.linalg.norm(self.velocity) > MAX_SPEED:
            self.velocity = (self.velocity / np.linalg.norm(self.velocity)) * MAX_SPEED
        new_position = self.position + self.velocity * TIME_STEP
        if not self.env.is_collision(new_position):
            self.position = new_position
            self.path.append(tuple(self.position))
            self.visited.add(tuple(self.position))
        self.acceleration.fill(0)

    def flocking(self, ants):
        """Apply flocking dynamics."""
        separation = np.zeros(2)
        alignment = np.zeros(2)
        cohesion = np.zeros(2)
        total = 0
        for other in ants:
            if other is not self:
                distance = np.linalg.norm(other.position - self.position)
                if distance < PERCEPTION_RADIUS:
                    diff = self.position - other.position
                    diff /= max(distance, 1e-5)
                    separation += diff
                    alignment += other.velocity
                    cohesion += other.position
                    total += 1
        if total > 0:
            separation /= total
            alignment /= total
            cohesion /= total
            cohesion = cohesion - self.position
            self.acceleration += SEPARATION_WEIGHT * separation
            self.acceleration += ALIGNMENT_WEIGHT * alignment
            self.acceleration += COHESION_WEIGHT * cohesion

    def apply_goal_attraction(self):
        """Attract towards the goal."""
        desired = self.goal_point - self.position
        desired = (desired / np.linalg.norm(desired)) * MAX_SPEED
        steering = desired - self.velocity
        self.acceleration += GOAL_ATTRACTION_WEIGHT * steering

    def apply_obstacle_avoidance(self):
        """Avoid nearby obstacles."""
        for obs in self.env.get_nearby_obstacles(self.position):
            diff = self.position - obs.position
            distance = np.linalg.norm(diff)
            if distance < obs.size + PERCEPTION_RADIUS:
                diff /= max(distance, 1e-5)
                self.acceleration += OBSTACLE_AVOIDANCE_WEIGHT * diff / distance

def evaporate_pheromones(pheromone_map):
    """Evaporate pheromones over time."""
    pheromone_map *= (1 - EVAPORATION_RATE)

def run_aco_simulation():
    """Run the complete ACO and Flocking simulation."""
    env = Environment()
    pheromone_map = np.zeros((ENV_WIDTH, ENV_HEIGHT))
    ants = [Ant(env, random.choice(START_POINTS), random.choice(GOAL_POINTS), pheromone_map) for _ in range(NUM_ANTS)]

    for iteration in range(NUM_ITERATIONS):
        evaporate_pheromones(pheromone_map)
        for ant in ants:
            ant.update(ants)
            if np.linalg.norm(ant.position - ant.goal_point) < 2.0:
                ant.completed = True
        if iteration % VISUALIZATION_INTERVAL == 0:
            visualize_environment(env, pheromone_map, ants, iteration)

def visualize_environment(env, pheromone_map, ants, iteration):
    """Visualize the simulation."""
    plt.figure(figsize=(10, 10))
    plt.imshow(pheromone_map, cmap="viridis", origin="lower", alpha=0.6)
    for ant in ants:
        plt.scatter(*ant.position, color="blue", s=10)
        path_x, path_y = zip(*ant.path)
        plt.plot(path_x, path_y, alpha=0.6, linewidth=0.5)
    for obs in env.obstacles:
        circle = plt.Circle(obs.position, obs.size, color="black", alpha=0.5)
        plt.gca().add_artist(circle)
    plt.title(f"Iteration {iteration}")
    plt.xlim(0, ENV_WIDTH)
    plt.ylim(0, ENV_HEIGHT)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    run_aco_simulation()
