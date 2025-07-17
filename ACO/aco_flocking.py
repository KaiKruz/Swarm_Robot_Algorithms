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
from datetime import datetime

# Environment Parameters
ENV_WIDTH = 100
ENV_HEIGHT = 100
OBSTACLE_COUNT = 15
OBSTACLE_SIZE_RANGE = (5, 15)
START_POINTS = [(10, 10)]
GOAL_POINTS = [(90, 90)]

# ACO Parameters
NUM_ANTS = 50
NUM_ITERATIONS = 200
EVAPORATION_RATE = 0.1
INITIAL_PHEROMONE = 1.0
ALPHA = 1.0  # Pheromone importance
BETA = 2.0   # Heuristic importance
Q = 100.0    # Pheromone deposit factor

# Flocking Parameters
SEPARATION_WEIGHT = 1.5
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 1.0
OBSTACLE_AVOIDANCE_WEIGHT = 3.0
GOAL_ATTRACTION_WEIGHT = 2.0
PERCEPTION_RADIUS = 10.0
MAX_SPEED = 2.0
MAX_FORCE = 0.1

# Simulation Parameters
TIME_STEP = 0.1
MAX_TIME = 100.0

RESULTS_DIR = "results"
DATA_DIR = os.path.join(RESULTS_DIR, "data")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

def ensure_directories_exist():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

class Obstacle:
    def __init__(self, position, size):
        self.position = np.array(position)
        self.size = size  # Radius for circular obstacles

    def contains(self, point):
        return np.linalg.norm(point - self.position) <= self.size

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

    def is_collision(self, point):
        for obs in self.obstacles:
            if obs.contains(point):
                return True
        return False

    def get_nearby_obstacles(self, point):
        indices = self.obstacle_tree.query_ball_point(point, PERCEPTION_RADIUS)
        return [self.obstacles[i] for i in indices]

class Ant:
    def __init__(self, env, start_point, goal_point, pheromone_map):
        self.env = env
        self.position = np.array(start_point, dtype=float)
        self.velocity = (np.random.rand(2) - 0.5) * MAX_SPEED
        self.acceleration = np.zeros(2)
        self.goal_point = np.array(goal_point, dtype=float)
        self.pheromone_map = pheromone_map
        self.path = deque()
        self.path.appendleft(tuple(self.position))
        self.visited = set()
        self.visited.add(tuple(self.position))
        self.completed = False

    def update(self, ants):
        if not self.completed:
            self.flocking(ants)
            self.apply_pheromone_attraction()
            self.apply_goal_attraction()
            self.apply_obstacle_avoidance()
            self.move()
            if np.linalg.norm(self.position - self.goal_point) < 2.0:
                self.completed = True

    def move(self):
        if np.linalg.norm(self.acceleration) > MAX_FORCE:
            self.acceleration = (self.acceleration / np.linalg.norm(self.acceleration)) * MAX_FORCE
        self.velocity += self.acceleration
        if np.linalg.norm(self.velocity) > MAX_SPEED:
            self.velocity = (self.velocity / np.linalg.norm(self.velocity)) * MAX_SPEED
        new_position = self.position + self.velocity * TIME_STEP
        if not self.env.is_collision(new_position):
            self.position = new_position
            self.path.appendleft(tuple(self.position))
            self.visited.add(tuple(self.position))
        else:
            self.velocity = -self.velocity * 0.5
        self.acceleration = np.zeros(2)

    def apply_force(self, force):
        self.acceleration += force

    def flocking(self, ants):
        separation, alignment, cohesion = np.zeros(2), np.zeros(2), np.zeros(2)
        total = 0
        for other in ants:
            if other is not self:
                distance = np.linalg.norm(other.position - self.position)
                if 0 < distance < PERCEPTION_RADIUS:
                    separation += (self.position - other.position) / distance
                    alignment += other.velocity
                    cohesion += other.position
                    total += 1
        if total > 0:
            separation = self.normalize(separation / total) * MAX_SPEED - self.velocity
            alignment = self.normalize(alignment / total) * MAX_SPEED - self.velocity
            cohesion = self.normalize((cohesion / total) - self.position) * MAX_SPEED - self.velocity
            self.apply_force(SEPARATION_WEIGHT * separation)
            self.apply_force(ALIGNMENT_WEIGHT * alignment)
            self.apply_force(COHESION_WEIGHT * cohesion)

    def apply_obstacle_avoidance(self):
        obstacles = self.env.get_nearby_obstacles(self.position)
        avoidance = np.zeros(2)
        for obs in obstacles:
            diff = self.position - obs.position
            distance = np.linalg.norm(diff)
            if distance < obs.size + PERCEPTION_RADIUS and distance > 0:
                diff /= distance
                avoidance += diff / (distance ** 2)
        if np.linalg.norm(avoidance) > 0:
            avoidance = self.normalize(avoidance) * MAX_SPEED - self.velocity
            self.apply_force(OBSTACLE_AVOIDANCE_WEIGHT * avoidance)

    def apply_goal_attraction(self):
        desired = self.normalize(self.goal_point - self.position) * MAX_SPEED
        steering = desired - self.velocity
        self.apply_force(GOAL_ATTRACTION_WEIGHT * steering)

    def apply_pheromone_attraction(self):
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
                gradient = self.normalize(gradient) * MAX_SPEED - self.velocity
                self.apply_force(gradient)

    @staticmethod
    def normalize(vector):
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

# Helper Functions
def calculate_path_length(path):
    length = 0.0
    prev_point = None
    for point in reversed(path):
        if prev_point is not None:
            length += np.linalg.norm(np.array(point) - np.array(prev_point))
        prev_point = point
    return length

def deposit_pheromones(pheromone_map, path, amount):
    for point in path:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < pheromone_map.shape[0] and 0 <= y < pheromone_map.shape[1]:
            pheromone_map[x, y] += amount

def run_aco_flocking_simulation():
    ensure_directories_exist()
    env = Environment()
    pheromone_map = np.ones((ENV_WIDTH, ENV_HEIGHT)) * INITIAL_PHEROMONE
    ants = [Ant(env, random.choice(START_POINTS), random.choice(GOAL_POINTS), pheromone_map) for _ in range(NUM_ANTS)]
    best_path, best_path_length = None, np.inf

    for iteration in range(NUM_ITERATIONS):
        for ant in ants:
            ant.update(ants)
        pheromone_map *= (1 - EVAPORATION_RATE)
        for ant in ants:
            if ant.completed:
                path_length = calculate_path_length(ant.path)
                if path_length < best_path_length:
                    best_path, best_path_length = list(ant.path), path_length
                deposit_pheromones(pheromone_map, ant.path, Q / path_length)
        print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Best path length: {best_path_length:.2f}")
        if best_path_length <= np.linalg.norm(np.array(START_POINTS[0]) - np.array(GOAL_POINTS[0])) * 1.1:
            print("Optimal path found.")
            break

    result_file = os.path.join(DATA_DIR, "simulation_results.csv")
    with open(result_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Best Path Length", "Iterations"])
        writer.writerow([best_path_length, NUM_ITERATIONS])

    return env, pheromone_map, best_path, ants

if __name__ == "__main__":
    run_aco_flocking_simulation()
