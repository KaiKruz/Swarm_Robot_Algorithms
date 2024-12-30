### src/environment.py

import numpy as np
import random
from scipy.spatial import KDTree

class Obstacle:
    def __init__(self, position, size):
        self.position = np.array(position)
        self.size = size

    def contains(self, point):
        return np.linalg.norm(point - self.position) <= self.size

class Environment:
    def __init__(self, width, height, obstacle_count, obstacle_size_range, survivor_count):
        self.width = width
        self.height = height
        self.obstacle_count = obstacle_count
        self.obstacle_size_range = obstacle_size_range
        self.survivor_count = survivor_count
        self.obstacles = self.generate_obstacles()
        self.survivors = self.generate_survivors()
        self.pheromone_map = np.ones((width, height)) * 1.0
        self.kd_tree = KDTree([obs.position for obs in self.obstacles])

    def generate_obstacles(self):
        obstacles = []
        for _ in range(self.obstacle_count):
            position = (
                random.uniform(self.obstacle_size_range[1], self.width - self.obstacle_size_range[1]),
                random.uniform(self.obstacle_size_range[1], self.height - self.obstacle_size_range[1])
            )
            size = random.uniform(*self.obstacle_size_range)
            obstacles.append(Obstacle(position, size))
        return obstacles

    def generate_survivors(self):
        survivors = []
        for _ in range(self.survivor_count):
            position = (
                random.uniform(0, self.width),
                random.uniform(0, self.height)
            )
            survivors.append(np.array(position))
        return survivors

    def is_collision(self, point):
        distances, _ = self.kd_tree.query([point], k=1)
        nearest_obstacle = distances[0] if distances else float('inf')
        return nearest_obstacle <= self.obstacle_size_range[1]