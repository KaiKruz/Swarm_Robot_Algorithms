### src/agent.py

import numpy as np
from collections import deque

class Agent:
    def __init__(self, environment, start_pos, energy_limit=500):
        self.environment = environment
        self.position = np.array(start_pos, dtype=float)
        self.energy = energy_limit
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
        # Intelligent movement using pheromone and survivor attraction
        direction = self.get_pheromone_gradient() + self.get_survivor_attraction()
        direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else (np.random.rand(2) - 0.5)
        new_position = self.position + direction
        if (0 <= new_position[0] < self.environment.width and
                0 <= new_position[1] < self.environment.height and
                not self.environment.is_collision(new_position)):
            self.position = new_position
            self.energy -= 1

    def check_survivors(self):
        for survivor in self.environment.survivors:
            if np.linalg.norm(self.position - survivor) < 10.0:
                self.detected_survivors.add(tuple(survivor))

    def leave_pheromone(self):
        x, y = int(self.position[0]), int(self.position[1])
        if 0 <= x < self.environment.width and 0 <= y < self.environment.height:
            self.environment.pheromone_map[x, y] += 1

    def get_pheromone_gradient(self):
        x, y = int(self.position[0]), int(self.position[1])
        gradient = np.zeros(2)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.environment.width and 0 <= ny < self.environment.height:
                    pheromone_level = self.environment.pheromone_map[nx, ny]
                    distance = np.sqrt(dx**2 + dy**2)
                    gradient += (pheromone_level / max(distance, 1e-5)) * np.array([dx, dy])
        return gradient

    def get_survivor_attraction(self):
        attraction = np.zeros(2)
        for survivor in self.environment.survivors:
            distance = np.linalg.norm(self.position - survivor)
            if distance > 0:
                attraction += (survivor - self.position) / (distance**2)
        return attraction

    def get_statistics(self):
        return len(self.detected_survivors), self.energy