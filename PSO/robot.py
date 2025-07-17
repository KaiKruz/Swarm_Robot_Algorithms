import numpy as np

class Robot:
    def __init__(self, id, position, env):
        self.id = id
        self.position = position
        self.velocity = np.zeros(2)
        self.pbest_position = position.copy()
        self.pbest_value = -float('inf')
        self.env = env
        self.energy = 1000.0  # Initial energy
        self.active = True
        self.survivors_detected = 0

        # PSO parameters
        self.w = 0.5
        self.c1 = 1.0
        self.c2 = 1.0

    def update(self):
        if not self.active:
            return

        # Update personal best
        fitness = self.compute_fitness()
        if fitness > self.pbest_value:
            self.pbest_value = fitness
            self.pbest_position = self.position.copy()

        # Update global best in the environment
        if fitness > self.env.gbest_value:
            self.env.gbest_value = fitness
            self.env.gbest_position = self.position.copy()

        # PSO Velocity Update
        r1 = np.random.rand()
        r2 = np.random.rand()
        cognitive = self.c1 * r1 * (self.pbest_position - self.position)
        social = self.c2 * r2 * (self.env.gbest_position - self.position)
        self.velocity = self.w * self.velocity + cognitive + social

        # Obstacle Avoidance
        obstacle_force = self.env.get_obstacle_force(self.position)
        self.velocity += obstacle_force

        # Normalize velocity
        speed = np.linalg.norm(self.velocity)
        max_speed = 1.0
        if speed > max_speed:
            self.velocity = (self.velocity / speed) * max_speed

        # Update position
        new_position = self.position + self.velocity

        # Check boundaries
        new_position = np.clip(new_position, [0, 0], self.env.size)

        # Update energy
        self.energy -= np.linalg.norm(self.velocity)
        if self.energy <= 0:
            self.active = False

        # Update position
        self.position = new_position

        # Survivor Detection
        if self.env.check_survivor(self.position):
            self.survivors_detected += 1
            self.env.leave_marker(self.position)

    def compute_fitness(self):
        # Fitness based on number of unique cells visited
        cell = tuple(self.position.astype(int))
        return self.env.coverage_map.get(cell, 0)

