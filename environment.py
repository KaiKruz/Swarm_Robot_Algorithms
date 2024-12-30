import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, size, num_obstacles, num_survivors):
        self.size = size
        self.obstacles = self.generate_obstacles(num_obstacles)
        self.survivors = self.generate_survivors(num_survivors)
        self.coverage_map = {}
        self.gbest_position = np.zeros(2)
        self.gbest_value = -float('inf')
        self.markers = []

    def generate_obstacles(self, num_obstacles):
        obstacles = []
        for _ in range(num_obstacles):
            position = np.random.rand(2) * self.size
            obstacles.append(position)
        return obstacles

    def generate_survivors(self, num_survivors):
        survivors = []
        for _ in range(num_survivors):
            position = np.random.rand(2) * self.size
            survivors.append(position)
        return survivors

    def get_obstacle_force(self, position):
        force = np.zeros(2)
        for obstacle in self.obstacles:
            distance = np.linalg.norm(position - obstacle)
            if distance < 5.0:
                # Repulsion force
                force += (position - obstacle) / (distance**2)
        return force

    def update_coverage(self, robots):
        for robot in robots:
            if robot.active:
                cell = tuple(robot.position.astype(int))
                self.coverage_map[cell] = self.coverage_map.get(cell, 0) + 1

    def calculate_coverage(self):
        total_cells = self.size[0] * self.size[1]
        covered_cells = len(self.coverage_map)
        return (covered_cells / total_cells) * 100

    def check_survivor(self, position):
        for survivor in self.survivors:
            if np.linalg.norm(position - survivor) < 2.0:
                self.survivors.remove(survivor)
                return True
        return False

    def leave_marker(self, position):
        self.markers.append(position)

    def update_stigmergy(self, robots):
        for robot in robots:
            if robot.active:
                for marker in self.markers:
                    if np.linalg.norm(robot.position - marker) < 5.0:
                        # Attract robot towards the marker
                        robot.velocity += (marker - robot.position) * 0.1

    def visualize(self, robots, step):
        plt.figure(figsize=(8, 8))
        plt.xlim(0, self.size[0])
        plt.ylim(0, self.size[1])

        # Plot obstacles
        for obstacle in self.obstacles:
            plt.plot(obstacle[0], obstacle[1], 'ks', markersize=8)

        # Plot survivors
        for survivor in self.survivors:
            plt.plot(survivor[0], survivor[1], 'ro', markersize=6)

        # Plot robots
        for robot in robots:
            if robot.active:
                plt.plot(robot.position[0], robot.position[1], 'bo', markersize=4)
            else:
                plt.plot(robot.position[0], robot.position[1], 'bx', markersize=4)

        # Plot markers
        for marker in self.markers:
            plt.plot(marker[0], marker[1], 'g*', markersize=6)

        plt.title(f'Step {step}')
        plt.show()

