import numpy as np
import matplotlib.pyplot as plt
import random

# Advanced Parameters for ACO
PHEROMONE_STRENGTH_RANGE = (50, 150)  # Variable pheromone strength range
PHEROMONE_DECAY_BASE = 0.02  # Base pheromone decay rate
EVAPORATION_RATE = 0.01  # Pheromone evaporation rate
SPEED_RANGE = (0.5, 1.5)  # Variable robot speeds
ALPHA = 1.0  # Influence of pheromone in decision making
BETA = 2.0  # Influence of distance to target in decision making
GRID_SIZE = 50  # Grid size (50x50)
NUM_ROBOTS = 15  # Number of robots
TARGET = (GRID_SIZE - 2, GRID_SIZE - 2)  # Target location for pathfinding
PHEROMONE_THRESHOLD = 0.1  # Minimum pheromone level for movement influence

class Robot:
    def __init__(self, start_position, grid_size):
        self.x, self.y = start_position
        self.grid_size = grid_size
        self.path = [start_position]  # Path the robot has taken
        self.speed = random.uniform(SPEED_RANGE[0], SPEED_RANGE[1])  # Variable speed
        self.obstacles = set()  # Obstacles in the environment

    def move(self, pheromone_map):
        """Move based on local pheromone levels and distance to the target."""
        neighbors = self.get_neighbors()
        if not neighbors:
            return

        probabilities = self.calculate_move_probabilities(neighbors, pheromone_map)
        next_move = random.choices(neighbors, weights=probabilities)[0]

        # Smooth movement interpolation for visual improvement
        self.x = int(self.x + (next_move[0] - self.x) * 0.3)
        self.y = int(self.y + (next_move[1] - self.y) * 0.3)
        self.path.append((self.x, self.y))

    def get_neighbors(self):
        """Get neighboring cells within the grid."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    neighbors.append((nx, ny))
        return neighbors

    def calculate_move_probabilities(self, neighbors, pheromone_map):
        """Calculate movement probabilities based on pheromone levels and distance to the target."""
        probabilities = []
        for nx, ny in neighbors:
            nx_int = int(nx)
            ny_int = int(ny)
            pheromone_level = np.power(pheromone_map[nx_int, ny_int], ALPHA)
            distance_to_target = 1 / (np.sqrt((TARGET[0] - nx) ** 2 + (TARGET[1] - ny) ** 2) + 1)
            if pheromone_level >= PHEROMONE_THRESHOLD:
                probability = pheromone_level * (distance_to_target ** BETA)
            else:
                probability = 1 / len(neighbors)  # Encourage exploration if pheromone is too low
            probabilities.append(probability)

        # Normalize probabilities
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            probabilities = [1 / len(neighbors) for _ in neighbors]

        return probabilities

    def drop_pheromone(self, pheromone_map):
        """Deposit variable pheromones based on progress toward the target."""
        distance_to_target = np.sqrt((TARGET[0] - self.x) ** 2 + (TARGET[1] - self.y) ** 2)
        pheromone_strength = np.interp(distance_to_target, [0, GRID_SIZE], PHEROMONE_STRENGTH_RANGE)
        pheromone_map[int(self.x), int(self.y)] += pheromone_strength

def evaporate_pheromones(pheromone_map, iteration):
    """Evaporate pheromones over time."""
    pheromone_map *= (1 - EVAPORATION_RATE)

def simulate_aco(robots, pheromone_map, obstacles, iterations=500):
    """Run the optimized Ant Colony Optimization simulation with path visualization."""
    plt.ion()  # Enable interactive plotting
    fig, ax = plt.subplots(figsize=(7, 7))

    for iteration in range(iterations):
        evaporate_pheromones(pheromone_map, iteration)

        for robot in robots:
            if 0 <= robot.x < GRID_SIZE and 0 <= robot.y < GRID_SIZE:
                robot.drop_pheromone(pheromone_map)
            robot.move(pheromone_map)

        # Update visualization periodically
        if iteration % 20 == 0:
            ax.clear()

            # Plot pheromone map
            ax.imshow(pheromone_map, cmap='viridis', origin='lower', alpha=0.6)

            # Plot obstacles
            for (ox, oy) in obstacles:
                ax.add_patch(plt.Rectangle((ox, oy), 1, 1, color='black'))

            # Plot target
            ax.scatter(TARGET[0], TARGET[1], c='red', marker='X', s=200)

            # Plot robots and their paths
            for robot in robots:
                path_x, path_y = zip(*robot.path)  # Unpack the path into x and y coordinates
                ax.plot(path_x, path_y, color='blue', linewidth=1, alpha=0.7)  # Draw the path
                ax.scatter(robot.x, robot.y, c='blue', s=10)  # Current robot position

            ax.set_xlim(0, GRID_SIZE)
            ax.set_ylim(0, GRID_SIZE)
            ax.grid(True)
            ax.set_aspect('equal')
            ax.set_title(f"Iteration: {iteration}")
            plt.draw()
            plt.pause(0.01)  # Short pause to update plot

    plt.ioff()
    plt.show()

# Initialize simulation
robots = [Robot((random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)), GRID_SIZE) for _ in range(NUM_ROBOTS)]
pheromone_map = np.zeros((GRID_SIZE, GRID_SIZE))
obstacles = [(random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)) for _ in range(10)]

# Run simulation with path visualization
simulate_aco(robots, pheromone_map, obstacles)
