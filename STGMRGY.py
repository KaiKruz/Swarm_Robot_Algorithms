import numpy as np
import matplotlib.pyplot as plt
import random

# Constants
GRID_SIZE = 100  # Size of the environment grid (100x100)
NUM_ROBOTS = 20  # Number of robots
PHEROMONE_STRENGTH = 100  # Initial pheromone strength
PHEROMONE_DECAY = 0.01  # Pheromone decay rate per iteration
PHEROMONE_DIFFUSION = 0.1  # Rate at which pheromones spread to neighboring cells
SURVIVOR_DETECTION_RADIUS = 5  # Distance within which robots can detect survivors
SENSE_RADIUS = 10  # Radius within which robots can sense pheromones and other robots
MAX_VELOCITY = 2.0  # Maximum robot velocity
COHESION_FACTOR = 0.05  # Weight for the cohesion rule
ALIGNMENT_FACTOR = 0.05  # Weight for the alignment rule
SEPARATION_FACTOR = 0.1  # Weight for the separation rule
MAX_ITERATIONS = 500  # Maximum number of iterations

class Robot:
    """Represents a robot in the swarm with flocking and survivor detection capabilities."""
    def __init__(self, start_position):
        self.position = np.array(start_position, dtype=float)
        self.velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])  # Random initial velocity
        self.survivor_detected = False  # Flag indicating if the robot has detected a survivor

    def update_velocity(self, robots, pheromone_map):
        """Update the velocity based on flocking rules and pheromone detection."""
        positions = np.array([robot.position for robot in robots if robot != self])
        velocities = np.array([robot.velocity for robot in robots if robot != self])

        distances = np.linalg.norm(positions - self.position, axis=1)
        neighbors = distances < SENSE_RADIUS

        if np.any(neighbors):
            cohesion = np.sum(positions[neighbors], axis=0) / np.sum(neighbors)
            alignment = np.sum(velocities[neighbors], axis=0) / np.sum(neighbors)

            # Avoid division by zero for separation
            separation_distances = distances[neighbors]
            separation_distances[separation_distances == 0] = 1e-6  # Set zero distances to a small value
            separation = np.sum((self.position - positions[neighbors]) / separation_distances[:, np.newaxis], axis=0)

            cohesion = (cohesion - self.position) * COHESION_FACTOR
            alignment = (alignment - self.velocity) * ALIGNMENT_FACTOR
            separation = separation * SEPARATION_FACTOR

            self.velocity += cohesion + alignment + separation

        # Sense pheromone levels in the surrounding grid and adjust velocity
        grid_x, grid_y = int(np.clip(self.position[0], 0, GRID_SIZE - 1)), int(np.clip(self.position[1], 0, GRID_SIZE - 1))
        if pheromone_map[grid_x, grid_y] > 0:
            self.velocity += np.random.uniform(-1, 1, 2) * pheromone_map[grid_x, grid_y] * 0.01  # Move toward pheromone

        # Clamp velocity to the maximum allowed velocity
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > MAX_VELOCITY:
            self.velocity = (self.velocity / velocity_magnitude) * MAX_VELOCITY

    def move(self):
        """Move the robot based on its velocity."""
        self.position += self.velocity
        # Ensure the robot stays within the bounds of the grid
        self.position = np.clip(self.position, 0, GRID_SIZE - 1)

    def detect_survivors(self, survivors):
        """Detect if a survivor is within range."""
        distances = np.linalg.norm(survivors - self.position, axis=1)
        if np.any(distances < SURVIVOR_DETECTION_RADIUS):
            self.survivor_detected = True
        return self.survivor_detected

    def deposit_pheromone(self, pheromone_map):
        """Deposit pheromones at the robot's current location."""
        if self.survivor_detected:
            x, y = int(np.clip(self.position[0], 0, GRID_SIZE - 1)), int(np.clip(self.position[1], 0, GRID_SIZE - 1))
            pheromone_map[x, y] += PHEROMONE_STRENGTH

def initialize_survivors(num_survivors, grid_size):
    """Initialize random survivor positions in the environment."""
    survivors = np.random.rand(num_survivors, 2) * grid_size  # Correct scaling for grid size
    return survivors

def initialize_robots(num_robots, grid_size):
    """Initialize random robot positions and velocities in the environment."""
    robots = [Robot(np.random.rand(2) * grid_size) for _ in range(num_robots)]
    return robots

def decay_and_diffuse_pheromones(pheromone_map):
    """Apply pheromone decay and diffusion."""
    pheromone_map *= (1 - PHEROMONE_DECAY)

    diffusion = PHEROMONE_DIFFUSION * (
        np.roll(pheromone_map, 1, axis=0) + np.roll(pheromone_map, -1, axis=0) +
        np.roll(pheromone_map, 1, axis=1) + np.roll(pheromone_map, -1, axis=1) - 4 * pheromone_map
    )
    pheromone_map += diffusion

    return pheromone_map

def simulate_flocking_and_detection(robots, survivors, iterations):
    """Simulate the swarm's flocking behavior and survivor detection."""
    pheromone_map = np.zeros((GRID_SIZE, GRID_SIZE))

    plt.ion()  # Enable interactive plotting
    fig, ax = plt.subplots(figsize=(7, 7))

    for iteration in range(iterations):
        for robot in robots:
            robot.update_velocity(robots, pheromone_map)
            robot.move()
            robot.survivor_detected = robot.detect_survivors(survivors)
            robot.deposit_pheromone(pheromone_map)

        pheromone_map = decay_and_diffuse_pheromones(pheromone_map)

        # Visualization of robots, survivors, and pheromones every 20 iterations
        if iteration % 20 == 0:
            ax.clear()
            ax.imshow(pheromone_map, cmap='hot', origin='lower')
            for survivor in survivors:
                ax.scatter(survivor[0], survivor[1], c='blue', marker='X', s=200)  # Plot survivors
            for robot in robots:
                ax.scatter(robot.position[0], robot.position[1], c='green', s=20)  # Plot robot positions
            ax.set_title(f"Iteration: {iteration}")
            plt.draw()
            plt.pause(0.1)

    plt.ioff()
    plt.show()

# Initialize survivors and robots
survivors = initialize_survivors(5, GRID_SIZE)
robots = initialize_robots(NUM_ROBOTS, GRID_SIZE)

# Simulate the swarm's behavior
simulate_flocking_and_detection(robots, survivors, MAX_ITERATIONS)
