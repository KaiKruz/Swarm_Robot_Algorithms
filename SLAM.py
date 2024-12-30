import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter

# Constants for the SLAM Algorithm
GRID_SIZE = 100  # Size of the map (100x100)
NUM_ROBOTS = 10  # Number of robots in the swarm
LIDAR_RANGE = 10  # Range of the LIDAR sensor (in grid cells)
SCAN_ANGLE = 360  # LIDAR scans 360 degrees
PARTICLE_COUNT = 100  # Number of particles per robot
MAX_ITERATIONS = 500  # Maximum number of iterations

# SLAM-specific parameters
MAP_UPDATE_THRESHOLD = 5  # Minimum change in local map to trigger a global map update
CONFIDENCE_DECAY = 0.95  # Confidence decay factor for map fusion

class Particle:
    """Represents a possible position and orientation of the robot."""
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta  # Orientation in radians

    def move(self, velocity, rotation, dt=1.0):
        """Move the particle based on velocity and rotation commands."""
        self.theta += rotation * dt
        self.x += velocity * np.cos(self.theta) * dt
        self.y += velocity * np.sin(self.theta) * dt

class Robot:
    """Represents a robot in the swarm equipped with LIDAR and a particle filter."""
    def __init__(self, start_position, lidar_range, grid_size):
        self.position = np.array(start_position, dtype=float)
        self.velocity = 1.0  # Constant velocity for now
        self.rotation = random.uniform(-0.1, 0.1)  # Small random rotation
        self.lidar_range = lidar_range
        self.grid_size = grid_size
        self.local_map = np.zeros((grid_size, grid_size))  # Local map of the robot
        self.particles = [Particle(*self.position, random.uniform(0, 2 * np.pi)) for _ in range(PARTICLE_COUNT)]
        self.confidence = np.ones((grid_size, grid_size))  # Confidence in the map data

    def lidar_scan(self):
        """Simulate a LIDAR scan and update the local map."""
        angles = np.linspace(0, 2 * np.pi, SCAN_ANGLE)
        for angle in angles:
            for distance in range(self.lidar_range):
                # Simulate a ray in the direction of `angle`
                x = int(self.position[0] + distance * np.cos(angle))
                y = int(self.position[1] + distance * np.sin(angle))
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    self.local_map[x, y] = 1  # Mark as an obstacle
                else:
                    break  # Stop if the ray goes out of bounds

    def update_particles(self):
        """Update the particles based on odometry data and resampling."""
        for particle in self.particles:
            particle.move(self.velocity, self.rotation)
            particle.x = np.clip(particle.x, 0, self.grid_size - 1)
            particle.y = np.clip(particle.y, 0, self.grid_size - 1)

        # Resampling: Importance sampling based on how well each particle fits the current LIDAR scan
        weights = np.array([self.calculate_particle_weight(p) for p in self.particles])
        weights += 1e-300  # Avoid division by zero
        weights /= weights.sum()
        indices = np.random.choice(range(PARTICLE_COUNT), size=PARTICLE_COUNT, p=weights)
        self.particles = [self.particles[i] for i in indices]

    def calculate_particle_weight(self, particle):
        """Calculate the weight of a particle based on the current LIDAR scan."""
        weight = 1.0
        for angle in np.linspace(0, 2 * np.pi, SCAN_ANGLE):
            for distance in range(1, self.lidar_range):
                x = int(particle.x + distance * np.cos(angle))
                y = int(particle.y + distance * np.sin(angle))
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    if self.local_map[x, y] == 1:
                        weight += 1  # Higher weight for matching obstacles
        return weight

    def fuse_local_map(self, global_map, global_confidence):
        """Fuse the local map into the global map using a weighted consensus approach."""
        update_threshold = np.abs(self.local_map - global_map) > MAP_UPDATE_THRESHOLD
        global_map[update_threshold] = (
            global_map[update_threshold] * global_confidence[update_threshold]
            + self.local_map[update_threshold] * self.confidence[update_threshold]
        ) / (global_confidence[update_threshold] + self.confidence[update_threshold])

        # Decay the confidence in global map areas to allow newer data to dominate
        global_confidence[update_threshold] *= CONFIDENCE_DECAY

def initialize_global_map(grid_size):
    """Initialize a global map and a confidence map."""
    global_map = np.zeros((grid_size, grid_size))
    global_confidence = np.ones((grid_size, grid_size))
    return global_map, global_confidence

def simulate_slam(robots, iterations=MAX_ITERATIONS):
    """Simulate the swarm SLAM process."""
    global_map, global_confidence = initialize_global_map(GRID_SIZE)

    plt.ion()  # Enable interactive plotting
    fig, ax = plt.subplots(figsize=(7, 7))

    for iteration in range(iterations):
        for robot in robots:
            # Perform LIDAR scan and update the local map
            robot.lidar_scan()
            robot.update_particles()

            # Fuse the local map into the global map
            robot.fuse_local_map(global_map, global_confidence)

        # Periodically visualize the global map
        if iteration % 10 == 0:
            ax.clear()
            smoothed_map = gaussian_filter(global_map, sigma=1)  # Smooth the global map for better visualization
            ax.imshow(smoothed_map, cmap='gray', origin='lower', alpha=0.8)
            for robot in robots:
                ax.scatter(robot.position[0], robot.position[1], c='blue', s=20)  # Plot robot positions
            ax.set_title(f"Iteration: {iteration}")
            plt.draw()
            plt.pause(0.1)

    plt.ioff()
    plt.show()

# Initialize robots with random starting positions
robots = [Robot(start_position=(random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)), lidar_range=LIDAR_RANGE, grid_size=GRID_SIZE) for _ in range(NUM_ROBOTS)]

# Run the swarm SLAM simulation
simulate_slam(robots)
