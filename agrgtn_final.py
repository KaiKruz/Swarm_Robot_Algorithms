import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree  # Faster neighbor search using k-d tree

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.sensing_range = 10  # A more practical sensing range
        self.speed = 0.5  # Robots move faster to speed up aggregation

    def move_towards(self, target_x, target_y):
        """Move towards a target point (with controlled speed)."""
        dx = target_x - self.x
        dy = target_y - self.y
        distance = np.sqrt(dx**2 + dy**2)
        if distance > 0:
            self.x += dx / distance * self.speed
            self.y += dy / distance * self.speed

    def apply_repulsion(self, neighbors):
        """Apply repulsion force to avoid crowding with nearby robots."""
        repulsion_x, repulsion_y = 0, 0
        for neighbor in neighbors:
            dx = self.x - neighbor.x
            dy = self.y - neighbor.y
            distance = np.sqrt(dx**2 + dy**2)
            if distance > 0 and distance < self.sensing_range / 2:  # Apply repulsion if too close
                repulsion_x += dx / distance
                repulsion_y += dy / distance
        return repulsion_x, repulsion_y

def aggregate_robots(robots, iterations=1000, plot_interval=50):
    plt.ion()  # Interactive mode on for real-time plotting
    fig, ax = plt.subplots()

    target_x, target_y = 50, 50  # Initial aggregation target (center of the space)
    
    # Main simulation loop
    for i in range(iterations):
        # Build a KD-Tree for efficient neighbor search
        positions = np.array([[robot.x, robot.y] for robot in robots])
        tree = cKDTree(positions)
        
        for robot in robots:
            # Get nearby neighbors within the sensing range
            neighbors_idx = tree.query_ball_point([robot.x, robot.y], r=robot.sensing_range)
            neighbors = [robots[idx] for idx in neighbors_idx if robots[idx] is not robot]
            
            # Calculate the centroid of neighbors if available, otherwise move to the global target
            if neighbors:
                centroid_x = np.mean([r.x for r in neighbors])
                centroid_y = np.mean([r.y for r in neighbors])
                # Apply aggregation towards the centroid
                robot.move_towards(centroid_x, centroid_y)
                
                # Apply repulsion to avoid crowding
                repulsion_x, repulsion_y = robot.apply_repulsion(neighbors)
                robot.x += repulsion_x * 0.05  # Adjust repulsion strength
                robot.y += repulsion_y * 0.05
            else:
                # Move towards the global aggregation target if no neighbors
                robot.move_towards(target_x, target_y)

        # Periodically update the plot
        if i % plot_interval == 0:
            ax.clear()
            ax.scatter([robot.x for robot in robots], [robot.y for robot in robots], c='blue', s=50)
            ax.scatter(target_x, target_y, c='red', marker='X', s=100)  # Plot the target point
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_title(f"Iteration: {i}")
            plt.draw()
            plt.pause(0.001)  # Shorter pause for faster updates

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Final display

# Initialize robots at random positions, scalable for larger numbers
num_robots = 100  # Increased number of robots for scalability
robots = [Robot(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_robots)]

# Run the optimized aggregation process
aggregate_robots(robots)
