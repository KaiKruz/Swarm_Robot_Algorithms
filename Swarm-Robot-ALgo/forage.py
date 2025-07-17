import matplotlib.pyplot as plt
import numpy as np

class CoverageRobot:
    def __init__(self, grid_size, grid_position, robot_id):
        self.grid_size = grid_size
        self.grid_position = grid_position
        self.position = [grid_position[0] * grid_size, grid_position[1] * grid_size]
        self.covered = set()  # Set for faster lookup and storage of covered cells
        self.robot_id = robot_id  # Unique ID for each robot
        self.moves = 0  # Counter for spiral movement steps

    def move_in_spiral(self):
        """Move in a spiral pattern for efficient coverage."""
        x, y = self.position
        if (x, y) not in self.covered:
            self.covered.add((x, y))  # Mark current position as covered

        # Spiral movement logic, changing direction based on boundaries
        if self.moves % 4 == 0:  # Move right
            if x < (self.grid_position[0] + 1) * self.grid_size - 1:
                self.position[0] += 1
            else:
                self.moves += 1
        elif self.moves % 4 == 1:  # Move down
            if y < (self.grid_position[1] + 1) * self.grid_size - 1:
                self.position[1] += 1
            else:
                self.moves += 1
        elif self.moves % 4 == 2:  # Move left
            if x > self.grid_position[0] * self.grid_size:
                self.position[0] -= 1
            else:
                self.moves += 1
        elif self.moves % 4 == 3:  # Move up
            if y > self.grid_position[1] * self.grid_size:
                self.position[1] -= 1
            else:
                self.moves += 1

        # Increment move counter to control direction switching
        self.moves += 1

def coverage_robots(robots, iterations=1000, plot_interval=50):
    """Simulate and visualize robot coverage in real-time."""
    plt.ion()  # Interactive mode for real-time plotting
    fig, ax = plt.subplots(figsize=(10, 10))

    # Calculate max grid size across all robots
    max_grid_size = max(robot.grid_size * (robot.grid_position[0] + 1) for robot in robots)

    for iteration in range(iterations):
        ax.clear()

        # Draw grid and robot positions
        for robot in robots:
            # Draw covered cells
            for (x, y) in robot.covered:
                rect = plt.Rectangle((x, y), 1, 1, color='blue', alpha=0.5)
                ax.add_patch(rect)

            # Draw robot position
            ax.scatter(robot.position[0], robot.position[1], c='red', s=10, label=f'Robot {robot.robot_id}')

        # Set grid visualization parameters
        ax.set_xlim(0, max_grid_size)
        ax.set_ylim(0, max_grid_size)
        ax.set_xticks(np.arange(0, max_grid_size + 1, 5))  # Larger ticks for readability
        ax.set_yticks(np.arange(0, max_grid_size + 1, 5))
        ax.grid(True)

        ax.set_title(f"Iteration: {iteration}")
        plt.draw()

        # Update plot at intervals for better performance
        if iteration % plot_interval == 0:
            plt.pause(0.01)  # Pause briefly to update the plot

        # Move all robots in a spiral
        for robot in robots:
            robot.move_in_spiral()

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Final display

# Grid and robot scaling parameters
total_grid_size = 100  # 100x100 grid
num_partitions = 10  # Grid divided into 10x10 sections
partition_size = total_grid_size // num_partitions

# Initialize robots for each partition in the grid
robots = []
robot_id = 1
for i in range(num_partitions):
    for j in range(num_partitions):
        robots.append(CoverageRobot(partition_size, (i, j), robot_id))
        robot_id += 1

# Run the optimized scaled coverage process
coverage_robots(robots)
