import matplotlib.pyplot as plt
import numpy as np

class CoverageRobot:
    def __init__(self, grid_size, grid_position):
        self.grid_size = grid_size
        self.grid_position = grid_position
        self.position = [grid_position[0] * grid_size, grid_position[1] * grid_size]
        self.covered = []

    def move_in_grid(self):
        if len(self.covered) < self.grid_size**2:
            x, y = self.position
            if (x, y) not in self.covered:
                self.covered.append((x, y))
            if x < (self.grid_position[0] + 1) * self.grid_size - 1:
                self.position[0] += 1
            elif y < (self.grid_position[1] + 1) * self.grid_size - 1:
                self.position[0] = self.grid_position[0] * self.grid_size
                self.position[1] += 1

def coverage_robots(robots, iterations=100):
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()

    for _ in range(iterations):
        ax.clear()

        # Draw the grid
        for robot in robots:
            for (x, y) in robot.covered:
                rect = plt.Rectangle((x, y), 1, 1, color='blue')
                ax.add_patch(rect)
            ax.scatter(robot.position[0], robot.position[1], c='red')

        ax.set_xlim(0, max(robot.grid_size * 2 for robot in robots))
        ax.set_ylim(0, max(robot.grid_size * 2 for robot in robots))
        ax.set_xticks(np.arange(0, max(robot.grid_size * 2 for robot in robots) + 1, 1))
        ax.set_yticks(np.arange(0, max(robot.grid_size * 2 for robot in robots) + 1, 1))
        ax.grid(True)
        plt.draw()
        plt.pause(0.1)  # Pause to update the plot

        for robot in robots:
            robot.move_in_grid()

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Final display

# Initialize robots in a 10x10 area divided into 2x2 grids
robots = [
    CoverageRobot(5, (0, 0)), CoverageRobot(5, (1, 0)),
    CoverageRobot(5, (0, 1)), CoverageRobot(5, (1, 1))
]

# Run the coverage process
coverage_robots(robots)
