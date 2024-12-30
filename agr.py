import random
import numpy as np
import matplotlib.pyplot as plt

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.sensing_range = 10  # Example sensing range

    def move_towards(self, target_x, target_y):
        dx = target_x - self.x
        dy = target_y - self.y
        distance = np.sqrt(dx**2 + dy**2)
        if distance > 0:
            self.x += dx / distance
            self.y += dy / distance

    def random_walk(self):
        self.x += random.uniform(-1, 1)
        self.y += random.uniform(-1, 1)

    def sense_robots(self, robots):
        neighbors = []
        for robot in robots:
            if robot is not self:
                distance = np.sqrt((self.x - robot.x)**2 + (self.y - robot.y)**2)
                if distance < self.sensing_range:
                    neighbors.append(robot)
        return neighbors

def aggregate_robots(robots, iterations=1000, plot_interval=10):
    plt.ion()  # Interactive mode on for real-time plotting
    fig, ax = plt.subplots()
    
    for i in range(iterations):
        for robot in robots:
            neighbors = robot.sense_robots(robots)
            if neighbors:
                center_x = np.mean([r.x for r in neighbors])
                center_y = np.mean([r.y for r in neighbors])
                robot.move_towards(center_x, center_y)
            else:
                robot.random_walk()

        if i % plot_interval == 0:  # Update plot every few iterations
            ax.clear()
            ax.scatter([robot.x for robot in robots], [robot.y for robot in robots], c='blue')
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_title(f"Iteration: {i}")
            plt.draw()
            plt.pause(0.1)  # Pause to update the plot

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Final display

# Initialize robots at random positions
robots = [Robot(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(20)]

# Run the aggregation process
aggregate_robots(robots)