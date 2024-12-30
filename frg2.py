import random
import numpy as np
import matplotlib.pyplot as plt

class ForagingRobot:
    def __init__(self, x, y, base_x, base_y):
        self.x = x
        self.y = y
        self.base_x = base_x
        self.base_y = base_y
        self.target_found = False
        self.target = None
        self.path = [(x, y)]  # Store the robot's path

    def move_randomly(self):
        self.x += random.uniform(-1, 1)
        self.y += random.uniform(-1, 1)
        self.path.append((self.x, self.y))  # Update the robot's path

    def detect_target(self, targets):
        for target in targets:
            if np.sqrt((self.x - target[0])**2 + (self.y - target[1])**2) < 5:
                self.target_found = True
                self.target = target
                targets.remove(target)
                break

    def move_to_target(self):
        if self.target:
            dx = self.target[0] - self.x
            dy = self.target[1] - self.y
            distance = np.sqrt(dx**2 + dy**2)
            if distance > 0:
                self.x += dx / distance * 0.1  # Move 10% of the distance to the target
                self.y += dy / distance * 0.1
                self.path.append((self.x, self.y))  # Update the robot's path

    def return_to_base(self):
        dx = self.base_x - self.x
        dy = self.base_y - self.y
        distance = np.sqrt(dx**2 + dy**2)
        if distance > 0:
            self.x += dx / distance * 0.1  # Move 10% of the distance to the base
            self.y += dy / distance * 0.1
            self.path.append((self.x, self.y))  # Update the robot's path
        if np.sqrt(dx**2 + dy**2) < 1:
            self.target_found = False
            self.target = None

def foraging_robots(robots, targets, iterations=1000, plot_interval=10):
    plt.ion()  # Interactive mode on for real-time plotting
    fig, ax = plt.subplots()

    for i in range(iterations):
        ax.clear()
        ax.scatter([robot.x for robot in robots], [robot.y for robot in robots], c='blue', label='Robots')
        ax.scatter([target[0] for target in targets], [target[1] for target in targets], c='red', label='Targets')
        ax.scatter(50, 50, c='green', marker='X', s=100, label='Base')  # Base location

        for robot in robots:
            if not robot.target_found:
                robot.move_randomly()
                robot.detect_target(targets)
            elif robot.target_found:
                robot.move_to_target()
                if robot.target is None:  # If the target is collected
                    robot.return_to_base()

            # Plot the robot's path
            ax.plot([x for x, y in robot.path], [y for x, y in robot.path], c='gray', alpha=0.5)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title(f"Iteration: {i}")
        ax.legend()
        plt.draw()
        plt.pause(0.1)  # Pause to update the plot every 0.1 seconds

        if i % plot_interval == 0:
            plt.savefig(f"iteration_{i}.png")  # Save a snapshot of the plot every plot_interval iterations

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Final display

# Initialize robots and targets
robots = [ForagingRobot(random.uniform(0, 100), random.uniform(0, 100), 50, 50) for _ in range(10)]
targets = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(5)]

# Run the foraging process
foraging_robots(robots, targets)