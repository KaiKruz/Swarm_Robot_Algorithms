import matplotlib.pyplot as plt
import numpy as np

class OptimizedRobot:
    def __init__(self, grid_size, grid_position, nest_position, robot_id):
        self.grid_size = grid_size
        self.grid_position = grid_position
        self.position = [grid_position[0] * grid_size, grid_position[1] * grid_size]
        self.nest_position = nest_position
        self.covered = []  # Cells the robot has covered
        self.has_target = False  # If the robot is carrying a target
        self.targets_collected = 0  # Number of targets collected
        self.robot_id = robot_id
        self.target_priority = None  # Current target the robot is foraging for

    def move_in_grid(self):
        """Move the robot to cover the grid systematically."""
        if len(self.covered) < self.grid_size**2 and not self.has_target:
            x, y = self.position
            if (x, y) not in self.covered:
                self.covered.append((x, y))
            if x < (self.grid_position[0] + 1) * self.grid_size - 1:
                self.position[0] += 1
            elif y < (self.grid_position[1] + 1) * self.grid_size - 1:
                self.position[0] = self.grid_position[0] * self.grid_size
                self.position[1] += 1

    def move_towards(self, target_position):
        """Move towards a specific position (used for foraging or returning to nest)."""
        if self.position[0] < target_position[0]:
            self.position[0] += 1
        elif self.position[0] > target_position[0]:
            self.position[0] -= 1
        if self.position[1] < target_position[1]:
            self.position[1] += 1
        elif self.position[1] > target_position[1]:
            self.position[1] -= 1

    def switch_to_foraging(self, target_positions):
        """Switch to foraging mode by finding the closest available target."""
        nearest_target = min(target_positions, key=lambda t: np.hypot(self.position[0] - t[0], self.position[1] - t[1]))
        self.target_priority = nearest_target
        return nearest_target

    def combined_behavior(self, target_positions, robot_positions):
        """Switch between coverage and foraging behavior based on conditions."""
        if not self.has_target:
            # If a target is nearby, switch to foraging
            if target_positions:
                nearest_target = self.switch_to_foraging(target_positions)
                if nearest_target:
                    self.move_towards(nearest_target)
                    current_position = tuple(self.position)
                    if current_position == nearest_target:
                        # Pick up the target and start heading to the nest
                        self.has_target = True
                        target_positions.remove(nearest_target)
                        self.target_priority = None  # Target collected
                else:
                    # Continue grid coverage if no targets nearby
                    self.move_in_grid()
            else:
                # Continue grid coverage if no targets available
                self.move_in_grid()
        else:
            # Foraging mode: Move back to the nest
            self.move_towards(self.nest_position)
            if self.position == self.nest_position:
                # Deliver the target and return to coverage
                self.has_target = False
                self.targets_collected += 1
                self.target_priority = None

def optimized_robots(robots, target_positions, iterations=300):
    plt.ion()  # Interactive mode
    fig, ax = plt.subplots()

    for _ in range(iterations):
        ax.clear()

        # Draw the grid and targets
        for (x, y) in target_positions:
            rect = plt.Rectangle((x, y), 1, 1, color='green')  # Targets are green
            ax.add_patch(rect)

        for robot in robots:
            # Show robot behavior (orange if carrying a target, red if covering)
            if robot.has_target:
                ax.scatter(robot.position[0], robot.position[1], c='orange', label=f'Robot {robot.robot_id}')
            else:
                ax.scatter(robot.position[0], robot.position[1], c='red', label=f'Robot {robot.robot_id}')

            # Draw covered cells as blue (with transparency)
            for (x, y) in robot.covered:
                rect = plt.Rectangle((x, y), 1, 1, color='blue', alpha=0.3)
                ax.add_patch(rect)

        ax.set_xlim(0, robots[0].grid_size * 2)
        ax.set_ylim(0, robots[0].grid_size * 2)
        ax.set_xticks(np.arange(0, robots[0].grid_size * 2 + 1, 1))
        ax.set_yticks(np.arange(0, robots[0].grid_size * 2 + 1, 1))
        ax.grid(True)
        plt.draw()
        plt.pause(0.05)  # Pause to update the plot

        # Each robot performs the combined behavior
        for robot in robots:
            robot.combined_behavior(target_positions, [(r.position[0], r.position[1]) for r in robots])

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Final display

# Initialize robots in a scalable grid with a nest in the center and multiple targets
grid_size = 10
nest_position = (grid_size // 2, grid_size // 2)
robots = [
    OptimizedRobot(5, (0, 0), nest_position, 1), OptimizedRobot(5, (1, 0), nest_position, 2),
    OptimizedRobot(5, (0, 1), nest_position, 3), OptimizedRobot(5, (1, 1), nest_position, 4)
]
target_positions = [tuple(np.random.randint(0, grid_size, size=2)) for _ in range(12)]  # 12 random targets

# Run the optimized combined coverage and foraging process
optimized_robots(robots, target_positions)
