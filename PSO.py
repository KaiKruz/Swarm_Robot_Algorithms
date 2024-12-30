import numpy as np
import matplotlib.pyplot as plt
import random

# PSO Parameters
GRID_SIZE = 50  # Size of the grid (50x50)
NUM_ROBOTS = 25  # Number of robots (increased for more effective coverage)
MAX_VELOCITY = 2  # Max velocity a robot can have
INITIAL_INERTIA = 0.9  # Initial inertia weight
MIN_INERTIA = 0.4  # Minimum inertia weight (for more convergence later)
COGNITIVE_INITIAL = 2.0  # Initial Cognitive coefficient (personal best influence)
SOCIAL_INITIAL = 1.0  # Initial Social coefficient (global best influence)
COGNITIVE_FINAL = 1.0  # Final Cognitive coefficient
SOCIAL_FINAL = 2.5  # Final Social coefficient
LOCAL_RADIUS = 12  # Initial radius to consider local best (for limited communication)
TARGET_COVERAGE = 0.75  # Target coverage percentage (optimized for 75% coverage)
MAX_ITERATIONS = 1000  # Maximum number of iterations
RANDOM_EXPLORATION_THRESHOLD = 20  # Iterations before adding controlled random walk
EARLY_STOPPING_THRESHOLD = 50  # Early stop if no significant improvement for 50 iterations

class Robot:
    def __init__(self, start_position):
        self.position = np.array(start_position, dtype=float)
        self.velocity = np.random.uniform(-1, 1, size=2)  # Random initial velocity
        self.personal_best = self.position.copy()  # Best-known position of this robot
        self.covered_area = set([tuple(self.position.astype(int))])  # Tracks the area covered by the robot
        self.iterations_without_improvement = 0  # Tracks the number of iterations without improving the personal best

    def update_velocity(self, global_best, local_best, inertia, cognitive_weight, social_weight):
        """Update the velocity based on personal, local, and global bests."""
        inertia_component = inertia * self.velocity
        cognitive_component = cognitive_weight * np.random.rand() * (self.personal_best - self.position)
        social_component = social_weight * np.random.rand() * (global_best - self.position)
        local_component = social_weight * np.random.rand() * (local_best - self.position)
        new_velocity = inertia_component + cognitive_component + social_component + local_component

        # Clamp the velocity to the maximum allowed velocity
        new_velocity = np.clip(new_velocity, -MAX_VELOCITY, MAX_VELOCITY)
        self.velocity = new_velocity

    def move(self):
        """Move the robot based on its velocity and update the covered area."""
        self.position += self.velocity
        # Ensure the robot stays within the grid boundaries
        self.position = np.clip(self.position, 0, GRID_SIZE - 1)
        self.covered_area.add(tuple(self.position.astype(int)))

    def update_personal_best(self):
        """Update the personal best if the current position covers new area."""
        current_coverage = len(self.covered_area)
        personal_best_coverage = len(set([tuple(self.personal_best.astype(int))]))
        if current_coverage > personal_best_coverage:
            self.personal_best = self.position.copy()
            self.iterations_without_improvement = 0  # Reset if improvement
        else:
            self.iterations_without_improvement += 1  # Increment stagnation counter if no improvement

def calculate_coverage(robots, grid_size):
    """Calculate the total coverage of the grid."""
    covered_cells = set()
    for robot in robots:
        covered_cells.update(robot.covered_area)
    return len(covered_cells) / (grid_size * grid_size)

def get_local_best(robot, robots, current_coverage):
    """Find the local best position based on robots within a dynamically adjusted radius."""
    adjusted_radius = max(LOCAL_RADIUS * (1 - current_coverage), 3)  # Shrink radius as coverage increases
    local_best = robot.personal_best.copy()
    local_robots = [r for r in robots if np.linalg.norm(r.position - robot.position) < adjusted_radius]
    if local_robots:
        best_local_robot = max(local_robots, key=lambda r: len(r.covered_area))
        local_best = best_local_robot.personal_best
    return local_best

def simulate_pso(robots, iterations=MAX_ITERATIONS):
    """Simulate the hyper-optimized PSO-based exploration."""
    plt.ion()  # Enable interactive plotting
    fig, ax = plt.subplots(figsize=(7, 7))

    global_best_position = np.mean([robot.position for robot in robots], axis=0)  # Initialize global best
    best_coverage = 0  # Track the best coverage achieved
    no_improvement_counter = 0  # Early stopping counter

    for iteration in range(iterations):
        # Dynamic inertia, cognitive, and social components
        inertia = INITIAL_INERTIA - ((INITIAL_INERTIA - MIN_INERTIA) * iteration / MAX_ITERATIONS)
        cognitive_weight = COGNITIVE_INITIAL - ((COGNITIVE_INITIAL - COGNITIVE_FINAL) * iteration / MAX_ITERATIONS)
        social_weight = SOCIAL_INITIAL + ((SOCIAL_FINAL - SOCIAL_INITIAL) * iteration / MAX_ITERATIONS)

        # Get the current coverage percentage
        current_coverage = calculate_coverage(robots, GRID_SIZE)

        # Update velocities and move robots
        for robot in robots:
            local_best_position = get_local_best(robot, robots, current_coverage)
            robot.update_velocity(global_best_position, local_best_position, inertia, cognitive_weight, social_weight)
            robot.move()
            robot.update_personal_best()

            # Controlled random walk if no improvement for several iterations
            if robot.iterations_without_improvement > RANDOM_EXPLORATION_THRESHOLD:
                robot.velocity = np.random.uniform(-MAX_VELOCITY, MAX_VELOCITY, size=2)
                robot.iterations_without_improvement = 0  # Reset exploration counter

        # Update global best based on the swarm's exploration progress
        global_best_position = np.mean([robot.personal_best for robot in robots], axis=0)

        # Calculate the current coverage of the grid
        coverage = calculate_coverage(robots, GRID_SIZE)

        # Early stopping mechanism if no significant improvement
        if coverage > best_coverage:
            best_coverage = coverage
            no_improvement_counter = 0  # Reset the early stopping counter
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= EARLY_STOPPING_THRESHOLD:
            print(f"Early stopping: No significant coverage improvement for {EARLY_STOPPING_THRESHOLD} iterations.")
            break

        # Plot the robots and coverage periodically
        if iteration % 10 == 0:
            ax.clear()
            coverage_map = np.zeros((GRID_SIZE, GRID_SIZE))
            for robot in robots:
                for cell in robot.covered_area:
                    coverage_map[cell] = 1
                ax.scatter(robot.position[0], robot.position[1], c='blue', s=20)  # Plot current robot positions

            # Visualize the covered area
            ax.imshow(coverage_map, cmap='Greens', origin='lower', alpha=0.6)
            ax.set_xlim(0, GRID_SIZE)
            ax.set_ylim(0, GRID_SIZE)
            ax.grid(True)
            ax.set_title(f"Iteration: {iteration}, Coverage: {coverage:.2%}")
            plt.draw()
            plt.pause(0.1)

        # Stop if the target coverage is reached
        if coverage >= TARGET_COVERAGE:
            print(f"Target coverage of {TARGET_COVERAGE*100}% reached at iteration {iteration}")
            break

    plt.ioff()
    plt.show()

# Initialize the robots at random starting positions within the grid
robots = [Robot(start_position=(random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))) for _ in range(NUM_ROBOTS)]

# Run the hyper-optimized PSO-based exploration simulation
simulate_pso(robots)