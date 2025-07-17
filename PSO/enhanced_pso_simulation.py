import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.animation import FuncAnimation
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PSO Parameters
GRID_SIZE = 50  # Size of the grid (50x50)
NUM_ROBOTS = 25  # Number of robots
MAX_VELOCITY = 2  # Max velocity a robot can have
INITIAL_INERTIA = 0.9  # Initial inertia weight
MIN_INERTIA = 0.4  # Minimum inertia weight (for convergence)
COGNITIVE_INITIAL = 2.0  # Initial Cognitive coefficient
SOCIAL_INITIAL = 1.0  # Initial Social coefficient
COGNITIVE_FINAL = 1.0  # Final Cognitive coefficient
SOCIAL_FINAL = 2.5  # Final Social coefficient
LOCAL_RADIUS = 12  # Initial radius to consider local best
TARGET_COVERAGE = 0.95  # Target coverage percentage
MAX_ITERATIONS = 1000  # Maximum number of iterations
RANDOM_EXPLORATION_THRESHOLD = 20  # Iterations before adding random walk
EARLY_STOPPING_THRESHOLD = 50  # Early stopping if no improvement
ENERGY_CONSUMPTION_RATE = 0.1  # Energy consumption per move
INITIAL_ENERGY = 100.0  # Initial energy for each robot

class Robot:
    def __init__(self, start_position, robot_id):
        self.id = robot_id
        self.position = np.array(start_position, dtype=float)
        self.velocity = np.random.uniform(-1, 1, size=2)
        self.personal_best = self.position.copy()
        self.covered_area = set([tuple(self.position.astype(int))])
        self.iterations_without_improvement = 0
        self.energy = INITIAL_ENERGY
        self.active = True  # Indicates if the robot is operational
        self.color = plt.cm.jet(robot_id / NUM_ROBOTS)  # Color for plotting

    def update_velocity(self, global_best, local_best, inertia, cognitive_weight, social_weight):
        """Update the velocity based on personal, local, and global bests."""
        if not self.active:
            return

        inertia_component = inertia * self.velocity
        cognitive_component = cognitive_weight * np.random.rand() * (self.personal_best - self.position)
        social_component = social_weight * np.random.rand() * (global_best - self.position)
        local_component = social_weight * np.random.rand() * (local_best - self.position)
        new_velocity = inertia_component + cognitive_component + social_component + local_component

        # Clamp the velocity
        new_velocity = np.clip(new_velocity, -MAX_VELOCITY, MAX_VELOCITY)
        self.velocity = new_velocity

    def move(self):
        """Move the robot and update the covered area."""
        if not self.active:
            return

        self.position += self.velocity
        self.position = np.clip(self.position, 0, GRID_SIZE - 1)
        self.covered_area.add(tuple(self.position.astype(int)))

        # Energy consumption
        self.energy -= ENERGY_CONSUMPTION_RATE * np.linalg.norm(self.velocity)
        if self.energy <= 0:
            self.active = False
            print(f"Robot {self.id} has depleted its energy and is inactive.")

    def update_personal_best(self):
        """Update the personal best position."""
        if not self.active:
            return

        current_coverage = len(self.covered_area)
        personal_best_coverage = len(set([tuple(self.personal_best.astype(int))]))
        if current_coverage > personal_best_coverage:
            self.personal_best = self.position.copy()
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1

def calculate_coverage(robots):
    """Calculate the total coverage of the grid."""
    covered_cells = set()
    for robot in robots:
        covered_cells.update(robot.covered_area)
    coverage_percentage = len(covered_cells) / (GRID_SIZE * GRID_SIZE)
    return coverage_percentage, covered_cells

def get_local_best(robot, robots, current_coverage):
    """Find the local best position."""
    adjusted_radius = max(LOCAL_RADIUS * (1 - current_coverage), 3)
    local_best = robot.personal_best.copy()
    local_robots = [r for r in robots if np.linalg.norm(r.position - robot.position) < adjusted_radius and r.active]
    if local_robots:
        best_local_robot = max(local_robots, key=lambda r: len(r.covered_area))
        local_best = best_local_robot.personal_best
    return local_best

def simulate_pso(robots):
    """Simulate the PSO-based exploration."""
    # Data for plotting and analysis
    coverage_history = []
    energy_history = []
    active_robots_history = []
    iterations = []

    # Initialize global best
    global_best_position = np.mean([robot.position for robot in robots if robot.active], axis=0)
    best_coverage = 0
    no_improvement_counter = 0

    # Set up Matplotlib animation
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.close(fig)  # Close the static figure

    def update_plot(iteration):
        nonlocal global_best_position, best_coverage, no_improvement_counter
        inertia = INITIAL_INERTIA - ((INITIAL_INERTIA - MIN_INERTIA) * iteration / MAX_ITERATIONS)
        cognitive_weight = COGNITIVE_INITIAL - ((COGNITIVE_INITIAL - COGNITIVE_FINAL) * iteration / MAX_ITERATIONS)
        social_weight = SOCIAL_INITIAL + ((SOCIAL_FINAL - SOCIAL_INITIAL) * iteration / MAX_ITERATIONS)

        current_coverage, covered_cells = calculate_coverage(robots)
        total_energy = sum(robot.energy for robot in robots)
        active_robots = sum(robot.active for robot in robots)

        # Update velocities and move robots
        for robot in robots:
            local_best_position = get_local_best(robot, robots, current_coverage)
            robot.update_velocity(global_best_position, local_best_position, inertia, cognitive_weight, social_weight)
            robot.move()
            robot.update_personal_best()

            # Random exploration
            if robot.iterations_without_improvement > RANDOM_EXPLORATION_THRESHOLD and robot.active:
                robot.velocity = np.random.uniform(-MAX_VELOCITY, MAX_VELOCITY, size=2)
                robot.iterations_without_improvement = 0

        # Update global best
        active_robot_positions = [robot.personal_best for robot in robots if robot.active]
        if active_robot_positions:
            global_best_position = np.mean(active_robot_positions, axis=0)

        # Early stopping check
        if current_coverage > best_coverage:
            best_coverage = current_coverage
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= EARLY_STOPPING_THRESHOLD:
            print(f"Early stopping at iteration {iteration}")
            ani.event_source.stop()

        # Record data
        coverage_history.append(current_coverage)
        energy_history.append(total_energy)
        active_robots_history.append(active_robots)
        iterations.append(iteration)

        # Visualization
        ax.clear()
        coverage_map = np.zeros((GRID_SIZE, GRID_SIZE))
        for robot in robots:
            for cell in robot.covered_area:
                coverage_map[cell] = 1

        sns.heatmap(coverage_map, cmap='Greens', cbar=False, ax=ax)
        ax.set_title(f"Iteration: {iteration}, Coverage: {current_coverage:.2%}, Active Robots: {active_robots}")
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # Plot robot positions
        for robot in robots:
            if robot.active:
                ax.plot(robot.position[0], robot.position[1], 'o', color=robot.color, markersize=5)

        # Stop if target coverage is reached
        if current_coverage >= TARGET_COVERAGE:
            print(f"Target coverage of {TARGET_COVERAGE*100}% reached at iteration {iteration}")
            ani.event_source.stop()

    # Create animation
    ani = FuncAnimation(fig, update_plot, frames=MAX_ITERATIONS, repeat=False, interval=100)

    # Display the animation
    from IPython.display import HTML
    HTML(ani.to_jshtml())

    # After simulation
    plt.show()

    # Plot KPIs over iterations
    fig_kpi, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].plot(iterations, [c * 100 for c in coverage_history], label='Coverage (%)')
    axs[0].set_title('Coverage Over Time')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Coverage (%)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(iterations, energy_history, label='Total Energy', color='orange')
    axs[1].set_title('Total Energy Over Time')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Total Energy')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(iterations, active_robots_history, label='Active Robots', color='green')
    axs[2].set_title('Active Robots Over Time')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Number of Active Robots')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Interactive Plotly Visualization
    fig_plotly = make_subplots(rows=1, cols=1)
    fig_plotly.add_trace(go.Heatmap(z=coverage_map, colorscale='Greens'), row=1, col=1)
    robot_positions = np.array([robot.position for robot in robots if robot.active])
    fig_plotly.add_trace(go.Scatter(x=robot_positions[:, 0], y=robot_positions[:, 1], mode='markers',
                                    marker=dict(color='blue', size=5), name='Robots'), row=1, col=1)
    fig_plotly.update_layout(title=f'Final Coverage: {best_coverage:.2%}')
    fig_plotly.show()

# Initialize robots
robots = [Robot(start_position=(random.uniform(0, GRID_SIZE - 1), random.uniform(0, GRID_SIZE - 1)), robot_id=i) for i in range(NUM_ROBOTS)]

# Run the simulation
simulate_pso(robots)
