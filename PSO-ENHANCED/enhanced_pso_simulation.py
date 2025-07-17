import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import time

# PSO Parameters
GRID_SIZE = 100  # Increased grid size for more complexity
NUM_ROBOTS = 50  # Number of robots
MAX_VELOCITY = 4  # Increased max velocity for faster coverage
MIN_VELOCITY = 0.5  # Minimum velocity to prevent stagnation
INITIAL_INERTIA = 0.9  # Initial inertia weight
FINAL_INERTIA = 0.4  # Final inertia weight
COGNITIVE_INITIAL = 2.5  # Initial Cognitive coefficient
SOCIAL_INITIAL = 1.5  # Initial Social coefficient
COGNITIVE_FINAL = 1.0  # Final Cognitive coefficient
SOCIAL_FINAL = 2.5  # Final Social coefficient
LOCAL_RADIUS = 15  # Communication radius
TARGET_COVERAGE = 0.99  # Target coverage percentage
MAX_ITERATIONS = 2000  # Increased iterations for thorough exploration
STAGNATION_THRESHOLD = 5  # Iterations without improvement before random exploration
EARLY_STOPPING_THRESHOLD = 150  # Early stopping if no improvement
ENERGY_CONSUMPTION_RATE = 0.05  # Energy consumption per move
INITIAL_ENERGY = 200.0  # Increased initial energy
RECHARGE_POINTS = [(GRID_SIZE/2, GRID_SIZE/2)]  # Locations where robots can recharge
OBSTACLE_COUNT = 20  # Number of obstacles
OBSTACLE_SIZE = 5  # Size of obstacles

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
        self.stagnation_counter = 0

    def update_velocity(self, global_best, local_best, inertia, cognitive_weight, social_weight, obstacles):
        """Update the velocity based on personal, local, and global bests, and avoid obstacles."""
        if not self.active:
            return

        r1, r2 = np.random.rand(), np.random.rand()
        inertia_component = inertia * self.velocity
        cognitive_component = cognitive_weight * r1 * (self.personal_best - self.position)
        social_component = social_weight * r2 * (global_best - self.position)
        local_component = social_weight * r2 * (local_best - self.position)

        # Obstacle avoidance using potential fields
        obstacle_avoidance = np.zeros(2)
        for obs in obstacles:
            vector_to_obstacle = self.position - obs.center
            distance = np.linalg.norm(vector_to_obstacle)
            if distance < obs.radius + 5:
                repulsion = (vector_to_obstacle / distance) * (1 / (distance - obs.radius + 0.1))
                obstacle_avoidance += repulsion

        new_velocity = inertia_component + cognitive_component + social_component + local_component + obstacle_avoidance

        # Clamp the velocity
        speed = np.linalg.norm(new_velocity)
        if speed > MAX_VELOCITY:
            new_velocity = (new_velocity / speed) * MAX_VELOCITY
        elif speed < MIN_VELOCITY:
            new_velocity = (new_velocity / speed) * MIN_VELOCITY

        self.velocity = new_velocity

    def move(self, obstacles):
        """Move the robot, avoid obstacles, and update the covered area."""
        if not self.active:
            return

        potential_position = self.position + self.velocity

        # Check for collision with boundaries
        potential_position = np.clip(potential_position, 0, GRID_SIZE - 1)

        # Check for collision with obstacles
        collision = False
        for obs in obstacles:
            if obs.contains(potential_position):
                collision = True
                break

        if not collision:
            self.position = potential_position
            self.covered_area.add(tuple(self.position.astype(int)))
        else:
            # Reflect velocity vector
            self.velocity = -self.velocity
            self.position += self.velocity
            self.position = np.clip(self.position, 0, GRID_SIZE - 1)
            self.covered_area.add(tuple(self.position.astype(int)))

        # Energy consumption
        self.energy -= ENERGY_CONSUMPTION_RATE * np.linalg.norm(self.velocity)
        if self.energy <= 0:
            self.active = False
            print(f"Robot {self.id} has depleted its energy and is inactive.")
        else:
            # Check for recharge
            for rp in RECHARGE_POINTS:
                if np.linalg.norm(self.position - np.array(rp)) < 2:
                    self.energy = INITIAL_ENERGY
                    print(f"Robot {self.id} has recharged at {rp}.")

    def update_personal_best(self):
        """Update the personal best position."""
        if not self.active:
            return

        current_coverage = len(self.covered_area)
        personal_best_coverage = len(set([tuple(self.personal_best.astype(int))]))
        if current_coverage > personal_best_coverage:
            self.personal_best = self.position.copy()
            self.iterations_without_improvement = 0
            self.stagnation_counter = 0
        else:
            self.iterations_without_improvement += 1
            self.stagnation_counter += 1

class Obstacle:
    def __init__(self, center):
        self.center = np.array(center)
        self.radius = OBSTACLE_SIZE

    def contains(self, point):
        return np.linalg.norm(point - self.center) < self.radius

def generate_obstacles(count):
    obstacles = []
    for _ in range(count):
        center = (
            random.uniform(OBSTACLE_SIZE, GRID_SIZE - OBSTACLE_SIZE),
            random.uniform(OBSTACLE_SIZE, GRID_SIZE - OBSTACLE_SIZE)
        )
        obstacles.append(Obstacle(center))
    return obstacles

def calculate_coverage(robots):
    """Calculate the total coverage of the grid."""
    covered_cells = set()
    for robot in robots:
        covered_cells.update(robot.covered_area)
    coverage_percentage = len(covered_cells) / (GRID_SIZE * GRID_SIZE)
    return coverage_percentage, covered_cells

def get_local_best(robot, robots):
    """Find the local best position."""
    local_robots = [r for r in robots if np.linalg.norm(r.position - robot.position) < LOCAL_RADIUS and r.active]
    if local_robots:
        best_local_robot = max(local_robots, key=lambda r: len(r.covered_area))
        return best_local_robot.personal_best
    else:
        return robot.personal_best

def simulate_pso(sim_id, results):
    """Simulate the PSO-based exploration."""
    robots = [Robot(
        start_position=(
            random.uniform(0, GRID_SIZE - 1),
            random.uniform(0, GRID_SIZE - 1)
        ),
        robot_id=i
    ) for i in range(NUM_ROBOTS)]

    obstacles = generate_obstacles(OBSTACLE_COUNT)

    # Data for plotting and analysis
    coverage_history = []
    energy_history = []
    active_robots_history = []
    iterations = []

    # Initialize global best
    global_best_position = np.mean([robot.position for robot in robots if robot.active], axis=0)
    best_coverage = 0
    no_improvement_counter = 0

    for iteration in range(MAX_ITERATIONS):
        inertia = INITIAL_INERTIA - ((INITIAL_INERTIA - FINAL_INERTIA) * iteration / MAX_ITERATIONS)
        cognitive_weight = COGNITIVE_INITIAL - ((COGNITIVE_INITIAL - COGNITIVE_FINAL) * iteration / MAX_ITERATIONS)
        social_weight = SOCIAL_INITIAL + ((SOCIAL_FINAL - SOCIAL_INITIAL) * iteration / MAX_ITERATIONS)

        current_coverage, _ = calculate_coverage(robots)
        total_energy = sum(robot.energy for robot in robots)
        active_robots = sum(robot.active for robot in robots)

        # Update velocities and move robots
        for robot in robots:
            local_best_position = get_local_best(robot, robots)
            robot.update_velocity(global_best_position, local_best_position, inertia, cognitive_weight, social_weight, obstacles)
            robot.move(obstacles)
            robot.update_personal_best()

            # Random exploration using Lévy flight if stagnation detected
            if robot.stagnation_counter > STAGNATION_THRESHOLD and robot.active:
                step_length = np.random.pareto(1.5) * MAX_VELOCITY
                angle = np.random.uniform(0, 2 * np.pi)
                robot.velocity = np.array([np.cos(angle), np.sin(angle)]) * step_length
                robot.stagnation_counter = 0

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
            print(f"Simulation {sim_id}: Early stopping at iteration {iteration}")
            break

        # Record data
        coverage_history.append(current_coverage)
        energy_history.append(total_energy)
        active_robots_history.append(active_robots)
        iterations.append(iteration)

        # Stop if target coverage is reached
        if current_coverage >= TARGET_COVERAGE:
            print(f"Simulation {sim_id}: Target coverage of {TARGET_COVERAGE*100}% reached at iteration {iteration}")
            break

    # Save results
    results[sim_id] = {
        'iterations': iterations,
        'coverage_history': coverage_history,
        'energy_history': energy_history,
        'active_robots_history': active_robots_history,
        'final_coverage': best_coverage,
        'robots': robots,
        'obstacles': obstacles
    }

def run_simulations(num_simulations=5):
    """Run multiple simulations and collect data."""
    results = {}
    threads = []

    for sim_id in range(num_simulations):
        thread = threading.Thread(target=simulate_pso, args=(sim_id, results))
        threads.append(thread)
        thread.start()
        time.sleep(0.1)  # Small delay to prevent race conditions

    # Wait for all simulations to complete
    for thread in threads:
        thread.join()

    return results

def analyze_results(results):
    """Analyze the results from multiple simulations."""
    num_simulations = len(results)
    avg_final_coverage = np.mean([res['final_coverage'] for res in results.values()])
    avg_iterations = np.mean([len(res['iterations']) for res in results.values()])
    avg_energy = np.mean([res['energy_history'][-1] for res in results.values()])
    avg_active_robots = np.mean([res['active_robots_history'][-1] for res in results.values()])

    print("\n--- Simulation Summary ---")
    print(f"Number of Simulations: {num_simulations}")
    print(f"Average Final Coverage: {avg_final_coverage*100:.2f}%")
    print(f"Average Iterations to Completion: {avg_iterations:.2f}")
    print(f"Average Total Energy Remaining: {avg_energy:.2f}")
    print(f"Average Active Robots at End: {avg_active_robots:.2f}")

    # Plotting average coverage over time
    plt.figure(figsize=(10, 6))
    for sim_id, res in results.items():
        plt.plot(res['iterations'], [c * 100 for c in res['coverage_history']], label=f"Sim {sim_id}")
    plt.title('Coverage Over Time Across Simulations')
    plt.xlabel('Iteration')
    plt.ylabel('Coverage (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run multiple simulations
simulation_results = run_simulations(num_simulations=5)

# Analyze and visualize results
analyze_results(simulation_results)
