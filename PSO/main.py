import numpy as np
import matplotlib.pyplot as plt
from robot import Robot
from environment import Environment

def main():
    # Simulation parameters
    NUM_ROBOTS = 20
    ENV_SIZE = (100, 100)
    NUM_STEPS = 1000

    # Initialize environment
    env = Environment(size=ENV_SIZE, num_obstacles=10, num_survivors=5)

    # Initialize robots
    robots = []
    for i in range(NUM_ROBOTS):
        position = np.random.rand(2) * ENV_SIZE
        robot = Robot(id=i, position=position, env=env)
        robots.append(robot)

    # Simulation loop
    coverage_data = []
    energy_data = []
    survivor_detection_data = []

    for step in range(NUM_STEPS):
        for robot in robots:
            if robot.active:
                robot.update()
        env.update_coverage(robots)
        env.update_stigmergy(robots)

        # Data collection
        coverage = env.calculate_coverage()
        total_energy = sum(robot.energy for robot in robots)
        survivors_detected = sum(robot.survivors_detected for robot in robots)

        coverage_data.append(coverage)
        energy_data.append(total_energy)
        survivor_detection_data.append(survivors_detected)

        # Optional: Visualization
        if step % 100 == 0:
            env.visualize(robots, step)

    # Results
    plt.figure()
    plt.plot(coverage_data)
    plt.xlabel('Time Step')
    plt.ylabel('Coverage (%)')
    plt.title('Coverage Over Time')
    plt.show()

    plt.figure()
    plt.plot(energy_data)
    plt.xlabel('Time Step')
    plt.ylabel('Total Energy')
    plt.title('Total Energy Over Time')
    plt.show()

    plt.figure()
    plt.plot(survivor_detection_data)
    plt.xlabel('Time Step')
    plt.ylabel('Survivors Detected')
    plt.title('Survivor Detection Over Time')
    plt.show()

if __name__ == '__main__':
    main()
