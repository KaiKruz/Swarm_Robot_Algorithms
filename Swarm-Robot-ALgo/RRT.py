import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import KDTree  # KD-Tree for faster nearest node search

# Optimized RRT Algorithm with KD-Tree for fast nearest node search
class RRT:
    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=1.0, goal_sample_rate=0.1, max_iter=300):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = [self.start]
        self.kd_tree = KDTree([(start[0], start[1])])

    def plan(self, animation=True):
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_node = self.node_list[self.get_nearest_node_index(rnd_node)]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if not self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)
                self.update_kd_tree(new_node)

                if self.calc_dist_to_goal(new_node.x, new_node.y) <= self.expand_dis:
                    final_node = self.steer(new_node, self.goal, self.expand_dis)
                    if not self.check_collision(final_node, self.obstacle_list):
                        return self.generate_final_course(len(self.node_list) - 1)
            
            if animation and i % 10 == 0:  # Only update visualization every 10 steps
                self.draw_graph(rnd_node)

        return None  # No path found

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(from_node, to_node)

        extend_length = min(extend_length, d)
        n_expand = int(extend_length / self.expand_dis)

        for _ in range(n_expand):
            new_node.x += self.expand_dis * np.cos(theta)
            new_node.y += self.expand_dis * np.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        new_node.parent = from_node
        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path[::-1]

    def calc_dist_to_goal(self, x, y):
        return np.hypot(x - self.goal.x, y - self.goal.y)

    def get_random_node(self):
        if random.random() > self.goal_sample_rate:
            return Node(random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand))
        else:
            return Node(self.goal.x, self.goal.y)

    def get_nearest_node_index(self, rnd_node):
        _, idx = self.kd_tree.query([rnd_node.x, rnd_node.y])
        return idx

    def update_kd_tree(self, new_node):
        points = [(node.x, node.y) for node in self.node_list]
        self.kd_tree = KDTree(points)  # Update the KD-Tree with new points

    def check_collision(self, node, obstacle_list):
        for (ox, oy, size) in obstacle_list:
            if any((ox - x) ** 2 + (oy - y) ** 2 <= size ** 2 for x, y in zip(node.path_x, node.path_y)):
                return True  # Collision detected, return early
        return False

    def calc_distance_and_angle(self, from_node, to_node):
        dx, dy = to_node.x - from_node.x, to_node.y - from_node.y
        return np.hypot(dx, dy), np.arctan2(dy, dx)

    def draw_graph(self, rnd_node=None):
        plt.clf()
        if rnd_node:
            plt.plot(rnd_node.x, rnd_node.y, "^k")  # Random point
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")  # Tree edges

        # Draw obstacles
        for (ox, oy, size) in self.obstacle_list:
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start.x, self.start.y, "xr")  # Start point
        plt.plot(self.goal.x, self.goal.y, "xg")    # Goal point
        plt.grid(True)
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.pause(0.001)  # Faster updates

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.path_x = []
        self.path_y = []

def main():
    print("Optimized RRT Path Planning with KD-Tree")

    # Define obstacles (circle format: (x, y, radius))
    obstacle_list = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2)
    ]

    # Define start and goal
    start = [0, 0]
    goal = [6, 10]
    rand_area = [-2, 15]

    # Set parameters for RRT
    rrt = RRT(start=start, goal=goal, rand_area=rand_area, obstacle_list=obstacle_list, expand_dis=0.5, goal_sample_rate=0.2)

    # Plan path
    path = rrt.plan(animation=True)

    # Check if path was found
    if path is None:
        print("No path found")
    else:
        print("Path found!")
        rrt.draw_graph()  # Draw final path
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')  # Plot the found path
        plt.show()

if __name__ == '__main__':
    main()
