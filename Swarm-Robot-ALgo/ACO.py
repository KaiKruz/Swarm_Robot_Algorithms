import numpy as np
import matplotlib.pyplot as plt

# City coordinates (x, y)
cities = np.array([
    [0, 0],   # City 0
    [2, 3],   # City 1
    [5, 4],   # City 2
    [1, 1],   # City 3
    [7, 3],   # City 4
    [6, 1],   # City 5
])

n_cities = len(cities)

# Calculate the distance matrix
def calculate_distance_matrix(cities):
    n = len(cities)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])
    return dist_matrix

dist_matrix = calculate_distance_matrix(cities)

# Parameters for the ACO
n_ants = 10
n_iterations = 100
alpha = 1.0  # Pheromone importance
beta = 2.0   # Distance importance
evaporation_rate = 0.5
pheromone_constant = 1.0
initial_pheromone = 1.0

# Initialize pheromone levels
pheromone_matrix = np.ones((n_cities, n_cities)) * initial_pheromone

# Function to calculate the probability of moving to the next city
def calculate_transition_probabilities(pheromone_matrix, dist_matrix, alpha, beta, current_city, visited):
    n = len(pheromone_matrix)
    probabilities = np.zeros(n)
    for i in range(n):
        if i not in visited:
            probabilities[i] = (pheromone_matrix[current_city, i] ** alpha) * ((1.0 / dist_matrix[current_city, i]) ** beta)
    return probabilities / np.sum(probabilities)

# Function to run the ACO algorithm
def ant_colony_optimization(dist_matrix, pheromone_matrix, alpha, beta, evaporation_rate, pheromone_constant, n_ants, n_iterations):
    n_cities = len(dist_matrix)
    best_route = None
    best_distance = float('inf')

    for iteration in range(n_iterations):
        all_routes = []
        all_distances = []

        for ant in range(n_ants):
            route = []
            visited = set()
            current_city = np.random.randint(n_cities)
            route.append(current_city)
            visited.add(current_city)

            while len(visited) < n_cities:
                probabilities = calculate_transition_probabilities(pheromone_matrix, dist_matrix, alpha, beta, current_city, visited)
                next_city = np.random.choice(range(n_cities), p=probabilities)
                route.append(next_city)
                visited.add(next_city)
                current_city = next_city  # Update the current city

            route.append(route[0])  # Return to start
            all_routes.append(route)
            distance = np.sum([dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)])
            all_distances.append(distance)

            if distance < best_distance:
                best_distance = distance
                best_route = route

        # Update pheromone matrix
        pheromone_matrix *= (1 - evaporation_rate)
        for route, distance in zip(all_routes, all_distances):
            for i in range(len(route) - 1):
                pheromone_matrix[route[i], route[i + 1]] += pheromone_constant / distance

        print(f"Iteration {iteration + 1}, Best Distance: {best_distance}")

    return best_route, best_distance

# Run the ACO algorithm
best_route, best_distance = ant_colony_optimization(dist_matrix, pheromone_matrix, alpha, beta, evaporation_rate, pheromone_constant, n_ants, n_iterations)

print("Best route found:", best_route)
print("Best distance:", best_distance)

# Plot the best route
plt.figure(figsize=(8, 6))
for i in range(len(best_route) - 1):
    city1 = best_route[i]
    city2 = best_route[i + 1]
    plt.plot([cities[city1, 0], cities[city2, 0]], [cities[city1, 1], cities[city2, 1]], 'bo-')
plt.scatter(cities[:, 0], cities[:, 1], color='red')
plt.title('Best Route Found by ACO')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid()
plt.show()
