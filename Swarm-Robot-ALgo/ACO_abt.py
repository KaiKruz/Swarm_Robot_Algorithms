import numpy as np
import matplotlib.pyplot as plt
import time

# Define the city coordinates (x, y)
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
evaporation_rate = 0.5
pheromone_constant = 1.0
initial_pheromone = 1.0

# Initialize pheromone levels
pheromone_matrix = np.ones((n_cities, n_cities)) * initial_pheromone

# Function to calculate the probability of moving to the next city
def calculate_transition_probabilities(pheromone_matrix, dist_matrix, alpha, beta, visited, current_city):
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
                probabilities = calculate_transition_probabilities(pheromone_matrix, dist_matrix, alpha, beta, visited, current_city)
                next_city = np.random.choice(range(n_cities), p=probabilities)
                route.append(next_city)
                visited.add(next_city)
                current_city = next_city

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

    return best_route, best_distance

# Function to test different alpha and beta values
def test_parameters(alpha_values, beta_values):
    results = {}
    times = {}
    for alpha in alpha_values:
        for beta in beta_values:
            print(f"Testing with alpha={alpha}, beta={beta}")
            pheromone_matrix = np.ones((n_cities, n_cities)) * initial_pheromone
            start_time = time.time()
            best_route, best_distance = ant_colony_optimization(dist_matrix, pheromone_matrix, alpha, beta, evaporation_rate, pheromone_constant, n_ants, n_iterations)
            end_time = time.time()
            elapsed_time = end_time - start_time
            results[(alpha, beta)] = best_distance
            times[(alpha, beta)] = elapsed_time
            print(f"Best distance for alpha={alpha}, beta={beta}: {best_distance}")
            print(f"Time taken for alpha={alpha}, beta={beta}: {elapsed_time:.2f} seconds\n")
    return results, times

# Define the alpha and beta values to test (including large values)
alpha_values = [0.5, 1.0, 2.0, 5.0, 10.0]
beta_values = [1.0, 2.0, 5.0, 10.0, 20.0]

# Run the tests
results, times = test_parameters(alpha_values, beta_values)

# Plotting the results
def plot_results(results, times, alpha_values, beta_values):
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    alpha_beta_pairs = list(results.keys())
    distances = [results[pair] for pair in alpha_beta_pairs]
    time_taken = [times[pair] for pair in alpha_beta_pairs]
    alpha_beta_labels = [f"α={pair[0]}, β={pair[1]}" for pair in alpha_beta_pairs]

    ax1.bar(alpha_beta_labels, distances, color='g', alpha=0.6, label='Best Distance')
    ax2.plot(alpha_beta_labels, time_taken, color='b', marker='o', label='Time Taken')

    ax1.set_xlabel('Alpha and Beta Values')
    ax1.set_ylabel('Best Distance', color='g')
    ax2.set_ylabel('Time Taken (s)', color='b')

    plt.title('ACO Results: Best Distance and Time Taken for Different Alpha and Beta Values')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot the results
plot_results(results, times, alpha_values, beta_values)

