import numpy as np

# Distance matrix representing the distances between each pair of cities
distances = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

# Number of cities and ants
num_cities = len(distances)
num_ants = 4
num_iterations = 100
alpha = 1  # Influence of pheromone
beta = 2   # Influence of distance
evaporation_rate = 0.5  # Rate of pheromone evaporation
pheromone_intensity = 1.0  # Pheromone intensity

# Initialize pheromone levels on each path
pheromones = np.ones((num_cities, num_cities))

def fuzzy_heuristic(distance):
    """Fuzzy logic heuristic for determining attractiveness."""
    if distance == 0:
        return 0
    elif distance <= 10:
        return 1
    elif distance <= 20:
        return 0.8
    else:
        return 0.5

def choose_next_city(pheromones, distances, visited, current_city):
    probabilities = []
    for city in range(num_cities):
        if city not in visited:
            pheromone_level = pheromones[current_city, city] ** alpha
            heuristic_value = (1 / distances[current_city, city]) ** beta
            probability = pheromone_level * heuristic_value
            probabilities.append(probability)
        else:
            probabilities.append(0)
    probabilities = np.array(probabilities)
    return np.random.choice(range(num_cities), p=probabilities / probabilities.sum())

def update_pheromones(pheromones, all_routes, all_distances, evaporation_rate):
    pheromones *= (1 - evaporation_rate)
    for route, distance in zip(all_routes, all_distances):
        for i in range(len(route) - 1):
            pheromones[route[i], route[i + 1]] += pheromone_intensity / distance
        pheromones[route[-1], route[0]] += pheromone_intensity / distance

def ant_colony_optimization():
    best_distance = float('inf')
    best_route = None
    
    for iteration in range(num_iterations):
        all_routes = []
        all_distances = []
        
        for ant in range(num_ants):
            visited = []
            current_city = np.random.randint(0, num_cities)
            visited.append(current_city)
            
            for _ in range(num_cities - 1):
                next_city = choose_next_city(pheromones, distances, visited, current_city)
                visited.append(next_city)
                current_city = next_city
            
            route_distance = sum(distances[visited[i], visited[i + 1]] for i in range(num_cities - 1))
            route_distance += distances[visited[-1], visited[0]]  # Return to start
            all_routes.append(visited)
            all_distances.append(route_distance)
            
            if route_distance < best_distance:
                best_distance = route_distance
                best_route = visited
        
        update_pheromones(pheromones, all_routes, all_distances, evaporation_rate)
        print(f"Iteration {iteration + 1}: Best distance = {best_distance}, Route = {best_route}")
    
    return best_route, best_distance

best_route, best_distance = ant_colony_optimization()
print(f"Best route found: {best_route} with distance {best_distance}")
