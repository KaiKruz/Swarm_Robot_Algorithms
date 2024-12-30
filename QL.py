import numpy as np

# Parameters for Q-learning
learning_rate = 0.1
discount_factor = 0.9
initial_epsilon = 0.1  # Initial exploration factor
epsilon_decay = 0.99  # Decay factor for epsilon
min_epsilon = 0.01  # Minimum epsilon
num_iterations = 1000  # Number of episodes
num_cities = 10  # Example with 10 cities

# Randomly generate distances between cities (10x10 matrix)
np.random.seed(42)
distances = np.random.randint(10, 100, size=(num_cities, num_cities))

# Q-table initialization
q_table = np.zeros((num_cities, num_cities))

def choose_next_city_qlearning(current_city, visited, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        # Exploration: choose a random unvisited city
        return np.random.choice([city for city in range(num_cities) if city not in visited])
    else:
        # Exploitation: choose the best city according to Q-table
        valid_q_values = q_table[current_city].copy()
        valid_q_values[visited] = -np.inf  # Set already visited cities to -inf
        return np.argmax(valid_q_values)

def update_q_table(current_city, next_city, reward):
    best_future_q = np.max(q_table[next_city])
    q_table[current_city, next_city] = (1 - learning_rate) * q_table[current_city, next_city] + \
                                       learning_rate * (reward + discount_factor * best_future_q)

def q_learning_optimization():
    epsilon = initial_epsilon
    best_distance = float('inf')
    best_route = None
    
    for episode in range(num_iterations):
        visited = []
        current_city = np.random.randint(0, num_cities)
        visited.append(current_city)
        route_distance = 0
        
        while len(visited) < num_cities:
            next_city = choose_next_city_qlearning(current_city, visited, epsilon)
            visited.append(next_city)
            route_distance += distances[current_city, next_city]
            update_q_table(current_city, next_city, -distances[current_city, next_city])
            current_city = next_city
        
        # Return to start
        route_distance += distances[visited[-1], visited[0]]
        update_q_table(current_city, visited[0], -distances[current_city, visited[0]])
        
        # Epsilon decay
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Update best distance and route
        if route_distance < best_distance:
            best_distance = route_distance
            best_route = visited
        
        if episode % 100 == 0:
            print(f"Episode {episode + 1}: Best distance = {best_distance}, Route = {best_route}")
    
    return best_route, best_distance

# Run the Q-learning optimization
best_route, best_distance = q_learning_optimization()
print(f"Best route found using Q-learning: {best_route} with distance {best_distance}")
