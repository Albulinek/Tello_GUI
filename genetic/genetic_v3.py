from typing import List, Tuple
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Tello Parameters
MaxForward = int  
MinForward = int
MaxYaw = int      
MinYaw = int      

max_forward: MaxForward = 100
min_forward: MinForward = 5
max_yaw: MaxYaw = 90
min_yaw: MinYaw = -90

# Spiral Parameters
radius: float = 100       
num_points: int = 50

# Genetic Algorithm Parameters
population_size: int = 100
num_generations: int = 100
mutation_rate: float = 0.1

# Create Target Spiral
theta: np.ndarray = np.linspace(0, 2 * np.pi, num_points)
x_target: np.ndarray = radius * np.cos(theta)
y_target: np.ndarray = radius * np.sin(theta)
target_spiral: np.ndarray = np.column_stack((x_target, y_target))

# DNA Class
class DNA:
    def __init__(self, genes: List[Tuple[float, float]]):
        self.genes: List[Tuple[float, float]] = genes

    def __repr__(self):
        return f"DNA(genes={self.genes})"

# TelloDrone Class
class TelloDrone:
    def __init__(self, dna: DNA = None):
        self.dna: DNA = dna if dna else DNA(self.generate_random_genes())
        self.fitness: float = self.calculate_fitness()

    def generate_random_genes(self) -> List[Tuple[float, float]]:
        return [(random.uniform(min_forward, max_forward), random.uniform(min_yaw, max_yaw)) for _ in range(num_points - 1)]

    def calculate_fitness(self) -> float:
        x, y = 0.0, 0.0
        drone_path = [(x, y)]
        for forward, yaw in self.dna.genes:
            x += forward * np.cos(np.radians(yaw))
            y += forward * np.sin(np.radians(yaw))
            drone_path.append((x, y))
        drone_path = np.array(drone_path)
        distances = np.linalg.norm(drone_path - target_spiral, axis=1)
        return -np.sum(distances)  # Minimize the total distance

    def __repr__(self):
        return f"TelloDrone(dna={self.dna}, fitness={self.fitness:.2f})"

# Genetic Operators
def select_parents(population: List[TelloDrone], num_parents: int = 2) -> List[TelloDrone]:
    fitness_scores = [drone.fitness for drone in population]
    min_fitness = min(fitness_scores)
    if min_fitness < 0:
        fitness_scores = [score - min_fitness + 1e-10 for score in fitness_scores]
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    probabilities[-1] = 1 - sum(probabilities[:-1])

    # Corrected np.random.choice call
    parent_indices = np.random.choice(len(population), size=num_parents, replace=False, p=probabilities)
    parents = [population[i] for i in parent_indices]
    return parents


def crossover(parents: List[TelloDrone]) -> List[TelloDrone]:
    crossover_point = random.randint(1, num_points - 1)
    child1_dna = DNA(parents[0].dna.genes[:crossover_point] + parents[1].dna.genes[crossover_point:])
    child2_dna = DNA(parents[1].dna.genes[:crossover_point] + parents[0].dna.genes[crossover_point:])
    return [TelloDrone(child1_dna), TelloDrone(child2_dna)]


def mutate(offspring: List[TelloDrone], mutation_rate: float) -> List[TelloDrone]:
    for child in offspring:
        for i in range(len(child.dna.genes)):
            if random.random() < mutation_rate:
                child.dna.genes[i] = (random.uniform(min_forward, max_forward), random.uniform(min_yaw, max_yaw))
        child.fitness = child.calculate_fitness()  # Recalculate fitness after mutation
    return offspring
    
# Main Genetic Algorithm Loop
population: List[TelloDrone] = [TelloDrone() for _ in range(population_size)]

for generation in range(num_generations):
    population = sorted(population, key=lambda drone: drone.fitness, reverse=True)
    fitness_scores = [drone.fitness for drone in population]
    parents = select_parents(population, fitness_scores)
    offspring = crossover(parents)
    offspring = mutate(offspring, mutation_rate)
    population = create_new_generation(parents, offspring)

best_drone = population[0]
print("Best Drone:", best_drone)

# Visualize Results
x, y = 0.0, 0.0
drone_path = [(x, y)]
for forward, yaw in best_drone.dna.genes:
    x += forward * np.cos(np.radians(yaw))
    y += forward * np.sin(np.radians(yaw))
    drone_path.append((x, y))

plt.figure(figsize=(8, 8))
plt.plot(x_target, y_target, label='Target Spiral')
plt.plot(*zip(*drone_path), label='Drone Path', marker='o', linestyle='--')
plt.legend()
plt.title('Spiral Curve Fitting with Tello Drone')
plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.grid(True)
plt.show()
