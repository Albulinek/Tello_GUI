import tensorflow as tf
import numpy as np
from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt
from fastdtw import fastdtw

# @tf.custom_gradient
# def dtw_tf(seq1, seq2):
#     # DTW Implementation (adapted from scipy for TensorFlow)
#     seq1 = seq1.numpy()
#     seq2 = seq2.numpy()
    
#     # compute euclidean distances between all points in seq1 and seq2
#     dist = np.array([[euclidean(p1, p2) for p2 in seq2] for p1 in seq1])

#     # initialize cost matrix
#     cost = np.ones((len(seq1) + 1, len(seq2) + 1)) * np.inf
#     cost[0, 0] = 0

#     # compute accumulated cost matrix
#     for i in range(1, len(seq1) + 1):
#         for j in range(1, len(seq2) + 1):
#             cost[i, j] = dist[i - 1, j - 1] + min(
#                 cost[i - 1, j],  # insertion
#                 cost[i, j - 1],  # deletion
#                 cost[i - 1, j - 1],  # match
#             )
    
#     distance = cost[-1, -1]  # final distance is the last element of the cost matrix

#     def grad(upstream):
#         return None, None # we are not training, so no need for gradients

#     return distance, grad

class TestCurve:
    @staticmethod
    def create_circle(center_x=0, center_y=0, radius=1):
        # testing code for equality to calculate validity and weight of gen alg
        # NOTE: test confirmed that weight calculated by this procedure can be used to evaluate fitness of gen alg
        points = np.array([[radius, 0]])

        for degree in range(1, 360, 1):
            radians = np.radians(degree)
            x = center_x + radius * np.cos(radians)
            y = center_y + radius * np.sin(radians)
            points = np.vstack((points, [x, y]))

        return points

    @staticmethod
    def create_spiral(center_x=0, center_y=0, num_turns=2, radius_growth=100):
        theta = np.linspace(0, num_turns * 2 * np.pi, 360)
        radius = radius_growth * theta
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)
        return np.column_stack((x, y))

    @staticmethod
    def create_polynomial(coeffs, x_shift=0, x_scale=1, rotation_angle=0):
        x_range = np.linspace(-1, 1, 100) * x_scale + \
            x_shift   # Shift and scale x-values
        y_values = np.polyval(coeffs, x_range)

        # Rotate the points (same as before)
        theta = np.radians(rotation_angle)
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array(((c, -s), (s, c)))
        points = np.column_stack((x_range, y_values))
        rotated_points = np.dot(points, rotation_matrix)

        return rotated_points

COMMANDS = (
            ["Turn " + str(i) for i in range(-355, 360, 5)]  # Turn commands with degrees
            + ["Forward " + str(i) for i in range(1, 101)]  # Forward commands with distance
            + ["Backward " + str(i) for i in range(1, 101)]  # Backward commands with distance
            + ["Strafe Right " + str(i) for i in range(1, 101)]  # Strafe Right commands with distance
            + ["Strafe Left " + str(i) for i in range(1, 101)]  # Strafe Left commands with distance
        )

class Drone:
    def __init__(self):
        pass


class GeneticAlgorithm:
    def __init__(
        self,
        target_curve,
        population_size=10,
        generations=1000,
        mutation_rate=0.01,
    ):
        self.target_curve = tf.constant(target_curve, dtype=tf.float32)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.commands = (
            ["Turn " + str(i) for i in range(-90, 90, 5)]  # Turn commands with degrees
            + ["Forward " + str(i) for i in range(1, 101)]  # Forward commands with distance
        )

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.randint(
                0, len(self.commands) -1, 100
            )  # Generate indices instead of strings
            population.append(individual)
        return population


    def simulate_drone(self, commands, start_pos=(0, 0), start_angle=0):
        x, y = start_pos
        positions = [(x, y)]
        angle = start_angle

        for command_index in commands:
            # Fetch the command string using the index
            command_str = self.commands[command_index]  

            # Split the command string into parts
            command_parts = command_str.split(" ") 

            # Combine multi-word command types into a single string
            command_type = " ".join(command_parts[:-1]) 

            # Extract the last element as the command value
            command_value = int(command_parts[-1])  

            if command_type == "Turn":
                angle += np.radians(command_value)
            else:
                distance = abs(command_value)
                x += distance * np.cos(angle)
                y += distance * np.sin(angle)
            positions.append((x, y))

        return positions


    def fitness(self, individual):
        positions = np.array(self.simulate_drone(individual))
        dtw_distance, _ = fastdtw(positions, self.target_curve.numpy(), dist=euclidean)

        return 1 / (dtw_distance + 1e-6)  # Convert to maximize fitness

    def selection(self, population, fitness_scores):
        # Stack fitness scores into a 2D tensor with batch_size = 1
        fitness_scores_tensor = tf.stack(fitness_scores)[tf.newaxis]

        # Normalize fitness scores
        fitness_scores_tensor = tf.math.softmax(-fitness_scores_tensor)

        # Sample indices using tf.random.categorical
        selected_indices = tf.random.categorical(
            tf.math.log(fitness_scores_tensor), self.population_size
        )

        # Remove the extra batch dimension and convert to numpy
        selected_indices = tf.squeeze(selected_indices).numpy().astype(int)

        selected = [population[i] for i in selected_indices]
        return selected

    def crossover(self, parent1, parent2, num_points=2):
        crossover_points = np.random.choice(
            len(parent1) - 1, num_points, replace=False
        )  # Choose crossover points without replacement
        crossover_points.sort()  # Ensure points are in ascending order
        crossover_points = [0] + list(crossover_points) + [
            len(parent1)
        ]  # Add start and end points

        child1, child2 = [], []
        for i in range(num_points + 1):
            if i % 2 == 0:  # Alternate segments from parents
                child1.extend(parent1[crossover_points[i] : crossover_points[i + 1]])
                child2.extend(parent2[crossover_points[i] : crossover_points[i + 1]])
            else:
                child1.extend(parent2[crossover_points[i] : crossover_points[i + 1]])
                child2.extend(parent1[crossover_points[i] : crossover_points[i + 1]])
        return child1, child2

    # def mutation(self, individual):
    #     for i in range(len(individual)):
    #         if np.random.rand() < self.mutation_rate:
    #             individual[i] = np.random.randint(0, len(self.commands))
    #     return individual
    def mutation(self, individual, mutation_rate=0.01):
        for i in range(len(individual)):
            if np.random.rand() < mutation_rate:
                individual[i] = np.random.randint(0, len(self.commands))
            # Additional Mutation Step: Small Random Perturbations
            if np.random.rand() < mutation_rate / 2:
                individual[i] += np.random.randint(-5, 6)
        return individual

    def best_individual(self, population):
        fitness_scores = [self.fitness(individual)
                          for individual in population]
        best_individual_idx = np.argmin(fitness_scores)
        best_individual = population[best_individual_idx]
        best_fitness = np.min(fitness_scores)
        best_path = self.simulate_drone(best_individual)
        return best_individual, best_fitness, best_path

    def evolve(self):
        population = self.initialize_population()
        for generation in range(self.generations):
            fitness_scores = [self.fitness(individual) for individual in population]

            selected = self.selection(population, fitness_scores)

            next_generation = []
            
            for i in range(0, self.population_size, 2):
                parent1, parent2 = selected[i], selected[i + 1]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                next_generation.extend([child1, child2])

            # Elitism (keep the best individual)
            best_individual_idx = np.argmax([self.fitness(ind) for ind in next_generation])
            best_individual = population[best_individual_idx]
            next_generation[-1] = best_individual # replace last with current best

            population = next_generation

            if generation % 10 == 0:
                # Pass the updated population to best_individual
                best_individual, best_fitness, best_path = self.best_individual(population)  
                print(f"Generation {generation}: Best Fitness = {best_fitness}")

                # Print the commands of the best individual of this generation
                # best_commands = [self.commands[i] for i in best_individual]
                # print(f"Best individual: {best_commands}")
                # print(len(best_commands))
                # print(f"Best path: {best_path}")
                # print(len(best_path))

                # Plot and save the figure for this generation
                plt.plot(test_data[:, 0], test_data[:, 1], "b-", label="Target Curve")
                plt.plot(
                    [x[0] for x in best_path],
                    [x[1] for x in best_path],
                    "g-",
                    label="Best Path",
                )
                # Plot individual paths for the selected drones
                for individual in selected:  
                    individual_path = self.simulate_drone(individual)
                    plt.plot(
                        [x[0] for x in individual_path],
                        [x[1] for x in individual_path],
                        "r-",
                        alpha=0.2,  # Make the lines semi-transparent for better visualization
                    )
                plt.legend()
                plt.show()
                plt.savefig(f"test_gen{generation}.png")
                plt.clf()

        return population, best_individual, best_path


if __name__ == "__main__":
    test_data = TestCurve.create_spiral(num_turns=0.9, radius_growth=100)
    ga = GeneticAlgorithm(test_data)
    population, best_individual, best_path = ga.evolve()

    plt.plot(test_data[:, 0], test_data[:, 1], "b-", label="Target Curve")
    plt.plot(
        [x[0] for x in best_path],
        [x[1] for x in best_path],
        "r-",
        label="Best Path",
    )
    plt.legend()
    plt.show()
    plt.savefig(f'test.png')

    # Print the commands of the best individual
    best_commands = [ga.commands[i] for i in best_individual]
    print(best_commands)
