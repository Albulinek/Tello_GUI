import tensorflow as tf

# import shapely
import numpy as np
import numpy.typing as npt


import matplotlib.pyplot as plt
# from fastdtw import fastdtw # type: ignore
from tslearn.metrics import dtw

from labellines import labelLine, labelLines

import shapely.geometry as sg # type: ignore

class TestCurve:
    # @staticmethod
    # def create_circle(center_x=0, center_y=0, radius=1):
    #     # testing code for equality to calculate validity and weight of gen alg
    #     # NOTE: test confirmed that weight calculated by this procedure can be used to evaluate fitness of gen alg
    #     points = np.array([[radius, 0]])

    #     for degree in range(1, 360, 1):
    #         radians = np.radians(degree)
    #         x = center_x + radius * np.cos(radians)
    #         y = center_y + radius * np.sin(radians)
    #         points = np.vstack((points, [x, y]))

    #     return points

    @staticmethod
    def create_sinusoid(amplitude: float = 25.0, length: float = 200.0, phase_shift: float = 0.0, num_points: int = 100) -> np.ndarray:
        """Creates a stretched sinusoid curve starting at (0, 0) with specified amplitude and length."""

        # Calculate frequency to achieve the desired length
        frequency = 2*np.pi / length  

        # Calculate the x-shift to make the sinusoid start at (0, 0)
        x_shift = -phase_shift / frequency

        # Calculate x values spanning the desired length
        x = np.linspace(x_shift, length + x_shift, num_points) 

        y = amplitude * np.sin(frequency * x, dtype=np.float32)  
        return np.column_stack((x, y))

    @staticmethod
    def create_spiral(center_x: np.float32=np.float32(0), center_y: np.float32=np.float32(0), num_turns:np.float32=np.float32(2), radius_growth: np.float32=np.float32(100)) -> npt.NDArray[np.float32]:
        theta = np.linspace(0, num_turns * 2 * np.pi, 360, dtype=np.float32)
        radius = radius_growth * theta
        x = center_x + radius * np.cos(theta, dtype=np.float32)
        y = center_y + radius * np.sin(theta, dtype=np.float32)
        return np.column_stack((x, y))
    
    @staticmethod
    def create_broken_line(points: list[tuple[float, float]]) -> npt.NDArray[np.float32]:
        line_segments = []
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            num_points_per_segment = 10  # Adjust for smoother lines
            segment_x = np.linspace(x1, x2, num_points_per_segment, dtype=np.float32)
            segment_y = np.linspace(y1, y2, num_points_per_segment, dtype=np.float32)
            line_segments.append(np.column_stack((segment_x, segment_y)))
        return np.vstack(line_segments)

    # @staticmethod
    # def create_polynomial(coeffs, x_shift=0, x_scale=1, rotation_angle=0):
    #     x_range = np.linspace(-1, 1, 100) * x_scale + \
    #         x_shift   # Shift and scale x-values
    #     y_values = np.polyval(coeffs, x_range)

    #     # Rotate the points (same as before)
    #     theta = np.radians(rotation_angle)
    #     c, s = np.cos(theta), np.sin(theta)
    #     rotation_matrix = np.array(((c, -s), (s, c)))
    #     points = np.column_stack((x_range, y_values))
    #     rotated_points = np.dot(points, rotation_matrix)

    #     return rotated_points

    @staticmethod
    def create_archimedean_spiral(a: float = 0, b: float = 1, num_points: int = 100) -> np.ndarray:
        """Creates an Archimedean spiral."""
        theta = np.linspace(0, 4 * np.pi, num_points)  
        x = (a + b * theta) * np.cos(theta, dtype=np.float32)
        y = (a + b * theta) * np.sin(theta, dtype=np.float32)
        return np.column_stack((x, y))

    @staticmethod
    def create_cycloid(r: float = 5, num_points: int = 100) -> np.ndarray:
        """Creates a cycloid curve."""
        t = np.linspace(0, 4 * np.pi, num_points)
        x = r * (t - np.sin(t, dtype=np.float32))
        y = r * (1 - np.cos(t, dtype=np.float32))
        return np.column_stack((x, y))

    @staticmethod
    def create_lissajous_curve(A: float = 10, B: float = 8, a: int = 3, b: int = 2, delta: float = np.pi / 2, num_points: int = 100) -> np.ndarray:
        """Creates a Lissajous curve."""
        t = np.linspace(0, 2 * np.pi, num_points)
        x = A * np.sin(a * t + delta, dtype=np.float32)
        y = B * np.sin(b * t, dtype=np.float32)
        return np.column_stack((x, y))





COMMAND_TYPES = ["Turn", "Forward"]  # Define the types of commands
MAX_TURN_ANGLE = 30  # Maximum turn angle in degrees
MAX_FORWARD_DISTANCE = 20  # Maximum forward distance # NOTE: Hardcoded, but should be dynamic

MUTATION_RATE = np.float32(0.05)
MUTATION_PROBABILITY = np.float32(0.96) # You can adjust this
BEST_RATIO = 0.05
CURVE_RATIO = 1.05

class Drone:
    # can we use tensors instead ndarray?
    def __init__(self, commands: npt.NDArray[np.uint32], target_curve: tf.Tensor):
        self.commands = commands
        self.x, self.y = np.float32(0), np.float32(0)  # Initial position # TODO: ensure the position match start of the curve (function|method)

        # Calculate initial angle based on target curve
        start_vector = target_curve[1] - target_curve[0]
        self.angle = np.arctan2(start_vector[1], start_vector[0])

        self.fitness = np.float32(0)
        self.dtw = np.float32(0)
        self.target_curve = target_curve

    def simulate(self) -> list[tuple[np.float32, np.float32]]:
        x, y = self.x, self.y
        angle = self.angle
        positions = [(x, y)]
        for command_type, command_value in self.commands:
            if command_type == "Turn":
                angle += np.radians(command_value)
            else:
                distance = command_value
                x += distance * np.cos(angle)
                y += distance * np.sin(angle)
                positions.append((x, y))

        return positions
    
    def calculate_fitness(self) -> np.float32:
        positions = np.array(self.simulate())

        # Intersection Check
        if self.detect_self_intersection(positions):
            self.fitness = np.float32(0)  # Intersection: Very low fitness
            return self.fitness
        
        self.dtw = np.float32(dtw(positions, self.target_curve))
        self.fitness = 1 / self.dtw
        return self.fitness
    

    def detect_self_intersection(self, flown_path: npt.NDArray[np.float32]) -> bool:
        """Detects self-intersections within a flown path."""

        # Convert path to Shapely LineString
        flown_line = sg.LineString(flown_path)

        # Method 1: Geometric Self-Intersection
        if not flown_line.is_simple:  # A non-simple LineString has self-intersections
            return True

        return False

class Generation:
    def __init__(self, drones: list[Drone]):
        self.drones = drones
        # self.fitness_scores: list[np.float32] = []

    def best_drone(self) -> tuple[Drone, np.float32, list[tuple[np.float32, np.float32]]]:
        fitness_scores = [drone.calculate_fitness() for drone in self.drones]
        best_idx = np.argmax(fitness_scores)

        best_individual = self.drones[best_idx]
        best_fitness = np.max(fitness_scores)
        best_path = best_individual.simulate()
        return best_individual, best_fitness, best_path


class GeneticAlgorithm:
    def __init__(
        self,
        target_curve: npt.NDArray[np.float32],
        population_size: int=200,
        number_of_generations: int=2000,
    ):
        self.target_curve = tf.constant(target_curve, dtype=tf.float32) # TODO: Convert to tensor, why?
        self.population_size = population_size
        self.number_of_generations = number_of_generations
        self.gene_size = int(len(target_curve) * CURVE_RATIO)

        
        self.forward_distance = int(np.ceil((self.calculate_curve_length(target_curve) * 1.1 * CURVE_RATIO) / (self.gene_size ))) # Adding smoothness 5%

        self.current_generation: Generation
    
    def calculate_curve_length(self, curve: np.ndarray) -> float:
        """Calculates the approximate length of a curve represented by an array of points."""
        distances = np.linalg.norm(curve[1:] - curve[:-1], axis=1)  # Calculate distances between consecutive points
        total_length = np.sum(distances)  # Sum the distances to get total length
        return total_length

    def initialize_population(self) -> list[Drone]:
        population = []
        for _ in range(self.population_size):
            individual_cmd = np.array(
                [
                    (cmd_type, np.random.randint(- MAX_TURN_ANGLE -1, MAX_TURN_ANGLE + 1)) if cmd_type == "Turn" else (cmd_type, np.random.randint(1, self.forward_distance + 1))
                    for cmd_type in np.random.choice(COMMAND_TYPES, size=self.gene_size)
                ],
                dtype=[('f0', np.str_, 16), ('f1', np.int32)],
            )
            population.append(Drone(individual_cmd, self.target_curve))
        return population


    # def fitness(self, individual: Drone) -> np.float32:
    #     positions = np.array(individual.simulate())
    #     dtw_distance = dtw(positions, self.target_curve)
    #     return 1 / dtw_distance
        
    def get_dtw(self, individual: Drone) -> np.float32:
        positions = np.array(individual.simulate())
        dtw_distance = dtw(positions, self.target_curve)
        return dtw_distance


    def selection(self, population: list[Drone]) -> list[Drone]:
        fitness_scores = [drone.fitness for drone in population]

        # Sort drones based on fitness (descending order)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]

        # Calculate the number of parents to select based on BEST_RATIO
        num_parents_to_select = int(BEST_RATIO * self.population_size)

        # Select the top parents
        selected = sorted_population[:num_parents_to_select]

        # Tournament selection
        tournament_size = 3  # You can adjust this
        tournament_participants = np.random.choice(sorted_population, size=tournament_size, replace=False)
        tournament_winner = max(tournament_participants, key=lambda drone: drone.fitness)
        selected.append(tournament_winner)

        # Random selection from less fit individuals
        less_fit_population = sorted_population[num_parents_to_select:]
        random_less_fit_individual = np.random.choice(less_fit_population)
        selected.append(random_less_fit_individual)

        return selected



    # def crossover(self, parent1: Drone, parent2: Drone) -> Drone:
    #     new_genes: npt.NDArray[np.uint32] = np.empty((0,2), dtype=[('f0', np.str_, 16), ('f1', np.int32)])
    #     # Select a random mid
    #     mid = np.floor(np.random.randint(len(parent1.commands)))

    #     for i in range(0, len(parent1.commands), 1):
    #         if (i > mid):
    #             new_genes = np.append(new_genes, parent1.commands[i])
    #         else:
    #             new_genes = np.append(new_genes, parent2.commands[i])
    #     return Drone(new_genes, self.target_curve)

    def crossover(self, parent1: Drone, parent2: Drone) -> Drone:
        # Initialize an empty array for the new genes
        new_genes = np.empty_like(parent1.commands)  
        
        # Create a mask with random boolean values
        gene_mask = np.random.rand(len(parent1.commands)) < 0.5 
        
        # Use the mask to select genes from parents
        new_genes[gene_mask] = parent1.commands[gene_mask]
        new_genes[~gene_mask] = parent2.commands[~gene_mask]
        
        return Drone(new_genes, self.target_curve)


    def mutation(self, individual: Drone, dynamic_mutation_rate: np.float32 = MUTATION_RATE) -> Drone:
        for i in range(len(individual.commands)):
            if np.random.random_sample() < dynamic_mutation_rate:
                cmd_type, cmd_value = individual.commands[i]

                cmd_type = np.random.choice(COMMAND_TYPES)
                if cmd_type == 'Turn':
                    cmd_value = np.random.randint(- MAX_TURN_ANGLE -1, MAX_TURN_ANGLE)
                elif cmd_type == 'Forward':
                    cmd_value = np.random.randint(1, self.forward_distance + 1)

                individual.commands[i] = (cmd_type, cmd_value)
        return individual

    def evolve(self) -> tuple[Drone, np.float32, list[tuple[np.float32, np.float32]]]:
        population = self.initialize_population()
        generations: list[Generation] = []

        for generation_num in range(self.number_of_generations):
            drones = [drone for drone in population]
            generation = Generation(drones)
            generations.append(generation)

            # Elitism
            best_drone, best_fitness, best_path = generation.best_drone()

            # fitness_scores = generation.fitness_scores
            selected = self.selection(population)

            self.current_generation = generation

            next_generation: list[Drone] = [best_drone]

            while len(next_generation) < self.population_size:
                parent1, parent2 = np.random.choice(selected, size=2, replace=False)
                child = self.crossover(parent1, parent2)

                # Mutate only a fraction of the next generation
                if np.random.rand() < MUTATION_PROBABILITY:
                    mut_child = self.mutation(child)
                    
                    next_generation.append(mut_child)
                else:
                    next_generation.append(child)

            population = next_generation

            if generation_num % 10 == 0:
                sgtw= best_drone.dtw
                print(f"Generation {generation_num}: Best Fitness = {best_fitness}")
                self.plot_generation(generation, selected, best_path, generation_num, best_fitness, sgtw)

        # Return the best drone and its fitness from the final generation
        best_drone, best_fitness, best_path = generations[-1].best_drone()
        return best_drone, best_fitness, best_path
    
    def plot_generation(self, generation: Generation, selected: list[Drone], best_path: list[tuple[np.float32, np.float32]], generation_num: int, best_fitness: np.float32, sgtw: np.float32) -> None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111) 

        ax.plot(test_data[:, 0], test_data[:, 1], "b-", label="Target Curve")
        for drone in generation.drones:
            individual_path = drone.simulate()
            plt.plot(
                [x[0] for x in individual_path],
                [x[1] for x in individual_path],
                "r-",
                alpha=0.1, 
            )
        
        for drone in selected:
            individual_path = drone.simulate()
            ax.plot(
                [x[0] for x in individual_path],
                [x[1] for x in individual_path],
                "--", color='orange', label=f'{drone.fitness:.6f}',
            )

        ax.plot(
            [x[0] for x in best_path],
            [x[1] for x in best_path],
            "g-",
            label="Best Path",
        )
        # labelLines(ax.get_lines(), align=False, fontsize=4) # Labels for testing
        plt.title(f"Generation {generation_num}, Best Fitness: {best_fitness:.6f}, DTW: {sgtw:.1f}")
        # plt.legend(["Best Path", "Target Curve"])
        plt.show()
        plt.savefig(f"test_gen{generation_num}.png")
        plt.close()
        plt.clf()


if __name__ == "__main__":
    # test_data = TestCurve.create_spiral(num_turns=np.float32(0.7), radius_growth=np.float32(50))
    # Define the points for the broken line
    points = [(0, 0), (50, 50), (100, 0), (150, 50), (200, 0)]  # Example broken line
    # test_data = TestCurve.create_broken_line(points)

    # Create a sinusoid with amplitude 25 and length 200
    test_data = TestCurve.create_sinusoid(amplitude=25, length=200) 


    # Create an Archimedean spiral with 200 points
    # test_data = TestCurve.create_archimedean_spiral(num_points=200)

    # Create a cycloid with radius 3 and 150 points
    # test_data = TestCurve.create_cycloid(r=3, num_points=150)

    # Create a Lissajous curve with specific parameters and 300 points
    # test_data = TestCurve.create_lissajous_curve(A=12, B=6, a=5, b=2, delta=0, num_points=300)

    ga = GeneticAlgorithm(test_data)

    best_drone, best_fitness, best_path = ga.evolve()

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
    # best_commands = [ga.commands[i] for i in best_drone.commands]
    print(best_drone.commands)
