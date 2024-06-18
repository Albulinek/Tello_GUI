import numpy
import random
import matplotlib.pyplot as plt

from fastdtw import fastdtw

STEP = 0


class Population:
    def __init__(self, test_data):
        self.population_size = 10
        self.population = []
        self.matingpool = []
        self.test_data = test_data
        self.max_fitness = 1
        self.best = None

        for i in range(0, self.population_size, 1):
            self.population.append(Dron(test_data))

    def print_plot(self):
        #while True:
        generation = 100

        for i in range(0, generation):
            if all(item.get_dna().get_lifespan() == True for item in self.population):
                self.evaluate_fitness()
                self.natural_selection()
                print('max fitness', self.max_fitness)
                print('----------------------\n')
                if i % 10 == 0:
                    fig = plt.figure()
                    best_drone_idx = max(range(len(self.population)), key=lambda i: self.population[i].get_fitness())
                    for pop_idx, pop in enumerate(self.population):
                        if pop_idx == best_drone_idx:
                            plt.plot(*pop.get_dna().get_genes()[:, :-1].T, 'g-')
                        else:
                            plt.plot(*pop.get_dna().get_genes()[:, :-1].T, 'r-')
                    plt.plot(*self.test_data.T, 'b-')
                    plt.show()
                    plt.savefig(f'test_gen{i}.png')
                    plt.close(fig)
                
        
        #if (self.max_fitness > 20):

        for pop in self.population:
            plt.plot(*pop.get_dna().get_genes()[:, :-1].T)
        plt.plot(*self.test_data.T)
        plt.show()
        plt.savefig('test.png')

    def evaluate_fitness(self):
        maxfit = 0
        self.best = None
        # Iterate through all rockets and calcultes their fitness
        for i in range(0, len(self.population)-1):
            # calculate fitness
            self.population[i].calculate_fitness()

            # normalize fitness

            if (self.population[i].get_fitness() > maxfit):
                maxfit = self.population[i].get_fitness()
                self.best = i
                self.max_fitness = maxfit

        for i in range(0, len(self.population)-1):
            self.population[i].set_fitness(maxfit)

        # Take population fitness  and scale it from 1 to 100
        for i in range(0, len(self.population)-1):

            if i == self.best:
                # Give the best a bit bigger chance to mate
                n = int(self.population[i].get_fitness() * 125)

            else:
                # self.population[i]._fitness = self.population[i]._fitness * 0.5
                n = int(self.population[i].get_fitness() * 100)

            for j in range(0, n, 1):

                self.matingpool.append(self.population[i])

    def natural_selection(self):
        # Create a new population of the same size
        new_population = []
        for _ in range(self.population_size):
            # Pick random partners from the mating pool
            parentA = random.choice(self.matingpool).get_dna()
            parentB = random.choice(self.matingpool).get_dna()
            child = Drone(self.test_data)  # Create a new Drone for the child
            # Crossover and Mutation
            child_genes = parentA.crossover(parentB)
            child_genes = child._dna.mutation(child_genes)  # Move mutation here
            child.set_commands(child_genes)
            new_population.append(child)

        self.matingpool = []
        self.population = new_population  # Replace the old population


class Drone:
    def __init__(self, test_data, commands=None):
        self._dna = Dna(commands=commands)
        self._fitness = 0
        self.test_data = test_data

    def get_dna(self):
        return self._dna

    def get_commands(self):
        return self._dna.get_commands()

    def set_commands(self, commands):
        self._dna = Dna(commands=commands)

    def get_fitness(self):
        return self._fitness

    def set_fitness(self, maxfit):
        self._fitness /= maxfit

    def calculate_fitness(self):
        target_curve = self.test_data
        drone_path = numpy.array(self.get_dna().get_genes())[:, :-1]

        # 1. DTW Distance:
        distance, _ = fastdtw(target_curve, drone_path)

        # 2. Curve-Specific Point Distance Penalty:
        point_distances = numpy.zeros(len(drone_path))
        for i, drone_point in enumerate(drone_path):
            point_distances[i] = numpy.min(numpy.linalg.norm(target_curve - drone_point, axis=1))
        avg_point_distance = numpy.mean(point_distances)
        point_distance_penalty = avg_point_distance**2  # Square the penalty for stronger emphasis

        # 3. Combined Fitness:
        # Lower fitness value is better (minimization problem)
        fitness = distance + point_distance_penalty  
        self._fitness = 1 / (fitness + 1e-6)  # Convert to maximize fitness


class Dna:
    def __init__(self, commands=None):
        # x y current_rotation
        # Start at the beginning of the curve with correct rotation
        first_point = test_data[0]
        second_point = test_data[1]

        delta_x = second_point[0] - first_point[0]
        delta_y = second_point[1] - first_point[1]
        initial_angle = numpy.degrees(numpy.arctan2(delta_y, delta_x))  # Calculate angle in degrees

        self.genes = numpy.array([[first_point[0], first_point[1], initial_angle]])  # Include rotation
        # self.genes = numpy.array([test_data[0], [0]])  # Use the first point of test_data
        self.recall = numpy.array([])

        self.genes_length = 50
        self.lifespan = False
        self.count = 0

        # Commands issued to drone
        self.issued_commands = []
        # Commands called from the list!
        self.called_commands = []

        # Available commands in SDK
        self.my_list = [self.command_rotate, self.command_move]

        if (commands is not None):
            if len(commands):
                self.old_called_commands = commands
                for i in self.old_called_commands:
                    self.count += 0
                    if i[0] == 'move':
                        try:
                            self.command_move(i[1], i[2], None)
                        except numpy.core._exceptions.UFuncTypeError as e:
                            # TODO: Handle further
                            print(
                                'Exception - member of numpy array is not a number: ', e)
                    elif i[0] == 'rotate':
                        self.command_rotate(None, None, i[1])
                # Too old to live
                self.lifespan = True
        else:
            for i in range(1, self.genes_length, 1):
                self.count += 1
                random.choice(self.my_list)(
                    numpy.random.randint(1, 40), numpy.random.randint(1, 40), numpy.random.randint(-180, 180))
            # Too old to live
            self.lifespan = True

    def get_lifespan(self):
        return self.lifespan

    def get_genes(self):
        return self.genes

    def set_commands(self, genes):
        self.called_commands = genes

    def get_commands(self):
        return self.called_commands

    def get_count(self):
        return self.count

    def crossover(self, partner):
        new_genes = []
        # Select a random mid
        mid = numpy.floor(numpy.random.randint(len(self.called_commands)))

        for i in range(0, len(self.called_commands), 1):
            if (i > mid):
                new_genes.append(self.called_commands[i])
            else:
                new_genes.append(partner.called_commands[i])
        return new_genes

    def mutation(self, genes):
        for i in range(0, len(genes)-1):
            # 2% chance to evolve
            if (numpy.random.uniform() < 0.02):
                random.choice([self.command_rotate, self.command_move])(
                    numpy.random.randint(1, 40), numpy.random.randint(1, 40), numpy.random.randint(-180, 180), True)
                genes[i] = self.recall
        return genes

    def translate_to_dronish(self):
        return '\n'.join(self.issued_commands)

    # allowed commands

    def command_move(self, *args):
        x = int(args[0])
        y = int(args[1])
        # we dont use Z for simulation, for now 2D
        r_x, r_y = self._calculate_rotation_xy(x, y)

        self.genes = numpy.vstack(
            (self.genes, numpy.array([r_x, r_y, self.genes[-1][2]])))

        if len(args) == 4:
            # If mutation took a place (funct from random pool was called with fourth  )
            self.recall = numpy.array(['move', int(x), int(y)])
        else:
            self.called_commands.append(['move', int(x), int(y)])

        self.issued_commands.append(f'go {x} {y} 0 60')

    @staticmethod
    def _normalize_rotation(deg: int):
        angle = deg % 360

        angle = (angle + 360) % 360

        if (angle > 180):
            angle -= 360

        return angle

    def _calculate_rotation_xy(self, x, y):
        radians = numpy.radians(self.genes[-1][2])
        c, s = numpy.cos(radians), numpy.sin(radians)
        j = numpy.matrix([[c, s], [-s, c]])
        try:
            m = numpy.dot(j, [x, y])
            return float(m.T[0]), float(m.T[1])
        # TODO: This should not be generic, we should handle failures
        except Exception as e:
            print(j, [x, y])

    def command_rotate(self, *args):  # deg: int):
        deg = int(args[2])
        # We can rotatce clockwise (cw) with negative as counterclockwise (ccw), we are skipping ccw
        try:
            self.genes[-1,
                       2] = self._normalize_rotation(self.genes[-1][2] + deg)
        except numpy.core._exceptions.UFuncTypeError as e:
            print('Exception - Cannot normalize, numpy array member is not a number:', e)
        if len(args) == 4:
            # If mutation took a place (funct from random pool was called with fourth  )
            self.recall = numpy.array(['rotate', int(deg)])
        else:
            self.called_commands.append(['rotate', int(deg)])

        self.issued_commands.append(f'cw {deg}')

    def _select_random_function(self):
        random.choice(self.my_list)


class TestCurve:
    @staticmethod
    def create_circle(center_x=0, center_y=0, radius=1):
        # testing code for equality to calculate validity and weight of gen alg
        # NOTE: test confirmed that weight calculated by this procedure can be used to evaluate fitness of gen alg
        points = numpy.array([[radius, 0]])

        for degree in range(1, 360, 1):
            radians = numpy.radians(degree)
            x = center_x + radius * numpy.cos(radians)
            y = center_y + radius * numpy.sin(radians)
            points = numpy.vstack((points, [x, y]))

        return points
    
    @staticmethod
    def create_spiral(center_x=0, center_y=0, num_turns=2, radius_growth=1):
        theta = numpy.linspace(0, num_turns * 2 * numpy.pi, 360)
        radius = radius_growth * theta
        x = center_x + radius * numpy.cos(theta)
        y = center_y + radius * numpy.sin(theta)
        return numpy.column_stack((x, y))
    
    @staticmethod
    def create_polynomial(coeffs, x_shift=0, x_scale=1, rotation_angle=0):
        x_range = numpy.linspace(-1, 1, 100) * x_scale + x_shift   # Shift and scale x-values
        y_values = numpy.polyval(coeffs, x_range)

        # Rotate the points (same as before)
        theta = numpy.radians(rotation_angle)
        c, s = numpy.cos(theta), numpy.sin(theta)
        rotation_matrix = numpy.array(((c, -s), (s, c)))
        points = numpy.column_stack((x_range, y_values))
        rotated_points = numpy.dot(points, rotation_matrix)

        return rotated_points

if __name__ == '__main__':
    # test_data = TestCurve.create_circle(0, 0, 75)
    test_data = TestCurve.create_spiral(num_turns=0.9, radius_growth=3)  # Adjust parameters as needed
    # Example usage
    # coeffs = [0.01, -0.05, 0.1, -0.2, 0.3]  
    # test_data = TestCurve.create_polynomial(coeffs, x_shift=50, x_scale=25)
    population = Population(test_data)
    population.print_plot()
