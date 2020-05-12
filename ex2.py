import numpy as np
import random
from copy import deepcopy as dc
from Bio_ex2 import show_image

ADJ_PATH = "Bio_ex2/adjMatrix.csv"
adjMatrix = np.loadtxt(ADJ_PATH, delimiter=",")
COLORS = ['G', 'R', 'B', 'L']
START_FIT = 0
NUM_OF_NODES = 12


class Engine:
    def __init__(self, graph, parameters):
        self.graph = graph
        self.population_size = parameters['population_size']
        self.mutation_probability = parameters['mutation_probability']
        self.crossover_probability = parameters['crossover_probability']
        self.num_of_vertex = parameters['num vertex']

    def run(self):
        population = self._initialize_population()
        avg_fitness, best_fitness = self.pick_best(population)
        print(avg_fitness,best_fitness[1])
        colors_map = show_image.set_coloring_for_image(best_fitness)
        show_image.image(colors_map)

    def _initialize_population(self):
        random.seed()
        population = []
        while len(population) < self.population_size:
            coloring = {}
            for v in graph.nodes.values():
                chosen_color = random.choice(COLORS)
                coloring[v] = chosen_color
            fitness = self.calc_fitness(coloring)
            population.append((coloring, fitness))
        return population

    def pick_best(self, population):

        population.sort(key=lambda x: x[1])
        fitnesses = [f[1] for f in population]
        avg = sum(fitnesses) / len(fitnesses)
        return avg, population[0]

    def calc_fitness(self, coloring):
        fitness = 0
        for v, color in coloring.items():
            fitness += v.validate_coloring(coloring, color)
        return fitness


class Graph:
    def __init__(self, adjMatrix):
        self.adjMatrix = adjMatrix
        self.nodes = {}

    def build_nodes(self):
        for i in range(NUM_OF_NODES):
            result = np.where(self.adjMatrix[i, :] == 1)
            n = np.add(result, 1)
            node_i = Node(ID=i + 1, neighbours=n)
            self.nodes[i + 1] = dc(node_i)
        for node in self.nodes.values():
            n_array = np.array(node.neighbours_index)
            for n in n_array:
                node.set_neighbours(self.nodes.get(n))


class Node:

    def __init__(self, ID, neighbours):
        self.ID = ID
        self.neighbours_index = neighbours[0]
        self.neighbours_obj = {}

    def set_neighbours(self, node_n):
        self.neighbours_obj[node_n.ID] = node_n

    def validate_coloring(self, coloring, this_color):
        count = 0
        for v, color in coloring.items():
            if v.ID in self.neighbours_index:
                if color == this_color:
                    count += 1
        return count

    def __hash(self):
        return self.ID


params = {
    'population_size': 100,
    'mutation_probability': 0.05,
    'crossover_probability': 0.8,
    'num vertex': 12
}

graph = Graph(adjMatrix)
graph.build_nodes()
engine = Engine(graph, params)
engine.run()
