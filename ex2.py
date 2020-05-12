import csv
import numpy as np
import random
from copy import deepcopy as dc
import cv2

adjMatrix = np.loadtxt(open("adjMatrix.csv"), delimiter=",")
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
        self.image()
        population = self._initialize_population()
        best = self.pick_best(population)


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
    def pick_best(self,population):
        population.sort(key=lambda x: x[1])
        return population[0]

    def calc_fitness(self, coloring):
        fitness = 0
        for v,color in coloring.items():
            fitness += v.validate_coloring(coloring, color)
        return fitness
    def image(self):
        img = cv2.imread('map.PNG')
        part8
        part9 = (200,200)
        part10 = (200,280)
        part11 = (240,20)
        part12 = (240, 160)
        img[part9[0]:210, part9[1]:260] = [0, 0, 255]
        img[part10[0]:210, part10[1]:350] = [0, 255, 255]
        img[part11[0]:250, part11[1]:155] = [0, 255, 0]
        img[part12[0]:250, part12[1]:420] = [0, 255, 255]

        cv2.imshow("img", img)

        cv2.waitKey()
        cv2.destroyAllWindows()

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
