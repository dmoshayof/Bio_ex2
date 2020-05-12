import csv
import numpy as np
import random
from copy import deepcopy as dc
import cv2

adjMatrix = np.loadtxt(open("Bio_ex2/adjMatrix.csv"), delimiter=",")
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
        best = self.pick_best(population)
        colors_map = self.set_coloring_for_image(best)
        self.image(colors_map)

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
    def set_coloring_for_image(self,coloring):
        colors_num = {}
        for v, color in coloring[0].items():
            if color == 'G':
                colors_num[v.ID] = [0,255,0]
            if color == 'R':
                colors_num[v.ID] = [0, 0, 255]
            if color == 'B':
                colors_num[v.ID] = [255,0,0]
            if color == 'L':
                colors_num[v.ID] = [0, 0, 0]
        return colors_num



    def image(self,colors_i):
        img = cv2.imread('Bio_ex2/map.PNG')
        part1 = (60,60)
        part2 = (100,80)
        part3 = (130,80)
        part4 = (200,80)
        part5 = (200,160)
        part6 = (60,290)
        part7 = (100,250)
        part8 = (130,300)
        part9 = (200,200)
        part10 = (200,280)
        part11 = (240,20)
        part12 = (240, 160)
        img[part1[0]:80, part1[1]:100] = colors_i[1]
        img[part2[0]:110, part2[1]:100] = colors_i[2]
        img[part3[0]:150, part3[1]:110] = colors_i[3]
        img[part4[0]:210, part4[1]:120] = colors_i[4]
        img[part5[0]:210, part5[1]:170] = colors_i[5]
        img[part6[0]:90, part6[1]:350] = colors_i[6]
        img[part7[0]:120, part7[1]:260] = colors_i[7]
        img[part8[0]:170, part8[1]:310] = colors_i[8]
        img[part9[0]:210, part9[1]:260] = colors_i[9]
        img[part10[0]:210, part10[1]:350] = colors_i[10]
        img[part11[0]:250, part11[1]:155] = colors_i[11]
        img[part12[0]:250, part12[1]:420] = colors_i[12]

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
