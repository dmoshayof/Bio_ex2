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
        print(best[1])
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
                colors_num[v.ID] = [152,251,152]
            if color == 'R':
                colors_num[v.ID] = [204, 255, 255]
            if color == 'B':
                colors_num[v.ID] = [245,147,101]
            if color == 'L':
                colors_num[v.ID] = [147, 112, 219]
        return colors_num



    def image(self,colors_i):
        img = cv2.imread('Bio_ex2/map.PNG')
        part1 = (41,58)
        part2 = (90,56,175)
        part3 = (118,57,108)
        part4 = (200,80)
        part5 = (195,130)
        part6 = (42,212,290)
        part7 = (89,212)
        part8 = (130,300)
        part9 = (156,174)
        part10 = (201,273,322,110)
        part11 = (218,15,10)
        part12 = (218, 160,10,373,265)

        img[part1[0]:85, part1[1]:208] = colors_i[1] #all

        img[part2[0]:113, part2[1]:208] = colors_i[2] #up
        img[part2[0]:153, part2[2]:208] = colors_i[2]  #down-side

        img[part3[0]:163, part3[1]:170] = colors_i[3] #main
        img[163:174, part3[2]:170] = colors_i[3]
        img[174:192, 156:170] = colors_i[3] #small squre

        img[167:213, 58:103] = colors_i[4] #main
        img[178:213, 103:127] = colors_i[4] #down
        img[178:192, 103:154] = colors_i[4] #side

        img[part5[0]:215, part5[1]:181] = colors_i[5]

        img[part6[0]:85, part6[1]:368] = colors_i[6] #main
        img[part6[0]:105, part6[2]:368] = colors_i[6] #down-side

        img[part7[0]:125, part7[1]:285] = colors_i[7] #all

        img[109:197, 288:318] = colors_i[8] #left
        img[129:197, 210:318] = colors_i[8]

        img[part9[0]:191, part9[1]:247] = colors_i[9] #up
        img[185:214, 185:270] = colors_i[9] #down

        img[part10[3]:212, part10[2]:368] = colors_i[10] #middel
        img[part10[0]:213, part10[1]:368] = colors_i[10] #down

        img[part11[0]:250, part11[1]:155] = colors_i[11] #down
        img[part11[1]:240, part11[1]:54] = colors_i[11] #side
        img[part11[2]:37, part11[1]:260] = colors_i[11] #up

        img[part12[2]:37, part12[4]:420] = colors_i[12] #up
        img[part12[2]:250, part12[3]:420] = colors_i[12] #side
        img[part12[0]:250, part12[1]:420] = colors_i[12] #down

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
    'population_size': 10000,
    'mutation_probability': 0.05,
    'crossover_probability': 0.8,
    'num vertex': 12
}

graph = Graph(adjMatrix)
graph.build_nodes()
engine = Engine(graph, params)
engine.run()
