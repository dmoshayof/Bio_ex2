import random
from copy import deepcopy as dc

import numpy as np

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
        self.top_solutions_probability = int(parameters['top_solutions_probability'] * self.population_size)
        self.mutation_probability = int(parameters['mutation_probability'] * self.population_size)
        self.crossover_probability = int(parameters['crossover_probability'] * self.population_size)
        self.num_of_vertex = parameters['num vertex']
        self.num_iter = parameters['number_of_iterations']

    def run(self):
        random.seed()
        population = self._initialize_population()
        for i in range(self.num_iter):
            avg_fitness, best_fitness, top_solutions, rest_population = self._pick_best(population)
            new_crossovers = self.crossover(rest_population)
            population = top_solutions + new_crossovers
            self.mutate(population)
            print(avg_fitness, best_fitness[1],len(population))
            colors_map = show_image.set_coloring_for_image(best_fitness)
            show_image.image(colors_map)
            if best_fitness[1] == 0:
                break

    def mutate(self,population):
        for i in range(self.mutation_probability):
            gene = random.choice(population)
            rand_vertex = random.choice(list(gene[0].keys()))
            gene[0][rand_vertex] = random.choice(COLORS)
            fitness = self._calc_fitness(gene[0])
            new_gene = (gene[0],fitness)
            population[population.index(gene)] = new_gene



    def crossover(self,population):
        new_population = []
        for gene1 in population:
            gene2 = random.choice(population)
            cutoff = random.randint(0,self.num_of_vertex)
            coloring = {}
            count = 0
            for v,color in gene1[0].items():
                if count == cutoff:
                    break
                coloring[v] = color
                count +=1
            count = 0
            for v, color in gene2[0].items():
                count +=1
                if count < cutoff:
                    continue
                coloring[v] = color
            fitness = self._calc_fitness(coloring)
            new_population.append((coloring, fitness))
        return new_population


    def _initialize_population(self):
        random.seed()
        population = []
        while len(population) < self.population_size:
            coloring = {}
            for v in graph.nodes.values():
                chosen_color = random.choice(COLORS)
                coloring[v] = chosen_color
            fitness = self._calc_fitness(coloring)
            population.append((coloring, fitness))
        return population

    def _pick_best(self, population):
        population.sort(key=lambda x: x[1])
        fitnesses = [f[1] for f in population]
        avg = sum(fitnesses) / len(fitnesses)
        top_solutions_rate = self.top_solutions_probability
        top_solutions = population[0:top_solutions_rate]
        rest = population[top_solutions_rate:]
        return avg, population[0], top_solutions, rest

    def _calc_fitness(self, coloring):
        fitness = 0
        for v, color in coloring.items():
            fitness += v.self_fitness(coloring, color)
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

    def self_fitness(self, coloring, this_color):
        count = 0
        for v, color in coloring.items():
            if v.ID in self.neighbours_index:
                if color == this_color:
                    count += 1
        return count

    def __hash(self):
        return self.ID


params = {
    'population_size': 1000,
    'mutation_probability': 0.2,
    'crossover_probability': 0.95,
    'num vertex': 12,
    'number_of_iterations': 20,
    'top_solutions_probability': 0.05
}

graph = Graph(adjMatrix)
graph.build_nodes()
engine = Engine(graph, params)
engine.run()
