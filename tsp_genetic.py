import pandas as pd
import numpy as np
import random
from haversine import haversine
import matplotlib.pyplot as plt

class Gene:
    def __init__(self, name, lat, lng):
        self.name = name
        self.lat = lat
        self.lng = lng

    def get_distance_to(self, dest):
        origin = (self.lat, self.lng)
        dest = (dest.lat, dest.lng)
        return int(haversine(origin, dest))

class Individual:
    def __init__(self, genes):
        assert(len(genes) > 3)
        self.genes = genes
        self.__reset_params()

    def swap(self, gene_1, gene_2):
        a, b = self.genes.index(gene_1), self.genes.index(gene_2)
        self.genes[b], self.genes[a] = self.genes[a], self.genes[b]
        self.__reset_params()

    def add(self, gene):
        self.genes.append(gene)
        self.__reset_params()

    @property
    def fitness(self):
        if self.__fitness == 0:
            self.__fitness = 1 / self.travel_cost
        return self.__fitness

    @property
    def travel_cost(self):  
        if self.__travel_cost == 0:
            for i in range(len(self.genes)):
                origin = self.genes[i]
                dest = self.genes[(i + 1) % len(self.genes)]
                self.__travel_cost += origin.get_distance_to(dest)
        return self.__travel_cost

    def __reset_params(self):
        self.__travel_cost = 0
        self.__fitness = 0

class GeneticAlgorithm:
    def __init__(self, csv_file, population_size=100, elite_size=20, mutation_rate=0.01, generations=500):
        self.csv_file = csv_file
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.cities = self.read_cities_from_csv()

    def read_cities_from_csv(self):
        cities_df = pd.read_csv(self.csv_file)
        return [Gene(row['city'], row['lat'], row['lng']) for _, row in cities_df.iterrows()]

    def create_initial_population(self):
        return [Individual(random.sample(self.cities, len(self.cities))) for _ in range(self.population_size)]

    def evolve(self, population):
        new_generation = []
        elitism_num = len(population) // 2

        elites = sorted(population, key=lambda x: x.fitness, reverse=True)[:elitism_num]
        new_generation.extend(elites)

        for _ in range(self.population_size - elitism_num):
            parent1, parent2 = random.sample(elites, 2)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_generation.append(child)

        return new_generation

    def crossover(self, parent_1, parent_2):
        genes_n = len(parent_1.genes)
        child_genes = [None for _ in range(genes_n)]
        start_at = random.randint(0, genes_n - 1)
        finish_at = (start_at + genes_n // 2) % genes_n
        for i in range(start_at, finish_at):
            child_genes[i] = parent_1.genes[i]
        j = 0
        for i in range(genes_n):
            if child_genes[i] is None:
                while parent_2.genes[j] in child_genes:
                    j += 1
                child_genes[i] = parent_2.genes[j]
                j += 1
        return Individual(child_genes)

    def mutate(self, individual):
        for _ in range(len(individual.genes)):
            if random.random() < self.mutation_rate:
                sel_genes = random.sample(individual.genes, 2)
                individual.swap(sel_genes[0], sel_genes[1])

    def run_genetic_algorithm(self):
        population = self.create_initial_population()
        best_distance = float('inf')
        best_route = None

        for _ in range(self.generations):
            population = self.evolve(population)
            current_best = min(population, key=lambda x: x.travel_cost)
            if current_best.travel_cost < best_distance:
                best_distance = current_best.travel_cost
                best_route = current_best

        return best_route.genes, best_distance

    def plot_route(self, best_route, save_path=None):
        lats = [gene.lat for gene in best_route]
        lngs = [gene.lng for gene in best_route]
        names = [gene.name for gene in best_route]

        plt.figure(figsize=(10, 6))
        plt.scatter(lngs, lats, color='blue')
        for i in range(len(best_route)):
            plt.annotate(names[i], (lngs[i], lats[i]), fontsize=9)
            plt.plot([lngs[i], lngs[(i + 1) % len(best_route)]],
                     [lats[i], lats[(i + 1) % len(best_route)]], color='red', linestyle='-', linewidth=1)
        plt.title('Best Route')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.show()
        


if __name__ == "__main__":
    csv_file = "cities.csv"  
    ga = GeneticAlgorithm(csv_file)
    best_route, best_distance = ga.run_genetic_algorithm()
    print("Best Route:", [gene.name for gene in best_route])
    print("Best Distance:", best_distance)
    ga.plot_route(best_route)
