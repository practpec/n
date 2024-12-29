# -------------------------------------------- LICENSE --------------------------------------------
#
# Copyright 2024 Ana Cequeira, Humberto Gomes, João Torres, José Lopes, José Matos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -------------------------------------------------------------------------------------------------

from dataclasses import dataclass
import itertools
import random

from IA.Distribution import Vehicle, Product

@dataclass
class BinPackingResult:
    results: list[tuple[Vehicle, list[Product]]]

GENERATIONS = 1000
POPULATION_SIZE = 100
BREEDERS = 10

def genetic_bin_pack(vehicle_costs: dict[Vehicle, float],
                     vehicles: dict[Vehicle, int],
                     products: dict[Product, int]) -> BinPackingResult:

    products_list = list(itertools.chain(*[[p] * i for p, i in products.items()]))
    vehicles_list = list(itertools.chain(*[[v] * i for v, i in vehicles.items()]))
    gene_length = sum(products.values())

    def fitness(gene: list[int]) -> float:
        total_weight = 0.0
        vehicle_weights: dict[int, float] = {}
        gene_vehicle_costs = 0.0
        for i, product_vehicle in enumerate(gene):
            if product_vehicle != -1:
                total_weight += products_list[i].weight

                if product_vehicle not in vehicle_weights:
                    vehicle_weights[product_vehicle] = 0.0
                    gene_vehicle_costs += vehicle_costs[vehicles_list[product_vehicle]]
                vehicle_weights[product_vehicle] += products_list[i].weight

        if any(vehicles_list[v].max_weight < w for v, w in vehicle_weights.items()):
            return 0

        # Positive reinforcement for more weight, negative reinforcement for more vehicles
        return total_weight * 1e6 - gene_vehicle_costs - len(vehicle_weights)

    def crossover(parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(individual: list[int]) -> list[int]:
        if random.random() < 0.5:
            individual[random.randint(0, gene_length - 1)] = \
                random.randint(-1, len(vehicles_list) - 1)
        else:
            from_vehicle = individual[random.randint(0, gene_length - 1)]
            to_vehicle = random.randint(-1, len(vehicles_list) - 1)
            individual = [ (to_vehicle if v == from_vehicle else v) for v in individual ]

        return individual

    population = [
        random.choices(range(-1, len(vehicles_list)), k=gene_length) for _ in range(POPULATION_SIZE)
    ]

    for _ in range(GENERATIONS):
        population.sort(key=fitness)
        new_population = population[-BREEDERS:]

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.choices(population[-BREEDERS:], k=2)
            child1, child2 = crossover(parent1, parent2)

            if random.random() < 0.1:
                child1 = mutate(child1)
            if random.random() < 0.1:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    best_gene = max(population, key=fitness)

    ret: dict[int, list[Product]] = {}
    for product_index, vehicle_index in enumerate(best_gene):
        if vehicle_index == -1:
            continue

        if vehicle_index not in ret:
            ret[vehicle_index] = []
        ret[vehicle_index].append(products_list[product_index])

    return BinPackingResult([(vehicles_list[v], lp) for v, lp in ret.items()])
