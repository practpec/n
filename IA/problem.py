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

import json

from IA.structs import Vehicle, Product, DistributionCenter, DeliveryTarget
from IA.map import Map, SearchHeuristic
from IA.binpack import BinPackingResult
from IA.graph import SearchAlgorithm, SearchResults

ProblemSolution = list[tuple[str, None | SearchResults | BinPackingResult]]

class Problem:
    def __init__(self, filepath: str) -> None:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vehicles = {
            name: Vehicle(vehicle['name'],
                          vehicle['max_fuel'],
                          vehicle['max_weight'],
                          vehicle['worst_weather'],
                          vehicle['speed'],
                          vehicle['image']) for name, vehicle in data['vehicles'].items()}

        distribution_center = DistributionCenter(
            data['distribution-center']['name'],
            data['distribution-center']['node'],
            {vehicles[v]: n for v, n in data['distribution-center']['vehicles'].items()}
        )

        self.map = Map(data['map']['location'],
                       data['map']['directed'],
                       data['map']['enable-weather'],
                       distribution_center)

        products = {
            name: Product(product['name'],
                          product['weight'],
                          product['color']) for name, product in data['products'].items()
        }

        for target in data['delivery-targets'].values():
            self.map.delivery_targets.append(DeliveryTarget(
                target['name'],
                target['node'],
                {products[p]: n for p, n in target['products'].items()})
            )

    def test(self) -> ProblemSolution:
        best_vehicle = Vehicle('best', float('inf'), float('inf'), 1.0, 1.0, '')
        algorithms = [
            # (SearchAlgorithm.DFS_FAST, SearchHeuristic.CARTESIAN),
            # (SearchAlgorithm.DFS_CORRECT, SearchHeuristic.CARTESIAN),
            # (SearchAlgorithm.ITERATIVE, SearchHeuristic.CARTESIAN),
            (SearchAlgorithm.BFS, SearchHeuristic.CARTESIAN),
            (SearchAlgorithm.DIJKSTRA, SearchHeuristic.CARTESIAN),
            # (SearchAlgorithm.GREEDY, SearchHeuristic.CARTESIAN),
            # (SearchAlgorithm.GREEDY, SearchHeuristic.MANHATTAN),
            (SearchAlgorithm.ASTAR, SearchHeuristic.CARTESIAN),
            # (SearchAlgorithm.ASTAR, SearchHeuristic.MANHATTAN),
        ]

        ret: ProblemSolution = [('Press RETURN to continue', None)]

        for target in self.map.delivery_targets:
            for algorithm_heuristic in algorithms:
                algorithm, heuristic = algorithm_heuristic

                results = self.map.search(self.map.distribution_center.node,
                                          target.node,
                                          algorithm,
                                          best_vehicle,
                                          heuristic)

                title = self.format_results_title(algorithm, heuristic, target, results)
                ret.append((title, results))

        return ret

    def format_results_title(self,
                             algorithm: SearchAlgorithm,
                             heuristic: SearchHeuristic,
                             target: DeliveryTarget,
                             results: SearchResults) -> str:

        title = algorithm.name
        if algorithm.value >= SearchAlgorithm.GREEDY.value:
            title += f' ({heuristic.name})'
        title += f': {self.map.distribution_center.name} -> {target.name} ('

        format_time = str(round(results.time * 1e3, 2)) + ' ms'
        if results.path is not None and results.cost is not None and results.distance is not None:
            rounded_cost = round(results.cost, 2)
            title += f'found in {format_time}, cost={rounded_cost}, Nnodes={len(results.path)}, '

        else:
            title += f'not found after {format_time}, '

        title += f'Nvisited={len(results.visited)})'
        return title
