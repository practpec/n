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
import heapq
import itertools
import json
from typing import Optional

from IA.binpack import bin_pack
from IA.structs import Vehicle, Product, DistributionCenter, DeliveryTarget
from IA.map import Map, SearchHeuristic
from IA.binpack import BinPackingResult
from IA.graph import SearchAlgorithm, SearchResults

@dataclass
class Death:
    node: int

ProblemSolution = list[tuple[str, None | SearchResults | BinPackingResult | Death]]

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

        products = {
            name: Product(product['name'],
                          product['weight'],
                          product['color']) for name, product in data['products'].items()
        }

        distribution_center = DistributionCenter(
            data['distribution-center']['name'],
            data['distribution-center']['node'],
            {vehicles[v]: n for v, n in data['distribution-center']['vehicles'].items()},
            {products[p]: n for p, n in data['distribution-center']['products'].items()}
        )

        self.map = Map(data['map']['location'],
                       data['map']['directed'],
                       data['map']['enable-weather'],
                       distribution_center)

        for target in data['delivery-targets'].values():
            self.map.delivery_targets.append(DeliveryTarget(
                target['name'],
                target['node'],
                target['time-limit'],
                {products[p]: n for p, n in target['products'].items()})
            )

    def test(self) -> ProblemSolution:
        best_vehicle = Vehicle('best', float('inf'), float('inf'), 1.0, 1.0, '')
        algorithms = [
            (SearchAlgorithm.DFS_FAST, SearchHeuristic.CARTESIAN),
            # Too slow to test
            # (SearchAlgorithm.DFS_CORRECT, SearchHeuristic.CARTESIAN),
            # (SearchAlgorithm.ITERATIVE, SearchHeuristic.CARTESIAN),
            (SearchAlgorithm.BFS, SearchHeuristic.CARTESIAN),
            (SearchAlgorithm.DIJKSTRA, SearchHeuristic.CARTESIAN),
            (SearchAlgorithm.GREEDY, SearchHeuristic.CARTESIAN),
            (SearchAlgorithm.GREEDY, SearchHeuristic.MANHATTAN),
            (SearchAlgorithm.ASTAR, SearchHeuristic.CARTESIAN),
            (SearchAlgorithm.ASTAR, SearchHeuristic.MANHATTAN),
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

                title = self.format_search_title(algorithm, heuristic, target, results)
                ret.append((title, results))

        return ret

    def solve(self) -> ProblemSolution:
        ret: ProblemSolution = [('Press RETURN to continue', None)]
        current_time = 0.0
        waiting: dict[float, list[Vehicle]] = {}

        targets = [(t.time_limit, t) for t in self.map.delivery_targets]
        heapq.heapify(targets)

        while targets:
            time_limit, target = heapq.heappop(targets)

            if not target.enough_resources_to_supply(self.map.distribution_center):
                ret.append((f'{target.name} não consegue ser suprido (sem recursos)',
                            Death(target.node)))
                continue

            must_wait = False
            while target.products:
                if must_wait or not self.map.distribution_center.vehicles:
                    min_time = min(time for time in waiting)
                    new_vehicles = waiting.pop(min_time)

                    current_time = min_time
                    must_wait = False

                    for vehicle in new_vehicles:
                        if vehicle not in self.map.distribution_center.vehicles:
                            self.map.distribution_center.vehicles[vehicle] = 0

                        self.map.distribution_center.vehicles[vehicle] += 1

                if current_time > time_limit:
                    ret.append((f'{target.name} não consegue ser suprido (sem tempo)',
                                Death(target.node)))
                    break

                costs, vehicles = self.calculate_vehicles(target,
                                                          ret,
                                                          current_time,
                                                          time_limit - current_time)
                if not vehicles:
                    if self.has_new_vehicles_waiting(waiting):
                        must_wait = True
                        continue

                    ret.append((f'{target.name} não consegue ser suprido (sem tempo)',
                                Death(target.node)))
                    break

                binpack = bin_pack(costs, vehicles, target.products)
                ret.append((self.format_binpack_title(target, current_time), binpack))

                self.map.distribution_center.discount_bin_pack(binpack)
                target.discount_bin_pack(binpack)
                self.enqueue_vehicles(binpack, costs, current_time, waiting)

        return ret

    def enqueue_vehicles(self,
                         binpack: BinPackingResult,
                         costs: dict[Vehicle, float],
                         current_time: float,
                         waiting: dict[float, list[Vehicle]]) -> None:

        for vehicle, _ in binpack.results:
            time_done = current_time + costs[vehicle] * 2 # * 2 for round trip

            to_add = waiting.get(time_done)
            if to_add is None:
                to_add = []
                waiting[time_done] = to_add

            to_add.append(vehicle)

    def has_new_vehicles_waiting(self, waiting: dict[float, list[Vehicle]]) -> bool:
        waiting_vehicles = set(itertools.chain(*waiting.values()))
        stationed_vehicles = set(self.map.distribution_center.vehicles)
        new_vehicles = waiting_vehicles - stationed_vehicles

        return len(new_vehicles) > 0

    def calculate_vehicles(self,
                           target: DeliveryTarget,
                           ret: ProblemSolution,
                           current_time: float,
                           max_time: float) -> tuple[dict[Vehicle, float], dict[Vehicle, int]]:

        def output_search(search: Optional[SearchResults], usages: list[Vehicle]) -> None:
            if search is None:
                return

            title = self.format_search_title(SearchAlgorithm.ASTAR,
                                             SearchHeuristic.CARTESIAN,
                                             target,
                                             search,
                                             usages,
                                             current_time)
            ret.append((title, search))

        costs: dict[Vehicle, float] = {}
        vehicles: dict[Vehicle, int] = {}

        previous: Optional[SearchResults] = None
        previous_usages: list[Vehicle] = []
        for vehicle, number in self.map.distribution_center.vehicles.items():
            previous_valid = False
            if previous is not None:
                new_search = self.map.results_valid_for_vehicle(previous, vehicle)

                if new_search is not None:
                    previous_valid = True
                    previous = new_search

            if not previous_valid:
                output_search(previous, previous_usages)

                previous_usages = []
                previous = self.map.search(self.map.distribution_center.node,
                                         target.node,
                                         SearchAlgorithm.ASTAR,
                                         vehicle,
                                         SearchHeuristic.CARTESIAN)

            previous_usages.append(vehicle)
            if isinstance(previous, SearchResults) and \
                previous.cost is not None and \
                previous.cost <= max_time:

                costs[vehicle] = previous.cost
                vehicles[vehicle] = number

        output_search(previous, previous_usages)
        return costs, vehicles

    def format_search_title(self,
                            algorithm: SearchAlgorithm,
                            heuristic: SearchHeuristic,
                            target: DeliveryTarget,
                            results: SearchResults,
                            vehicles: Optional[list[Vehicle]] = None,
                            time: Optional[float] = None) -> str:

        title = ''
        if time is not None:
            title = self.format_time(time) + ': '

        if vehicles is None:
            title += algorithm.name
            if algorithm.value >= SearchAlgorithm.GREEDY.value:
                title += f' ({heuristic.name}): '
        else:
            title += ' = '.join(v.name for v in vehicles) + ': '

        title += f'{self.map.distribution_center.name} -> {target.name} ('

        format_time = str(round(results.time * 1e3, 2)) + ' ms'
        if results.path is not None and results.cost is not None and results.distance is not None:
            rounded_cost = round(results.distance, 2)
            title += f'found in {format_time}, dist={rounded_cost}, Nnodes={len(results.path)}, '

        else:
            title += f'not found after {format_time}, '

        title += f'Nvisited={len(results.visited)})'
        return title

    def format_binpack_title(self, target: DeliveryTarget, time: float) -> str:
        return f'{self.format_time(time)}: Bin packing to {target.name}'

    def format_time(self, time: float) -> str:
        minutes, seconds = divmod(round(time), 60)
        if minutes < 60:
            return f'{minutes}m{seconds}s'
        else:
            hour, minutes = divmod(minutes, 60)
            return f'{hour}h{minutes}m{seconds}s'
