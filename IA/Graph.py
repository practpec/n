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

from collections import deque
from enum import Enum
import heapq
import time
from typing import Callable, NamedTuple, Optional, cast

class SearchResults(NamedTuple):
    path: Optional[list[int]]
    cost: Optional[float]     # cost after real_cost function
    distance: Optional[float] # cost according to the graph edges
    visited: list[int]
    time: float

class SearchAlgorithm(Enum):
    DFS = 1
    ITERATIVE = 2
    BFS = 3
    DIJKSTRA = 4
    GREEDY = 5
    ASTAR = 6

class Graph:
    def __init__(self) -> None:
        self.edges: dict[int, dict[int, float]] = {}

    def raw_dfs(self,
                source: int,
                target: int,
                cost_limit: float,
                real_cost: Callable[[int, int], float],
                can_pass: Callable[[int, int], bool],
                limit_cost: Callable[[int, int], float]) -> SearchResults:

        start = time.process_time()

        edge = [(source, 0)]          # Algorithm's stack (because Python's stack is very small)
        cost: float = 0.0             # Limit (fuel) cost
        visited: dict[int, None] = {} # Ordered set

        while edge:
            expanded, edge_visit_count = edge[-1]
            visited[expanded] = None

            if expanded == target:
                path = [v for v, _ in edge]
                cost = self.cost_from_path(path, real_cost)
                distance = self.cost_from_path(path, lambda o, d: self.edges[o][d])
                visited_ret = list(visited)

                end = time.process_time()
                return SearchResults(path, cost, distance, visited_ret, end - start)

            adjacents = list(self.edges[expanded])
            if edge_visit_count == len(adjacents):
                if len(edge) >= 2:
                    edge_limit_cost = limit_cost(edge[-2][0], edge[-1][0])
                    cost -= edge_limit_cost

                edge.pop()
            else:
                edge[-1] = (expanded, edge_visit_count + 1)

                to_visit = adjacents[edge_visit_count]
                if to_visit not in visited and can_pass(expanded, to_visit):
                    total_limit_cost = cost + limit_cost(expanded, to_visit)
                    if total_limit_cost <= cost_limit:
                        cost = total_limit_cost
                        edge.append((adjacents[edge_visit_count], 0))

        end = time.process_time()
        return SearchResults(None, None, None, list(visited), end - start)

    def raw_iterative(self,
                      source: int,
                      target: int,
                      cost_limit: float,
                      real_cost: Callable[[int, int], float],
                      can_pass: Callable[[int, int], bool],
                      limit_cost: Callable[[int, int], float]) -> SearchResults:

        start = time.process_time()

        max_depth = 0
        last_visited = -1
        while True:
            results = self.raw_dfs(source, target, max_depth, real_cost, can_pass, lambda s, t: 1)
            if results.path:
                end = time.process_time()

                path_limit_cost = self.cost_from_path(results.path, limit_cost)
                if path_limit_cost > cost_limit:
                    return SearchResults(None, None, None, list(results.visited), end - start)
                else:
                    return SearchResults(results.path,
                                         results.cost,
                                         results.distance,
                                         results.visited,
                                         end - start)


            if len(results.visited) == last_visited:
                # Whole graph explored. Can't go any further
                end = time.process_time()
                return SearchResults(None, None, None, list(results.visited), end - start)

            last_visited = len(results.visited)
            max_depth += 1

    def raw_bfs(self,
                source: int,
                target: int,
                cost_limit: float,
                real_cost: Callable[[int, int], float],
                can_pass: Callable[[int, int], bool],
                limit_cost: Callable[[int, int], float]) -> SearchResults:

        start = time.process_time()

        edge = deque([source])                               # Nodes to be expanded
        parents: dict[int, Optional[int]] = { source: None } # To back trace the path
        limit_costs: dict[int, float] = { source: 0.0 }      # Fuel costs to each node

        while edge:
            expanded = edge[0]
            if expanded == target:
                path = self.path_from_parents(target, parents)
                cost = self.cost_from_path(path, real_cost)
                distance = self.cost_from_path(path, lambda o, d: self.edges[o][d])
                visited = list(parents)

                end = time.process_time()
                return SearchResults(path, cost, distance, visited, end - start)

            edge.popleft()
            for to_visit in self.edges[expanded]:
                if to_visit not in parents and can_pass(expanded, to_visit):
                    total_cost = limit_costs[expanded] + limit_cost(expanded, to_visit)
                    if total_cost < cost_limit:
                        edge.append(to_visit)
                        parents[to_visit] = expanded
                        limit_costs[to_visit] = total_cost


        end = time.process_time()
        return SearchResults(None, None, None, list(parents), end - start)

    def raw_dijkstra(self,
                     source: int,
                     target: int,
                     cost_limit: float,
                     real_cost: Callable[[int, int], float],
                     can_pass: Callable[[int, int], bool],
                     limit_cost: Callable[[int, int], float]) -> SearchResults:

        start = time.process_time()

        edge = [(0.0, source)]                               # Nodes to be expanded
        parents: dict[int, Optional[int]] = { source: None } # To back trace the path
        limit_costs: dict[int, float] = { source: 0.0 }      # Fuel costs to each node

        while edge:
            cost_to_expanded, expanded = heapq.heappop(edge)
            if expanded == target:
                path = self.path_from_parents(target, parents)
                cost = self.cost_from_path(path, real_cost)
                distance = self.cost_from_path(path, lambda o, d: self.edges[o][d])
                visited = list(parents)

                end = time.process_time()
                return SearchResults(path, cost, distance, visited, end - start)

            for to_visit in self.edges[expanded]:
                if to_visit not in parents and can_pass(expanded, to_visit):
                    total_cost = limit_costs[expanded] + limit_cost(expanded, to_visit)
                    if total_cost < cost_limit:
                        cost_to_visit = cost_to_expanded + self.edges[expanded][to_visit]
                        heapq.heappush(edge, (cost_to_visit, to_visit))
                        parents[to_visit] = expanded
                        limit_costs[to_visit] = total_cost

        end = time.process_time()
        return SearchResults(None, None, None, list(parents), end - start)

    def raw_greedy(self,
                   source: int,
                   target: int,
                   cost_limit: float,
                   real_cost: Callable[[int, int], float],
                   can_pass: Callable[[int, int], bool],
                   limit_cost: Callable[[int, int], float],
                   heuristic: Callable[[int], float]) -> SearchResults:

        start = time.process_time()

        edge = [(source, 0)]          # Algorithm's stack (because Python's stack is very small)
        cost: float = 0.0             # Limit (fuel) cost
        visited: dict[int, None] = {} # Ordered set
        while edge:
            expanded, edge_visit_count = edge[-1]
            visited[expanded] = None

            if expanded == target:
                path = [v for v, _ in edge]
                cost = self.cost_from_path(path, real_cost)
                distance = self.cost_from_path(path, lambda o, d: self.edges[o][d])
                visited_ret = list(visited)

                end = time.process_time()
                return SearchResults(path, cost, distance, visited_ret, end - start)

            adjacents = sorted(self.edges[expanded], key=heuristic)
            if edge_visit_count == len(adjacents):
                if len(edge) >= 2:
                    edge_limit_cost = limit_cost(edge[-2][0], edge[-1][0])
                    cost -= edge_limit_cost

                edge.pop()
            else:
                edge[-1] = (expanded, edge_visit_count + 1)

                to_visit = adjacents[edge_visit_count]
                if to_visit not in visited and can_pass(expanded, to_visit):
                    total_limit_cost = cost + limit_cost(expanded, to_visit)
                    if total_limit_cost <= cost_limit:
                        cost = total_limit_cost
                        edge.append((adjacents[edge_visit_count], 0))

        end = time.process_time()
        return SearchResults(None, None, None, list(visited), end - start)

    def raw_astar(self,
                  source: int,
                  target: int,
                  cost_limit: float,
                  real_cost: Callable[[int, int], float],
                  can_pass: Callable[[int, int], bool],
                  limit_cost: Callable[[int, int], float],
                  heuristic: Callable[[int], float]) -> SearchResults:

        start = time.process_time()

        edge = [(heuristic(source), 0.0, source)]            # Nodes to be expanded
        parents: dict[int, Optional[int]] = { source: None } # To back trace the path
        limit_costs: dict[int, float] = { source: 0.0 }      # Fuel costs to each node

        while edge:
            _, g_cost_expanded, expanded = heapq.heappop(edge)
            if expanded == target:
                path = self.path_from_parents(target, parents)
                cost = self.cost_from_path(path, real_cost)
                distance = self.cost_from_path(path, lambda o, d: self.edges[o][d])
                visited = list(parents)

                end = time.process_time()
                return SearchResults(path, cost, distance, visited, end - start)

            for to_visit in self.edges[expanded]:
                if to_visit not in parents and can_pass(expanded, to_visit):
                    total_cost = limit_costs[expanded] + limit_cost(expanded, to_visit)
                    if total_cost < cost_limit:
                        g_cost_to_visit = g_cost_expanded + self.edges[expanded][to_visit]
                        h_cost_to_visit = g_cost_to_visit + heuristic(to_visit)

                        heapq.heappush(edge, (h_cost_to_visit, g_cost_to_visit, to_visit))
                        parents[to_visit] = expanded
                        limit_costs[to_visit] = total_cost


        end = time.process_time()
        return SearchResults(None, None, None, list(parents), end - start)

    def raw_search(self,
                   source: int,
                   target: int,
                   algorithm: SearchAlgorithm,
                   cost_limit: float,
                   real_cost: Callable[[int, int], float],
                   can_pass: Callable[[int, int], bool],
                   limit_cost: Callable[[int, int], float],
                   heuristic: Callable[[int], float]) -> SearchResults:

        match algorithm:
            case SearchAlgorithm.DFS:
                return self.raw_dfs(source, target, cost_limit, real_cost, can_pass, limit_cost)
            case SearchAlgorithm.ITERATIVE:
                return self.raw_iterative(source,
                                          target,
                                          cost_limit,
                                          real_cost,
                                          can_pass,
                                          limit_cost)
            case SearchAlgorithm.BFS:
                return self.raw_bfs(source, target, cost_limit, real_cost, can_pass, limit_cost)
            case SearchAlgorithm.DIJKSTRA:
                return self.raw_dijkstra(source,
                                         target,
                                         cost_limit,
                                         real_cost,
                                         can_pass,
                                         limit_cost)
            case SearchAlgorithm.GREEDY:
                return self.raw_greedy(source,
                                       target,
                                       cost_limit,
                                       real_cost,
                                       can_pass,
                                       limit_cost,
                                       heuristic)
            case SearchAlgorithm.ASTAR:
                return self.raw_astar(source,
                                      target,
                                      cost_limit,
                                      real_cost,
                                      can_pass,
                                      limit_cost,
                                      heuristic)

    def path_from_parents(self, target: int, parents: dict[int, Optional[int]]) -> list[int]:
        path = [target]

        while parents[target] is not None:
            next_vertex = cast(int, parents[target])
            path.append(next_vertex)
            target = next_vertex

        return path[::-1]

    def cost_from_path(self, path: list[int], real_cost: Callable[[int, int], float]) -> float:
        cost = 0.0
        i = 0
        while i < len(path) - 1:
            cost += real_cost(path[i], path[i + 1])
            i += 1

        return cost
