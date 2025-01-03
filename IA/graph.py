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
    DFS_FAST = 1
    DFS_CORRECT = 2
    ITERATIVE = 3
    BFS = 4
    DIJKSTRA = 5
    GREEDY = 6
    ASTAR = 7

class CanPass(Enum):
    NO = 0
    NO_WITH_FUEL_LOSS = 1
    YES = 2

class Graph:
    def __init__(self) -> None:
        self.edges: dict[int, dict[int, float]] = {}

    def raw_dfs_fast(self,
                     source: int,
                     target: int,
                     cost_limit: float,
                     real_cost: Callable[[int, int], float],
                     can_pass: Callable[[int, int], CanPass],
                     limit_cost: Callable[[int, int], float]) -> SearchResults:

        start = time.process_time()

        edge = [(source, 0)]          # Algorithm's stack (because Python's stack is very small)
        cost: float = 0.0             # Limit (fuel) cost
        visited: dict[int, None] = {} # Ordered set

        count_backtracking_fuel: bool = False # Weather to count cost of backtracking
        backtracking_cost: float = 0.0        # Fuel cost in backtracking due to new weather info
        backtracking_distance: float = 0.0    # Backtracked distance due to new weather info

        while edge:
            expanded, edge_visit_count = edge[-1]
            visited[expanded] = None

            if expanded == target:
                path = [v for v, _ in edge]
                cost = self.cost_from_path(path, real_cost) + backtracking_cost
                distance = \
                    self.cost_from_path(path, lambda o, d: self.edges[o][d]) + backtracking_distance
                visited_ret = list(visited)

                end = time.process_time()
                return SearchResults(path, cost, distance, visited_ret, end - start)

            adjacents = list(self.edges[expanded])
            if edge_visit_count == len(adjacents):
                if len(edge) >= 2:
                    edge_limit_cost = limit_cost(edge[-2][0], edge[-1][0])
                    if not count_backtracking_fuel:
                        cost -= edge_limit_cost
                    else:
                        backtracking_cost += real_cost(edge[-2][0], edge[-1][0])
                        backtracking_distance += self.edges[edge[-2][0]][edge[-1][0]]

                edge.pop()
            else:
                edge[-1] = (expanded, edge_visit_count + 1)
                to_visit = adjacents[edge_visit_count]
                pass_permission = can_pass(expanded, to_visit)

                if pass_permission == CanPass.NO_WITH_FUEL_LOSS:
                    count_backtracking_fuel = True

                if to_visit not in visited and pass_permission == CanPass.YES:
                    count_backtracking_fuel = False

                    total_limit_cost = cost + limit_cost(expanded, to_visit)
                    if total_limit_cost <= cost_limit:
                        cost = total_limit_cost
                        edge.append((adjacents[edge_visit_count], 0))

        end = time.process_time()
        return SearchResults(None, None, None, list(visited), end - start)

    def raw_dfs_correct(self,
                        source: int,
                        target: int,
                        cost_limit: float,
                        real_cost: Callable[[int, int], float],
                        can_pass: Callable[[int, int], CanPass],
                        limit_cost: Callable[[int, int], float]) -> SearchResults:

        start = time.process_time()

        edge = [(source, 0)]          # Algorithm's stack (because Python's stack is very small)
        cost: float = 0.0             # Limit (fuel) cost
        visited: dict[int, None] = {} # Ordered set

        count_backtracking_fuel: bool = False # Weather to count cost of backtracking
        backtracking_cost: float = 0.0        # Fuel cost in backtracking due to new weather info
        backtracking_distance: float = 0.0    # Backtracked distance due to new weather info

        while edge:
            expanded, edge_visit_count = edge[-1]
            visited[expanded] = None

            path = [v for v, _ in edge]
            if expanded == target:
                cost = self.cost_from_path(path, real_cost) + backtracking_cost
                distance = \
                    self.cost_from_path(path, lambda o, d: self.edges[o][d]) + backtracking_distance
                visited_ret = list(visited)

                end = time.process_time()
                return SearchResults(path, cost, distance, visited_ret, end - start)

            adjacents = list(self.edges[expanded])
            if edge_visit_count == len(adjacents):
                if len(edge) >= 2:
                    edge_limit_cost = limit_cost(edge[-2][0], edge[-1][0])
                    if not count_backtracking_fuel:
                        cost -= edge_limit_cost
                    else:
                        backtracking_cost += real_cost(edge[-2][0], edge[-1][0])
                        backtracking_distance += self.edges[edge[-2][0]][edge[-1][0]]

                edge.pop()
            else:
                edge[-1] = (expanded, edge_visit_count + 1)
                to_visit = adjacents[edge_visit_count]
                pass_permission = can_pass(expanded, to_visit)

                if pass_permission == CanPass.NO_WITH_FUEL_LOSS:
                    count_backtracking_fuel = True

                if to_visit not in path and pass_permission == CanPass.YES:
                    count_backtracking_fuel = False

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
                      can_pass: Callable[[int, int], CanPass],
                      limit_cost: Callable[[int, int], float]) -> SearchResults:

        start = time.process_time()

        max_depth = 0
        last_visited = -1
        while True:
            results = self.raw_dfs_correct(source,
                                           target,
                                           max_depth,
                                           real_cost,
                                           can_pass,
                                           lambda s, t: 1.0)

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
                can_pass: Callable[[int, int], CanPass],
                limit_cost: Callable[[int, int], float]) -> SearchResults:

        start = time.process_time()

        edge = deque([source])                               # Nodes to be expanded
        parents: dict[int, Optional[int]] = { source: None } # To back trace the path
        limit_costs: dict[int, float] = { source: 0.0 }      # Fuel costs to each node

        backtracking_cost: float = 0.0     # Fuel cost in backtracking due to new weather info
        backtracking_distance: float = 0.0 # Backtracked distance due to new weather info

        while edge:
            expanded = edge[0]
            if expanded == target:
                path = self.path_from_parents(target, parents)
                cost = self.cost_from_path(path, real_cost) + backtracking_cost
                distance = self.cost_from_path(path, lambda o, d: self.edges[o][d]) + backtracking_distance
                visited = list(parents)

                end = time.process_time()
                return SearchResults(path, cost, distance, visited, end - start)

            edge.popleft()
            visitable_nodes = 0
            not_visited_nodes = 0

            for to_visit in self.edges[expanded]:
                pass_permission = can_pass(expanded, to_visit)
                if to_visit not in parents:
                    visitable_nodes += 1

                    if pass_permission == CanPass.YES:
                        total_cost = limit_costs[expanded] + limit_cost(expanded, to_visit)
                        if total_cost < cost_limit:
                            edge.append(to_visit)
                            parents[to_visit] = expanded
                            limit_costs[to_visit] = total_cost
                    elif pass_permission == CanPass.NO_WITH_FUEL_LOSS:
                        not_visited_nodes += 1

            if visitable_nodes == not_visited_nodes and not_visited_nodes > 0 and edge:
                # Nowhere to go due to bad randomly generated weather
                path_back, path_forward = self.backtrack_path(expanded, edge[0], parents)

                backtracking_cost += self.cost_from_path(path_back, real_cost)
                backtracking_cost += self.cost_from_path(path_forward, real_cost)

                backtracking_distance += self.cost_from_path(path_back, lambda o, d: self.edges[o][d])
                backtracking_distance += self.cost_from_path(path_forward, lambda o, d: self.edges[o][d])

        end = time.process_time()
        return SearchResults(None, None, None, list(parents), end - start)

    def raw_dijkstra(self,
                     source: int,
                     target: int,
                     cost_limit: float,
                     real_cost: Callable[[int, int], float],
                     can_pass: Callable[[int, int], CanPass],
                     limit_cost: Callable[[int, int], float]) -> SearchResults:

        start = time.process_time()

        edge = [(0.0, source)]                               # Nodes to be expanded
        parents: dict[int, Optional[int]] = { source: None } # To back trace the path
        costs: dict[int, float] = { source: 0.0 }            # Costs from source to nodes
        definitive: set[int] = set()                         # Costs for these can't be changed
        limit_costs: dict[int, float] = { source: 0.0 }      # Fuel costs to each node

        backtracking_cost: float = 0.0     # Fuel cost in backtracking due to new weather info
        backtracking_distance: float = 0.0 # Backtracked distance due to new weather info

        while edge:
            cost_to_expanded, expanded = heapq.heappop(edge)
            definitive.add(expanded)

            if expanded == target:
                path = self.path_from_parents(target, parents)
                cost = self.cost_from_path(path, real_cost) + backtracking_cost
                distance = self.cost_from_path(path, lambda o, d: self.edges[o][d]) + backtracking_distance
                visited = list(parents)

                end = time.process_time()
                return SearchResults(path, cost, distance, visited, end - start)

            visitable_nodes = 0
            not_visited_nodes = 0

            for to_visit in self.edges[expanded]:
                pass_permission = can_pass(expanded, to_visit)
                if to_visit not in definitive:
                    visitable_nodes += 1

                    if pass_permission == CanPass.YES:
                        total_cost = limit_costs[expanded] + limit_cost(expanded, to_visit)
                        if total_cost < cost_limit:
                            cost_to_visit = cost_to_expanded + real_cost(expanded, to_visit)
    
                            if to_visit not in costs:
                                parents[to_visit] = expanded
                                costs[to_visit] = cost_to_visit
                                limit_costs[to_visit] = total_cost
                                heapq.heappush(edge, (cost_to_visit, to_visit))
                            elif cost_to_visit < costs[to_visit]:
                                parents[to_visit] = expanded
                                costs[to_visit] = cost_to_visit
    
                                # Decrease-key min heap (can be done in log(n) time but I'm stupid)
                                for i, (_, node) in enumerate(edge):
                                    if node == to_visit:
                                        edge[i] = (cost_to_visit, to_visit)
                                heapq.heapify(edge)
    
                                limit_costs[to_visit] = min(total_cost, limit_costs[to_visit])

                    elif pass_permission == CanPass.NO_WITH_FUEL_LOSS:
                        not_visited_nodes += 1

            if visitable_nodes == not_visited_nodes and not_visited_nodes > 0 and edge:
                # Nowhere to go due to bad randomly generated weather
                path_back, path_forward = self.backtrack_path(expanded, edge[0][1], parents)

                backtracking_cost += self.cost_from_path(path_back, real_cost)
                backtracking_cost += self.cost_from_path(path_forward, real_cost)

                backtracking_distance += self.cost_from_path(path_back, lambda o, d: self.edges[o][d])
                backtracking_distance += self.cost_from_path(path_forward, lambda o, d: self.edges[o][d])

        end = time.process_time()
        return SearchResults(None, None, None, list(parents), end - start)

    def raw_greedy(self,
                   source: int,
                   target: int,
                   cost_limit: float,
                   real_cost: Callable[[int, int], float],
                   can_pass: Callable[[int, int], CanPass],
                   limit_cost: Callable[[int, int], float],
                   heuristic: Callable[[int], float]) -> SearchResults:

        start = time.process_time()

        edge = [(source, 0)]          # Algorithm's stack (because Python's stack is very small)
        cost: float = 0.0             # Limit (fuel) cost
        visited: dict[int, None] = {} # Ordered set

        count_backtracking_fuel: bool = False # Weather to count cost of backtracking
        backtracking_cost: float = 0.0        # Fuel cost in backtracking due to new weather info
        backtracking_distance: float = 0.0    # Backtracked distance due to new weather info

        while edge:
            expanded, edge_visit_count = edge[-1]
            visited[expanded] = None

            if expanded == target:
                path = [v for v, _ in edge]
                cost = self.cost_from_path(path, real_cost) + backtracking_cost
                distance = self.cost_from_path(path, lambda o, d: self.edges[o][d]) + backtracking_distance
                visited_ret = list(visited)

                end = time.process_time()
                return SearchResults(path, cost, distance, visited_ret, end - start)

            adjacents = sorted(self.edges[expanded], key=heuristic)
            if edge_visit_count == len(adjacents):
                if len(edge) >= 2:
                    edge_limit_cost = limit_cost(edge[-2][0], edge[-1][0])
                    if not count_backtracking_fuel:
                        cost -= edge_limit_cost
                    else:
                        backtracking_cost += real_cost(edge[-2][0], edge[-1][0])
                        backtracking_distance += self.edges[edge[-2][0]][edge[-1][0]]

                edge.pop()
            else:
                edge[-1] = (expanded, edge_visit_count + 1)
                to_visit = adjacents[edge_visit_count]
                pass_permission = can_pass(expanded, to_visit)

                if pass_permission == CanPass.NO_WITH_FUEL_LOSS:
                    count_backtracking_fuel = True

                if to_visit not in visited and pass_permission == CanPass.YES:
                    count_backtracking_fuel = False

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
                  can_pass: Callable[[int, int], CanPass],
                  limit_cost: Callable[[int, int], float],
                  heuristic: Callable[[int], float]) -> SearchResults:

        start = time.process_time()

        edge = [(heuristic(source), 0.0, source)]            # Nodes to be expanded
        parents: dict[int, Optional[int]] = { source: None } # To back trace the path
        costs: dict[int, float] = { source: 0.0 }            # Costs from source to nodes
        definitive: set[int] = set()                         # Costs for these can't be changed
        limit_costs: dict[int, float] = { source: 0.0 }      # Fuel costs to each node

        backtracking_cost: float = 0.0     # Fuel cost in backtracking due to new weather info
        backtracking_distance: float = 0.0 # Backtracked distance due to new weather info

        while edge:
            _, g_cost_expanded, expanded = heapq.heappop(edge)
            definitive.add(expanded)

            if expanded == target:
                path = self.path_from_parents(target, parents)
                cost = self.cost_from_path(path, real_cost) + backtracking_cost
                distance = self.cost_from_path(path, lambda o, d: self.edges[o][d]) + backtracking_distance
                visited = list(parents)

                end = time.process_time()
                return SearchResults(path, cost, distance, visited, end - start)

            visitable_nodes = 0
            not_visited_nodes = 0

            for to_visit in self.edges[expanded]:
                pass_permission = can_pass(expanded, to_visit)
                if to_visit not in definitive:
                    visitable_nodes += 1

                    if pass_permission == CanPass.YES:
                        total_cost = limit_costs[expanded] + limit_cost(expanded, to_visit)
                        if total_cost < cost_limit:
                            g_cost_to_visit = g_cost_expanded + real_cost(expanded, to_visit)
                            h_cost_to_visit = g_cost_to_visit + heuristic(to_visit)

                            if to_visit not in costs:
                                parents[to_visit] = expanded
                                costs[to_visit] = g_cost_to_visit
                                limit_costs[to_visit] = total_cost
                                heapq.heappush(edge, (h_cost_to_visit, g_cost_to_visit, to_visit))
                            elif g_cost_to_visit < costs[to_visit]:
                                parents[to_visit] = expanded
                                costs[to_visit] = g_cost_to_visit

                                # Decrease-key min heap (can be done in log(n) time but I'm stupid)
                                for i, (_h, _g, node) in enumerate(edge):
                                    if node == to_visit:
                                        edge[i] = (h_cost_to_visit, g_cost_to_visit, to_visit)
                                heapq.heapify(edge)

                                limit_costs[to_visit] = min(total_cost, limit_costs[to_visit])
                    elif pass_permission == CanPass.NO_WITH_FUEL_LOSS:
                        not_visited_nodes += 1

            if visitable_nodes == not_visited_nodes and not_visited_nodes > 0 and edge:
                # Nowhere to go due to bad randomly generated weather
                path_back, path_forward = self.backtrack_path(expanded, edge[0][2], parents)

                backtracking_cost += self.cost_from_path(path_back, real_cost)
                backtracking_cost += self.cost_from_path(path_forward, real_cost)

                backtracking_distance += self.cost_from_path(path_back, lambda o, d: self.edges[o][d])
                backtracking_distance += self.cost_from_path(path_forward, lambda o, d: self.edges[o][d])

        end = time.process_time()
        return SearchResults(None, None, None, list(parents), end - start)

    def raw_search(self,
                   source: int,
                   target: int,
                   algorithm: SearchAlgorithm,
                   cost_limit: float,
                   real_cost: Callable[[int, int], float],
                   can_pass: Callable[[int, int], CanPass],
                   limit_cost: Callable[[int, int], float],
                   heuristic: Callable[[int], float]) -> SearchResults:

        match algorithm:
            case SearchAlgorithm.DFS_FAST:
                return self.raw_dfs_fast(source,
                                         target,
                                         cost_limit,
                                         real_cost,
                                         can_pass,
                                         limit_cost)
            case SearchAlgorithm.DFS_CORRECT:
                return self.raw_dfs_correct(source,
                                            target,
                                            cost_limit,
                                            real_cost,
                                            can_pass,
                                            limit_cost)
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

    def path_from_parents_until(self, target: int, max_parent: int, parents: dict[int, Optional[int]]) -> list[int]:
        path = [target]

        while parents[target] is not None:
            next_vertex = cast(int, parents[target])
            path.append(next_vertex)
            target = next_vertex

            if next_vertex == max_parent:
                break

        return path[::-1]

    def cost_from_path(self, path: list[int], real_cost: Callable[[int, int], float]) -> float:
        cost = 0.0
        i = 0
        while i < len(path) - 1:
            cost += real_cost(path[i], path[i + 1])
            i += 1

        return cost

    def find_first_common_parent(self, parents: dict[int, Optional[int]], n1: int, n2: int) -> int:
        n1_path = self.path_from_parents(n1, parents)
        n2_path = set(self.path_from_parents(n2, parents))

        for i in range(-1, -len(n1_path) -1, -1):
            if n1_path[i] in n2_path:
                return n1_path[i]

        return -1 # unreachable

    def backtrack_path(self, n1: int, n2: int, parents: dict[int, Optional[int]]) -> tuple[list[int], list[int]]:
        common_parent = self.find_first_common_parent(parents, n1, n2)
        path_back = self.path_from_parents_until(n1, common_parent, parents)[::-1]
        path_forward = self.path_from_parents_until(n2, common_parent, parents)
        return path_back[::-1], path_forward
