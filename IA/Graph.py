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

from typing import Callable, NamedTuple, Optional, cast

class SearchResults(NamedTuple):
    path: Optional[list[int]]
    cost: Optional[float]
    visited: list[int]

class Graph:
    def __init__(self) -> None:
        self.edges: dict[int, dict[int, float]] = {}

    # pylint: disable-next=too-many-arguments too-many-positional-arguments
    def raw_bfs(self,
                source: int,
                target: int,
                cost_limit: float,
                can_pass: Callable[[int, int], bool],
                limit_cost: Callable[[int, int], float]) -> SearchResults:

        edge = deque([source])
        parents: dict[int, Optional[int]] = { source: None }
        costs: dict[int, float] = { source: 0.0 }

        while edge:
            expanded = edge[0]
            if expanded == target:
                path = self.path_from_parents(target, parents)
                cost = self.cost_from_path(path)
                visited = list(parents)
                return SearchResults(path, cost, visited)

            edge.popleft()
            for to_visit in self.edges[expanded]:
                if to_visit not in parents and can_pass(expanded, to_visit):
                    total_cost = costs[expanded] + limit_cost(expanded, to_visit)
                    if total_cost < cost_limit:
                        edge.append(to_visit)
                        parents[to_visit] = expanded
                        costs[to_visit] = total_cost

        return SearchResults(None, None, list(parents))

    def path_from_parents(self, target: int, parents: dict[int, Optional[int]]) -> list[int]:
        path = [target]

        while parents[target] is not None:
            next_vertex = cast(int, parents[target])
            path.append(next_vertex)
            target = next_vertex

        return path[::-1]

    def cost_from_path(self, path: list[int]) -> float:
        cost = 0.0
        i = 0
        while i < len(path) - 1:
            cost += self.edges[path[i]][path[i + 1]]
            i += 1

        return cost
