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

import math
from enum import Enum
import random
from typing import NamedTuple, Optional

from bs4 import BeautifulSoup
from perlin_noise import PerlinNoise
import requests

from IA.graph import CanPass, Graph, SearchAlgorithm, SearchResults
from IA.structs import DeliveryTarget, DistributionCenter, Vehicle

EARTH_RADIUS = 6378137.0 # meters

class Point(NamedTuple):
    x: float
    y: float

class SearchHeuristic(Enum):
    CARTESIAN = 1
    MANHATTAN = 2

class Map(Graph):
    def __init__(self,
                 location: str,
                 directed: bool,
                 enable_weather: bool,
                 distribution_center: DistributionCenter) -> None:

        super().__init__()
        self.enable_weather = enable_weather
        self.coordinates: dict[int, Point] = {}
        self.weather: dict[tuple[int, int], float] = {}
        self.distribution_center = distribution_center
        self.delivery_targets: list[DeliveryTarget] = []

        xml_text = self.cached_request(location)
        soup = BeautifulSoup(xml_text, 'xml')
        self.add_nodes_in_xml(soup)
        self.add_ways_in_xml(soup, directed)
        self.generate_weather(enable_weather)

    def cached_request(self, location: str) -> str:
        try:
            with open(f'/tmp/{location}.xml', 'r', encoding='utf-8') as f:
                return f.read()
        except IOError:
            request_result = requests.post('https://overpass-api.de/api/interpreter', data={
                'data':
                    f'[out:xml]; area[name = "{location}"]; (way(area)[highway]; ); (._;>;); out;'
            }, timeout=30)
            request_result.raise_for_status()

            with open(f'/tmp/{location}.xml', 'w', encoding='utf-8') as f:
                f.write(request_result.text)

            return request_result.text

    def add_nodes_in_xml(self, soup: BeautifulSoup) -> None:
        for node_element in soup.find_all('node'):
            # Mercator's projection
            lat = math.log(
                math.tan(math.pi / 4 + math.radians(float(node_element['lat'])) / 2)
            ) * EARTH_RADIUS
            lon = math.radians(float(node_element['lon'])) * EARTH_RADIUS

            # NOTE: It's easier to invert latitude here than to deal with a scary transformation
            # matrix in the UI
            node_id = int(node_element['id'])
            self.coordinates[node_id] = Point(lon, -lat)
            self.edges[node_id] = {}

    def add_ways_in_xml(self, soup: BeautifulSoup, directed: bool) -> None:
        for way_element in soup.find_all('way'):
            children = way_element.find_all('nd')

            i = 0
            while i < len(children) - 1:
                ref1 = int(children[i]['ref'])
                ref2 = int(children[i + 1]['ref'])
                source = self.coordinates[ref1]
                target = self.coordinates[ref2]
                cost = ((source.x - target.x) ** 2 + (source.y - target.y) ** 2) ** 0.5

                self.edges[ref1][ref2] = cost
                if not directed:
                    self.edges[ref2][ref1] = cost
                i += 1

    def generate_weather(self, enable_weather: bool) -> None:
        noise = PerlinNoise(octaves=2, seed=70)

        map_width = \
            max(point.x for point in self.coordinates.values()) - \
            min(point.x for point in self.coordinates.values())
        map_height = \
            max(point.y for point in self.coordinates.values()) - \
            min(point.y for point in self.coordinates.values())

        for source, targets in self.edges.items():
            for target in targets:
                if not enable_weather:
                    self.weather[(source, target)] = 0.0
                    continue

                x = self.coordinates[source].x / map_width
                y = self.coordinates[source].y / map_height
                weather_source = noise((x, y)) * 3

                x = self.coordinates[target].x / map_width
                y = self.coordinates[target].y / map_height
                weather_target = noise((x, y)) * 3

                weather_edge = max(0.0, min((weather_source + weather_target) / 2, 1.0))
                self.weather[(source, target)] = weather_edge

    def search(self,
               source: int,
               target: int,
               algorithm: SearchAlgorithm,
               vehicle: Vehicle,
               heuristic: SearchHeuristic = SearchHeuristic.CARTESIAN) -> SearchResults:

        def real_cost(source: int, target: int) -> float:
            distance = self.edges[source][target]
            weather = self.weather[(source, target)]
            return vehicle.calculate_travel_time(distance, weather)

        def can_pass(source: int, target: int) -> CanPass:
            if self.enable_weather and random.random() < 0.001:
                self.weather[(source, target)] = min(vehicle.worst_weather + 0.01, 1.0)
                return CanPass.NO_WITH_FUEL_LOSS

            return CanPass.YES if self.weather[(source, target)] <= vehicle.worst_weather else CanPass.NO

        def limit_cost(source: int, target: int) -> float:
            distance = self.edges[source][target]
            weather = self.weather[(source, target)]
            return vehicle.calculate_spent_fuel(distance, weather)

        def cartesian(source: int) -> float:
            source_coords = self.coordinates[source]
            target_coords = self.coordinates[target]

            # Don't take sqrt (better performance for comparison only)
            return (
                (source_coords.x - target_coords.x) ** 2 +
                (source_coords.y - target_coords.y) ** 2
            ) / vehicle.speed

        def cartesian_astar(source: int) -> float:
            source_coords = self.coordinates[source]
            target_coords = self.coordinates[target]

            return ((
                (source_coords.x - target_coords.x) ** 2 +
                (source_coords.y - target_coords.y) ** 2
            ) ** 0.5) / vehicle.speed

        def manhattan(source: int) -> float:
            source_coords = self.coordinates[source]
            target_coords = self.coordinates[target]

            return (
                abs(source_coords.x - target_coords.x) +
                abs(source_coords.y - target_coords.y)
            ) / vehicle.speed

        heuristic_function = {
            SearchHeuristic.CARTESIAN:
                cartesian_astar if algorithm == SearchAlgorithm.ASTAR else cartesian,
            SearchHeuristic.MANHATTAN: manhattan
        }[heuristic]

        return self.raw_search(source,
                               target,
                               algorithm,
                               vehicle.max_fuel,
                               real_cost,
                               can_pass,
                               limit_cost,
                               heuristic_function)

    def results_valid_for_vehicle(self, results: SearchResults, vehicle: Vehicle) -> \
        Optional[SearchResults]:

        if results.path is None or results.cost is None:
            return None

        if results.cost >= vehicle.max_fuel:
            return None

        i = 0
        cost = 0.0
        while i < len(results.path) - 1:
            distance = self.edges[results.path[i]][results.path[i + 1]]
            weather = self.weather[(results.path[i], results.path[i + 1])]
            if weather >= vehicle.worst_weather:
                return None

            i += 1
            cost += vehicle.calculate_travel_time(distance, weather)

        return SearchResults(results.path, cost, results.distance, results.visited, results.time)
