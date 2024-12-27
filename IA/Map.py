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
from typing import NamedTuple

from bs4 import BeautifulSoup
import requests

from IA.Graph import Graph

EARTH_RADIUS = 6378137.0 # meters

class Point(NamedTuple):
    x: float
    y: float

class Map(Graph):
    def __init__(self, location: str) -> None:
        super().__init__()
        self.coordinates: dict[int, Point] = {}

        xml_text = self.cached_request(location)
        soup = BeautifulSoup(xml_text, 'xml')
        self.add_nodes_in_xml(soup)
        self.add_ways_in_xml(soup)

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

    def add_ways_in_xml(self, soup: BeautifulSoup) -> None:
        for way_element in soup.find_all('way'):
            children = way_element.find_all('nd')

            i = 0
            while i < len(children) - 1:
                ref1 = int(children[i]['ref'])
                ref2 = int(children[i + 1]['ref'])
                source = self.coordinates[ref1]
                target = self.coordinates[ref2]
                cost = ((source.x - target.y) ** 2 + (source.y - target.y) ** 2) ** 0.5

                self.edges[ref1][ref2] = cost
                i += 1
