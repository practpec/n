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

@dataclass
class Vehicle:
    name: str            # Vehicle name
    max_fuel: float      # km (adusted to weather)
    max_weight: float    # kg
    worst_weather: float # [0, 1]
    speed: float         # m/s
    image: str           # Path to image (64x64 BMP)

    def calculate_spent_fuel(self, cost: float, weather: float) -> float:
        return cost * (1 + weather) # Spend more fuel when weather is worse

    def calculate_travel_time(self, cost: float, weather: float) -> float:
        return (cost * (1 + weather)) / self.speed

    def __hash__(self) -> int:
        return hash(self.name)

@dataclass
class Product:
    name: str
    weight: float
    color: tuple[int, int, int]

    def __hash__(self) -> int:
        return hash(self.name)

@dataclass
class BinPackingResult:
    results: list[tuple[Vehicle, list[Product]]]

@dataclass
class DistributionCenter:
    name: str                    # Presentation name
    node: int                    # OSM node
    vehicles: dict[Vehicle, int] # Number available of each vehicle
    products: dict[Product, int] # Number available of each product

    def discount_bin_pack(self, bp_results: BinPackingResult) -> None:
        for vehicle, products in bp_results.results:
            self.vehicles[vehicle] -= 1
            if self.vehicles[vehicle] == 0:
                del self.vehicles[vehicle]

            for product in products:
                self.products[product] -= 1
                if self.products[product] == 0:
                    del self.products[product]


@dataclass
class DeliveryTarget:
    name: str                    # Presentation name
    node: int                    # OSM node
    time_limit: float            # in seconds
    products: dict[Product, int] # Number needed o each product

    def enough_resources_to_supply(self, center: DistributionCenter) -> bool:
        for product, n in self.products.items():
            if center.products[product] < n:
                return False
        return True

    def discount_bin_pack(self, bp_results: BinPackingResult) -> None:
        for _, products in bp_results.results:
            for product in products:
                self.products[product] -= 1
                if self.products[product] == 0:
                    del self.products[product]
