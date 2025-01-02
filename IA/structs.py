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
class DistributionCenter:
    name: str
    node: int
    vehicles: dict[Vehicle, int]

@dataclass
class DeliveryTarget:
    name: str
    node: int
    products: dict[Product, int]
