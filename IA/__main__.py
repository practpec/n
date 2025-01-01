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

from pprint import pprint

from IA.Graph import SearchAlgorithm
from IA.Map import Map, SearchHeuristic
from IA.Problem import EventSequence
from IA.UI import UI
from IA.Distribution import Person, Car, Motorcycle, DistributionCenter, DeliveryTarget, Product
from IA.BinPacking import genetic_bin_pack

def main() -> None:
    vehicle_costs = { Car(): 100.0, Motorcycle(): 1000.0, Person(): 10.0 }
    vehicles = { Car(): 1, Person(): 1 }
    products = { Product('alface', 3, (0, 255, 0)): 10, Product('gasoil', 5, (255, 0, 0)): 2 }
    bin_packing = genetic_bin_pack(vehicle_costs, vehicles, products)
    pprint(bin_packing)

    pmap = Map('Braga (São Vítor)', DistributionCenter(439359565))
    pmap.delivery_targets.append(DeliveryTarget('Tribunal', 1605296772))
    pmap.delivery_targets.append(DeliveryTarget('Escola Secundária Alberto Sampaio', 4090416170))
    pmap.delivery_targets.append(DeliveryTarget('Instituto de Nanotecnologia', 1607596806))
    pmap.delivery_targets.append(DeliveryTarget('Cemitério', 2361505854))
    pmap.delivery_targets.append(DeliveryTarget('Universidade do Minho', 227110575))
    pmap.delivery_targets.append(DeliveryTarget('Olimpo', 1193997031))
    pmap.delivery_targets.append(DeliveryTarget('Happy China', 4674683655))

    results1 = pmap.search(4643306970, 2681450633, SearchAlgorithm.DFS, Car())
    print('cost:', results1.cost, 's')
    print('distance:', results1.distance, 'm')

    results0 = pmap.search(4643306970, 2681450633, SearchAlgorithm.ITERATIVE, Car())
    print('cost:', results0.cost, 's')
    print('distance:', results0.distance, 'm')

    results2 = pmap.search(4643306970, 12085192464, SearchAlgorithm.BFS, Car())
    print('cost:', results2.cost, 's')
    print('distance:', results2.distance, 'm')

    results3 = pmap.search(4643306970, 12085192464, SearchAlgorithm.DIJKSTRA, Car())
    print('cost:', results3.cost, 's')
    print('distance:', results3.distance, 'm')

    results4 = pmap.search(4643306970,
                           12085192464,
                           SearchAlgorithm.GREEDY,
                           Car(),
                           SearchHeuristic.CARTESIAN)
    print('cost:', results3.cost, 's')
    print('distance:', results3.distance, 'm')

    results5 = pmap.search(4643306970,
                           12085192464,
                           SearchAlgorithm.GREEDY,
                           Car(),
                           SearchHeuristic.MANHATTAN)
    print('cost:', results4.cost, 's')
    print('distance:', results4.distance, 'm')

    results6 = pmap.search(4643306970,
                           12085192464,
                           SearchAlgorithm.ASTAR,
                           Car(),
                           SearchHeuristic.CARTESIAN)
    print('cost:', results6.cost, 's')
    print('distance:', results6.distance, 'm')

    results7 = pmap.search(4643306970,
                           12085192464,
                           SearchAlgorithm.ASTAR,
                           Car(),
                           SearchHeuristic.MANHATTAN)
    print('cost:', results7.cost, 's')
    print('distance:', results7.distance, 'm')

    seq: EventSequence = [
        ('Press RETURN to advance simulation', None),
        # ('Bin packing', bin_packing),
        ('DFS', results1),
        ('Iterative', results0),
        ('BFS', results2),
        ('Dijkstra', results3),
        # ('Greedy (cartesian heuristic)', results4),
        # ('Greedy (manhattan heuristic)', results5),
        ('A* (cartesian heuristic)', results6),
        ('A* (manhattan heuristic)', results7),
    ]
    UI(pmap, seq)

if __name__ == '__main__':
    main()
