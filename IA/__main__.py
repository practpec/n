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

from IA.Map import Map
from IA.Problem import EventSequence
from IA.UI import UI
from IA.Vehicle import Person, Car, Motorcycle

def main() -> None:
    pmap: Map = Map('Braga (São Vítor)')

    results1 = pmap.bfs(4643306970, 12085192464, Car())
    print('cost:', results1.cost, 's')
    print('distance:', results1.distance, 'm')

    results2 = pmap.bfs(4643306970, 12085192464, Motorcycle())
    print('cost:', results2.cost, 's')
    print('distance:', results2.distance, 'm')

    results3 = pmap.bfs(4643306970, 12085192464, Person())
    print('cost:', results3.cost, 's')
    print('distance:', results3.distance, 'm')

    seq: EventSequence = [
        ('Press RETURN to advance simulation', None),
        ('Carro', results1),
        ('Mota', results2),
        ('Pessoa', results3),
    ]
    UI(pmap, seq)

if __name__ == '__main__':
    main()
