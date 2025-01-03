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

import pickle
import sys

from IA.problem import Problem
from IA.ui import UI

def usage() -> None:
    print('''
Usages:
  python -m IA simulate problem.json [output.pickle]
  python -m IA replay   problem.json simulation.pickle
  python -m IA test     problem.json
    ''', file = sys.stderr)

    sys.exit(1)

def main(argv: list[str]) -> None:
    if len(argv) < 2:
        usage()

    if argv[0] == 'simulate' and len(argv) in [2, 3]:
        problem = Problem(argv[1])
        solution = problem.solve()

        if len(argv) == 3:
            with open(argv[2], 'wb') as f:
                pickle.dump(solution, f)

        UI(problem.map, solution)
    elif argv[0] == 'replay' and len(argv) == 3:
        problem = Problem(argv[1])
        with open(argv[2], 'rb') as f:
            solution = pickle.load(f)

        UI(problem.map, solution)
    elif argv[0] == 'test' and len(argv) == 2:
        problem = Problem(argv[1])
        solution = problem.test()
        UI(problem.map, solution)
    else:
        usage()

if __name__ == '__main__':
    main(sys.argv[1:])
