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

import sys

import pygame

from IA.Map import Map, Point

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

class UI:
    def __init__(self, pmap: Map) -> None:
        self.map = pmap

        self.generate_initial_projection(pmap)

        pygame.init()
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.game_loop()

    def generate_initial_projection(self, pmap: Map) -> None:
        minx = min(point.x for point in pmap.coordinates.values())
        maxx = max(point.x for point in pmap.coordinates.values())
        miny = min(point.y for point in pmap.coordinates.values())
        maxy = max(point.y for point in pmap.coordinates.values())

        scalex = abs(WINDOW_WIDTH / (maxx - minx))
        scaley = abs(WINDOW_HEIGHT / (maxy - miny))

        if scalex < scaley:
            self.movement_unit = 0.05 * (maxx - minx)
            self.scale = scalex
        else:
            self.movement_unit = 0.05 * (maxy - miny)
            self.scale = scaley

        self.translatex = minx
        self.translatey = miny
        self.zoom = 0.90

    def game_loop(self) -> None:
        while True:
            for event in pygame.event.get():
                self.handle_event(event)

            self.window.fill((0, 0, 0))

            for edge_source, edge_targets in self.map.edges.items():
                for edge_target in edge_targets:
                    source = self.point_to_screen(self.map.coordinates[edge_source])
                    target = self.point_to_screen(self.map.coordinates[edge_target])

                    pygame.draw.line(self.window, (255, 255, 255), source, target)

            pygame.display.flip()

    def handle_event(self, event: pygame.event.Event) -> None:
        zoom_factor = 1.1

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.MOUSEWHEEL:
            self.zoom *= zoom_factor if event.y > 0 else 1 / zoom_factor
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.translatex -= self.movement_unit / self.zoom
            elif event.key == pygame.K_RIGHT:
                self.translatex += self.movement_unit / self.zoom
            elif event.key == pygame.K_UP:
                self.translatey -= self.movement_unit / self.zoom
            elif event.key == pygame.K_DOWN:
                self.translatey += self.movement_unit / self.zoom
            elif event.key == pygame.K_PLUS:
                self.zoom *= zoom_factor
            elif event.key == pygame.K_MINUS:
                self.zoom /= zoom_factor

    def point_to_screen(self, p: Point) -> tuple[float, float]:
        initial_x = (p.x - self.translatex) * self.scale
        initial_y = (p.y - self.translatey) * self.scale

        half_window_width = WINDOW_WIDTH / 2
        half_window_height = WINDOW_HEIGHT / 2

        zoomed_x = (initial_x - half_window_width) * self.zoom + half_window_width
        zoomed_y = (initial_y - half_window_height) * self.zoom + half_window_height

        return zoomed_x, zoomed_y
