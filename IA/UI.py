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

from IA.BinPacking import BinPackingResult
from IA.Graph import SearchResults
from IA.Map import Map, Point
from IA.Problem import EventSequence

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

class UI:
    def __init__(self, pmap: Map, seq: EventSequence) -> None:
        self.map = pmap
        self.seq = seq
        self.seq_position = 0
        self.showing_weather = False

        self.generate_initial_projection(pmap)

        pygame.init()
        self.vehicles = {
            'Person': pygame.image.load('res/person.bmp'),
            'Motorcycle': pygame.image.load('res/motorcycle.bmp'),
            'Car': pygame.image.load('res/car.bmp')
        }

        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(self.seq[self.seq_position][0])
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

            seq_event = self.seq[self.seq_position][1]

            self.window.fill((0, 0, 0))
            if not isinstance(seq_event, BinPackingResult):
                self.render_map(seq_event is None)
            else:
                self.render_bin_packing_results(seq_event)

            if isinstance(seq_event, SearchResults):
                self.render_search_results(seq_event)

            pygame.display.flip()

    def handle_event(self, event: pygame.event.Event) -> None:
        zoom_factor = 1.1

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.MOUSEWHEEL:
            self.zoom *= zoom_factor if event.y > 0 else 1 / zoom_factor
        elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            lowest_distance_node = next(iter(self.map.edges))
            lowest_distance_sq = float('inf')
            for node in self.map.edges:
                x, y = self.point_to_screen(self.map.coordinates[node])
                dist_sq = (x - mx) ** 2 + (y - my) ** 2

                if dist_sq < lowest_distance_sq:
                    lowest_distance_node = node
                    lowest_distance_sq = dist_sq

            print(f'https://www.openstreetmap.org/node/{lowest_distance_node}')

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
            elif event.key == pygame.K_w:
                self.showing_weather = not self.showing_weather
            elif event.key == pygame.K_RETURN:
                self.seq_position = min(self.seq_position + 1, len(self.seq) - 1)
                pygame.display.set_caption(self.seq[self.seq_position][0])
            elif event.key == pygame.K_BACKSPACE:
                self.seq_position = max(self.seq_position - 1, 0)
                pygame.display.set_caption(self.seq[self.seq_position][0])

    def render_map(self, draw_way_points: bool) -> None:
        for edge_source, edge_targets in self.map.edges.items():
            for edge_target in edge_targets:
                source = self.point_to_screen(self.map.coordinates[edge_source])
                target = self.point_to_screen(self.map.coordinates[edge_target])

                if self.showing_weather:
                    weather = self.map.weather[(edge_source, edge_target)] * 255
                    edge_color = (255, round(255 - weather), round(255 - weather))
                else:
                    edge_color = (155, 155, 155)

                pygame.draw.line(self.window, edge_color, source, target)

        if draw_way_points:
            self.render_triangle(self.map.coordinates[self.map.distribution_center.node],
                                 (0, 255, 255))

            for delivery_target in self.map.delivery_targets:
                self.render_triangle(self.map.coordinates[delivery_target.node], (255, 0, 0))

    def render_triangle(self, point: Point, color: tuple[int, int, int]) -> None:
        x, y = self.point_to_screen(point)
        pygame.draw.polygon(self.window, color, [ (x, y), (x - 5, y - 10), (x + 5, y - 10) ])

    def render_search_results(self, res: SearchResults) -> None:
        for node in res.visited:
            pos = self.point_to_screen(self.map.coordinates[node])
            color = (0, 255, 255) if res.path is not None and node in res.path else (255, 0, 0)
            pygame.draw.circle(self.window, color, pos, 3)

        if res.path is not None:
            i = 0
            while i < len(res.path) - 1:
                source = self.point_to_screen(self.map.coordinates[res.path[i]])
                target = self.point_to_screen(self.map.coordinates[res.path[i + 1]])

                pygame.draw.line(self.window, (0, 255, 255), source, target, 2)
                i += 1

    def render_bin_packing_results(self, res: BinPackingResult) -> None:
        bar_translate = (WINDOW_WIDTH - (len(res.results) * 96 - 32)) / 2
        max_vehicle_weight = max(vehicle.max_weight for vehicle, _ in res.results)

        for i, (vehicle, products) in enumerate(res.results):
            left = bar_translate + i * 96
            rect_height = (WINDOW_HEIGHT - 200) * (vehicle.max_weight / max_vehicle_weight)
            rect = pygame.Rect(left, WINDOW_HEIGHT - 100 - rect_height, 64, rect_height)
            height_scale_factor = rect_height / vehicle.max_weight

            pygame.draw.rect(self.window, (255, 255, 255), rect, 1)
            self.window.blit(self.vehicles[type(vehicle).__name__], (left, WINDOW_HEIGHT - 90))

            # Fill colors
            product_start = float(WINDOW_HEIGHT - 100)
            for product in products:
                product_height = product.weight * height_scale_factor
                product_start -= product_height

                rect = pygame.Rect(left + 1, product_start, 62, product_height)
                pygame.draw.rect(self.window, product.color, rect)

            # Draw separator lines
            product_start = WINDOW_HEIGHT - 100
            for product in products:
                product_height = product.weight * height_scale_factor
                product_start -= product_height
                pygame.draw.line(self.window,
                                 (255, 255, 255),
                                 (left, product_start),
                                 (left + 63, product_start))

    def point_to_screen(self, p: Point) -> tuple[float, float]:
        initial_x = (p.x - self.translatex) * self.scale
        initial_y = (p.y - self.translatey) * self.scale

        half_window_width = WINDOW_WIDTH / 2
        half_window_height = WINDOW_HEIGHT / 2

        zoomed_x = (initial_x - half_window_width) * self.zoom + half_window_width
        zoomed_y = (initial_y - half_window_height) * self.zoom + half_window_height

        return zoomed_x, zoomed_y
