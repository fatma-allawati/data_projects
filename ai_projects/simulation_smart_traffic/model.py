#Author: Fatma Al Lawati
from __future__ import annotations
from typing import List, Tuple, Set
import random
from mesa import Model
from mesa.time import SimultaneousActivation, RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector


from agents import roadCell, trafficLight, carAgent

Coords = Tuple[int, int]

def mean_or_zero(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0

class trafficModel(Model):
    """A simple traffic simulation on a grid with multiple horizontal & vertical roads.
    Intersections have traffic lights that cycle NS/EW.
    Cars spawn at edges, move, obey lights, and exit at boundaries.
    """

    def __init__(self, 
                 width=24, 
                 height=24, 
                 n_h_roads=3, 
                 n_v_roads=3,
                 ns_green=12, 
                 ew_green=12, 
                 spawn_rate=0.15, 
                 max_cars=400, 
                 seed=None):
        super().__init__(seed=seed)
        self.grid = MultiGrid(width, height, torus=False)

        self.ns_green = int(ns_green)
        self.ew_green = int(ew_green)
        self.spawn_rate = float(spawn_rate)
        self.max_cars = int(max_cars)

        self.light_scheduler = RandomActivation(self)
        self.scheduler = SimultaneousActivation(self)

        self.cars_exited = 0
        self.total_spawned = 0
        self.reserved_positions: set[Coords] = set()

        # New: Intersection reservation dictionary
        self.intersection_reserved: dict[Coords, carAgent] = {}

        self.road_rows = self._evenly_spaced_positions(height, n_h_roads)
        self.road_cols = self._evenly_spaced_positions(width, n_v_roads)

        self._build_roads_and_lights()
        self.spawn_points = self._compute_spawn_points()

        self.datacollector = DataCollector(
            model_reporters={
                "Throughput": lambda m: m.cars_exited,
                "CarsAlive": lambda m: len([a for a in m.scheduler.agents if isinstance(a, carAgent)]),
                "AvgWaitCurrentCars": lambda m: mean_or_zero([a.wait for a in m._cars_list()]),
                "AvgWaitPerCarExited": lambda m: 0.0,
            }
        )

        # Inside trafficModel class
    def _evenly_spaced_positions(self, size: int, n: int) -> list[int]:
        """Return n positions between [2, size-3] roughly evenly spaced so roads aren't glued to borders"""
        if n <= 0:
            return []
        if n == 1:
            return [size // 2]
        gap = (size - 4) / (n - 1)
        return [int(2 + round(i * gap)) for i in range(n)]

    def _build_roads_and_lights(self):
        """Create road cells and traffic lights at intersections"""
        uid = self.next_id
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                on_h = y in self.road_rows
                on_v = x in self.road_cols
                if on_h or on_v:
                    orientation = "X" if (on_h and on_v) else ("EW" if on_h else "NS")
                    rc = roadCell(uid(), self, (x, y), orientation)
                    self.grid.place_agent(rc, (x, y))

        for y in self.road_rows:
            for x in self.road_cols:
                offset = random.randrange(self.ns_green + self.ew_green)
                tl = trafficLight(self.next_id(), self, (x, y), phase_offset=offset)
                self.grid.place_agent(tl, (x, y))
                self.light_scheduler.add(tl)

    def _compute_spawn_points(self) -> list[Tuple[Coords, str]]:
        points: list[Tuple[Coords, str]] = []

        # Horizontal roads: spawn from left/right edges
        max_x = self.grid.width - 1
        for y in self.road_rows:
            points.append(((0, y), "E"))
            points.append(((max_x, y), "W"))

        # Vertical roads: spawn from top/bottom edges
        max_y = self.grid.height - 1
        for x in self.road_cols:
            points.append(((x, 0), "S"))
            points.append(((x, max_y), "N"))

        # Remove duplicates (corners)
        unique = {}
        for p, d in points:
            unique[(p, d)] = (p, d)
        return list(unique.values())
    
    def _cars_list(self) -> List[carAgent]:
        # Only CarAgent instances in the scheduler (SimultaneousActivation)
        return [a for a in self.scheduler.agents if isinstance(a, carAgent)]

    def _count_cars(self) -> int:
        return len(self._cars_list())

    def _avg_wait_of_exited(self) -> float:
        # For simplicity we don't keep per-car exit waits; feel free to extend.
        # This metric is a placeholder. We'll use current-cars average instead.
        return 0.0
    
    def spawn_cars(self):
        if self._count_cars() >= self.max_cars:
            return
        random.shuffle(self.spawn_points)
        for (p, d) in self.spawn_points:
            if random.random() < self.spawn_rate:
                # Place only if tile is road and doesn't have a car already
                cell_agents = self.grid.get_cell_list_contents(p)
                has_car = any(isinstance(a, carAgent) for a in cell_agents)
                is_road = any(isinstance(a, roadCell) for a in cell_agents)
                if (not has_car) and is_road:
                    car = carAgent(self.next_id(), self, p, d)
                    self.grid.place_agent(car, p)
                    self.scheduler.add(car)
                    self.total_spawned += 1
                # Stop early if we got too many
                if self._count_cars() >= self.max_cars:
                    break    

    def step(self):
        # Reset normal road reservations
        self.reserved_positions.clear()
        # Reset intersection reservations will be cleared when cars leave
        self.light_scheduler.step()
        self.spawn_cars()
        self.scheduler.step()
        self.datacollector.collect(self)