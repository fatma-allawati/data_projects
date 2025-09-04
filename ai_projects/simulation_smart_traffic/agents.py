#Author: Fatma Al Lawati
from __future__ import annotations
from typing import Optional, Tuple, List
import random
from mesa import Agent


Coords: Tuple[int, int]

#Road cell (drawing all roads)
class roadCell(Agent):
    """Passive tile agent so canvas grid can draw road background.
    Oriantation: "NS" vertical only, "EW" horizontal only, "X" intersections.
    """

    def __init__(self, unique_id, model, pos: Coords, oriantation: str):
        super().__init__(unique_id, model)
        self.pos = pos
        self.oriantation = oriantation #NS, EW, or X

    def step(self):
        pass

#Traffic light at intersections
class trafficLight(Agent):
    """Two phase traffic light that alternates:
        If phase = 0 then NS green and EW red
        If phase = 1 then EW red and NS green
    Duration controlled by model params.
    """

    def __init__(self, unique_id, model, pos: Coords, phase_offset: int = 0):
        super().__init__(unique_id, model)
        self.pos = pos
        self.phase = 0 #Strat NS green
        self.t = phase_offset #Step counter with offset to stagger light

    def step(self):
        self.t += 1
        ns_green = self.model.ns_green
        ew_green = self.model.ew_green

        #Total length of a full cycle
        period = ns_green + ew_green 

        #Current phase based on timer within period
        phase_time = self.t % period
        self.phase = 0 if phase_time < ns_green else 1 
    
    def is_green_for(self, moving_dir: str) -> bool:
        """moving_dir for {N, S, E, W}"""

        if self.phase == 0:
            #NS green
            return moving_dir in("N","S")
        else:
            #EW green
            return moving_dir in("E","W")
        
class carAgent(Agent):
    """Car moves cell by cell along roads:
        - Respects red lights at intersections
        - Avoids collisions
        - Turns only when safe
        - Exits the grid at boundaries
    """

    TURN_PROBS = (0.2, 0.6, 0.2)

    def __init__(self, unique_id, model, pos: Coords, direction: str):
        super().__init__(unique_id, model)
        self.pos = pos
        self.dir = direction
        self.wait = 0
        self.next_pos: Optional[Coords] = None
        self.exiting = False
        self.to_remove = False
        self.in_intersection = False  # Track if car is inside an intersection

    def forward(self, pos: Coords, direction: str) -> Coords:
        x, y = pos
        return {"N": (x, y - 1), "S": (x, y + 1), "E": (x + 1, y), "W": (x - 1, y)}[direction]

    def left_of(self, direction: str) -> str:
        return {"N":"W","W":"S","S":"E","E":"N"}[direction]

    def right_of(self, direction: str) -> str:
        return {"N":"E","E":"S","S":"W","W":"N"}[direction]

    def on_grid(self, p: Coords) -> bool:
        x, y = p
        return 0 <= x < self.model.grid.width and 0 <= y < self.model.grid.height

    def is_road(self, p: Coords) -> bool:
        return any(isinstance(a, roadCell) for a in self.model.grid.get_cell_list_contents(p))

    def has_car(self, p: Coords) -> bool:
        return any(isinstance(a, carAgent) for a in self.model.grid.get_cell_list_contents(p))

    def traffic_light_in(self, p: Coords) -> Optional[trafficLight]:
        for a in self.model.grid.get_cell_list_contents(p):
            if isinstance(a, trafficLight):
                return a
        return None

    def at_intersection(self, p: Coords) -> bool:
        return self.traffic_light_in(p) is not None

    # ---------------- Planning Phase ----------------
    def step(self):
        self.next_pos = None
        self.exiting = False

        target = self.forward(self.pos, self.dir)

        # Exit grid
        if not self.on_grid(target):
            self.exiting = True
            self.to_remove = True
            return

        # Normal road check
        if not self.is_road(target):
            self.wait += 1
            return

        # Intersection logic
        if self.at_intersection(target):
            light = self.traffic_light_in(target)
            if light is None or not light.is_green_for(self.dir):
                self.wait += 1
                return

            # Only enter if intersection free
            if target in self.model.intersection_reserved:
                self.wait += 1
                return

            # Reserve intersection and move in
            self.model.intersection_reserved[target] = self
            self.next_pos = target
            self.in_intersection = True

            # Decide new direction for next step
            left_p, straight_p, right_p = self.TURN_PROBS
            choice = random.random()
            if choice < left_p:
                self.dir = self.left_of(self.dir)
            elif choice < left_p + straight_p:
                self.dir = self.dir
            else:
                self.dir = self.right_of(self.dir)
            return

        # Normal road
        if self.has_car(target) or target in self.model.reserved_positions:
            self.wait += 1
            return

        self.model.reserved_positions.add(target)
        self.next_pos = target

    # ---------------- Apply Movement Phase ----------------
    def advance(self):
        if self.to_remove:
            self.model.grid.remove_agent(self)
            self.model.scheduler.remove(self)
            self.model.cars_exited += 1
            return

        if self.next_pos is not None:
            self.model.grid.move_agent(self, self.next_pos)

        # Free intersection reservation if leaving
        if self.in_intersection:
            if self.pos in self.model.intersection_reserved and self.model.intersection_reserved[self.pos] is self:
                del self.model.intersection_reserved[self.pos]
            self.in_intersection = False