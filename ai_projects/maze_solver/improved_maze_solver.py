#Author: Fatma Al Lawati

"""
Genetic Algorithm Maze Solver (Feasible Path Version)
This GA evolves valid paths through a grid maze, ensuring individuals do not move into walls.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable

# Maze Definition
coords = Tuple[int, int]

@dataclass
class Maze:
    grid: List[List[int]]  # 0 = free, 1 = wall
    start: coords
    goal: coords

    @property
    def height(self) -> int:
        return len(self.grid)
    
    @property
    def width(self) -> int:
        return len(self.grid[0]) if self.grid else 0
    
    def in_bounds(self, pos: coords) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width
    
    def is_free(self, pos: coords) -> bool:
        r, c = pos
        return self.in_bounds(pos) and self.grid[r][c] == 0
    
    def neighbors(self, pos: coords) -> List[coords]:
        r, c = pos
        candidates = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
        return [p for p in candidates if self.is_free(p)]
    
    def manhattan(self, a: coords, b: coords) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Individual
@dataclass
class Individual:
    path: List[coords]
    fitness: float = float('-inf')
    reached_goal: bool = False

# GA Parameters
@dataclass
class GAConfig:
    pop_size: int = 50
    mutation_rate: float = 0.2
    crossover_rate: float = 0.9
    elitism: int = 5
    tournament_k: int = 3
    max_generations: int = 50
    stagnation_patience: int = 50
    goal_bonus: float = 1000.0

# GA Logic
class MazeGA:
    def __init__(self, maze: Maze, cfg: GAConfig):
        self.maze = maze
        self.cfg = cfg
        self.rng = random.Random()
        self.population: List[Individual] = []
        self.best: Optional[Individual] = None
    
    # --- Population Initialization ---
    def random_path(self) -> List[coords]:
        """Generate a random feasible path from start to goal using random walk"""
        path = [self.maze.start]
        visited = {self.maze.start}
        while path[-1] != self.maze.goal and len(path) < self.maze.height*self.maze.width*2:
            neighbors = [n for n in self.maze.neighbors(path[-1]) if n not in visited]
            if not neighbors:
                # Dead-end: backtrack
                path.pop()
                if not path:
                    path = [self.maze.start]
                    visited = {self.maze.start}
                continue
            nxt = self.rng.choice(neighbors)
            path.append(nxt)
            visited.add(nxt)
        return path
    
    def init_population(self):
        self.population = [Individual(self.random_path()) for _ in range(self.cfg.pop_size)]
    
    # --- Fitness Evaluation ---
    def evaluate(self, ind: Individual):
        """Fitness = -path length + goal bonus if reached"""
        dist_to_goal = self.maze.manhattan(ind.path[-1], self.maze.goal)
        reached = ind.path[-1] == self.maze.goal
        ind.reached_goal = reached
        ind.fitness = -len(ind.path) + (self.cfg.goal_bonus if reached else 0) - dist_to_goal
    
    def evaluate_population(self):
        for ind in self.population:
            self.evaluate(ind)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        if self.best is None or self.population[0].fitness > self.best.fitness:
            self.best = Individual(self.population[0].path[:], self.population[0].fitness, self.population[0].reached_goal)
    
    # --- Selection ---
    def tournament_select(self) -> Individual:
        candidates = self.rng.sample(self.population, self.cfg.tournament_k)
        return max(candidates, key=lambda x: x.fitness)
    
    # --- Crossover ---
    def crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        if self.rng.random() > self.cfg.crossover_rate:
            return Individual(p1.path[:]), Individual(p2.path[:])
        
        # Find common nodes
        set1 = set(p1.path)
        set2 = set(p2.path)
        common = list(set1 & set2)
        if not common:
            return Individual(p1.path[:]), Individual(p2.path[:])
        cx_point = self.rng.choice(common)
        i1 = p1.path.index(cx_point)
        i2 = p2.path.index(cx_point)
        child1_path = p1.path[:i1] + p2.path[i2:]
        child2_path = p2.path[:i2] + p1.path[i1:]
        return Individual(child1_path), Individual(child2_path)
    
    # --- Mutation ---
    def mutate(self, ind: Individual):
        if self.rng.random() > self.cfg.mutation_rate:
            return
        if len(ind.path) < 3:
            return
        idx = self.rng.randrange(1, len(ind.path)-1)
        # Reroute a small subpath
        new_path = ind.path[:idx]
        visited = set(new_path)
        while new_path[-1] != self.maze.goal and len(new_path) < self.maze.height*self.maze.width:
            neighbors = [n for n in self.maze.neighbors(new_path[-1]) if n not in visited]
            if not neighbors:
                break
            nxt = self.rng.choice(neighbors)
            new_path.append(nxt)
            visited.add(nxt)
        ind.path = new_path
    
    # --- Next Generation ---
    def next_generation(self):
        new_pop: List[Individual] = []
        # Elitism
        elites = [Individual(ind.path[:], ind.fitness, ind.reached_goal) for ind in self.population[:self.cfg.elitism]]
        new_pop.extend(elites)
        # Fill rest
        while len(new_pop) < self.cfg.pop_size:
            p1 = self.tournament_select()
            p2 = self.tournament_select()
            c1, c2 = self.crossover(p1, p2)
            self.mutate(c1)
            self.mutate(c2)
            new_pop.append(c1)
            if len(new_pop) < self.cfg.pop_size:
                new_pop.append(c2)
        self.population = new_pop

# --- Utility ---
def render_maze_with_path(maze: Maze, path: Iterable[coords]) -> str:
    path_set = set(path)
    chars = []
    for r in range(maze.height):
        row_chars = []
        for c in range(maze.width):
            pos = (r,c)
            if pos == maze.start:
                row_chars.append('S')
            elif pos == maze.goal:
                row_chars.append('G')
            elif maze.grid[r][c] == 1:
                row_chars.append('#')
            elif pos in path_set:
                row_chars.append('*')
            else:
                row_chars.append('.')
        chars.append(''.join(row_chars))
    return '\n'.join(chars)

# --- Demo Mazes ---
def medium_maze() -> Maze:
    grid = [
        [0,1,0,0,0,0,0,0,0,0],
        [0,1,0,1,1,1,1,1,1,0],
        [0,1,0,1,0,0,0,0,1,0],
        [0,1,0,1,0,1,1,0,1,0],
        [0,0,0,1,0,1,0,0,1,0],
        [1,1,0,1,0,1,0,1,1,0],
        [0,0,0,0,0,1,0,0,0,0],
        [0,1,1,1,1,1,0,1,1,1],
        [0,0,0,0,0,0,0,1,0,0],
        [1,1,1,1,1,1,0,1,0,0],
    ]
    return Maze(grid=grid, start=(0,0), goal=(9,9))

# --- Run GA ---
def run_ga(maze: Maze, cfg: GAConfig, seed: Optional[int]=None):
    ga = MazeGA(maze, cfg)
    if seed is not None:
        ga.rng.seed(seed)
        random.seed(seed)
    ga.init_population()
    best = None
    best_gen = 0
    no_improve = 0

    for gen in range(1, cfg.max_generations+1):
        ga.evaluate_population()
        current_best = ga.population[0]
        if best is None or current_best.fitness > best.fitness:
            best = Individual(current_best.path[:], current_best.fitness, current_best.reached_goal)
            best_gen = gen
            no_improve = 0
        else:
            no_improve += 1
        
        if current_best.reached_goal:
            print(f"Reached goal in generation {gen} with fitness {current_best.fitness}")
            best = Individual(current_best.path[:], current_best.fitness, current_best.reached_goal)
            best_gen = gen
            break
        
        if no_improve >= cfg.stagnation_patience:
            print(f"No improvement for {cfg.stagnation_patience} generations. Stopping at gen {gen}.")
            break
        
        if gen % 10 == 0:
            print(f"Gen {gen:4d} | best fitness: {current_best.fitness:.2f} | reached: {current_best.reached_goal}")
        
        ga.next_generation()
    
    return best, best_gen

def print_solution(maze: Maze, best: Individual):
    art = render_maze_with_path(maze, best.path)
    print("\nBest Individual Summary:")
    print(f" Fitness : {best.fitness:.2f}")
    print(f" Reached goal: {best.reached_goal}")
    print(f" Steps used : {len(best.path)}")
    print("\nPath visualization (* marks visited cells):")
    print(art)

# --- Main ---
if __name__ == "__main__":
    maze = medium_maze()
    cfg = GAConfig(
        pop_size=200,
        mutation_rate=0.3,
        crossover_rate=0.9,
        elitism=5,
        tournament_k=3,
        max_generations=300,
        stagnation_patience=50,
        goal_bonus=1000.0
    )
    seed = 42
    best, best_gen = run_ga(maze, cfg, seed)
    print_solution(maze, best)
    print(f"\nBest found at generation: {best_gen}")
