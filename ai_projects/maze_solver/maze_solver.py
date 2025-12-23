#Author: Fatma Al Lawati

"""
Genetic Algorithm Maze Solver
Python implementetion of GA that evolves move sequence to solve a grid maze. No third party libraries required.

How it works:
    1. Maze: 2D grid of 0 (free) and 1 (wall). Start 'S' and goal 'G' positions.
    2. DNA: Fixed length list of moves from {U, D, L, R}.
    3. Fitness: Closer to goal is better (distance). Bonuses for reaching goal quickly, penalties for hitting walls/revising cells. 
    4. Selection: Tournament selection.
    5. Crossover: Single point.
    6. Mutation: Per gene with rate mutatuin rate.
    7. Elitism: Top K individuals copied to next generations.
"""

from __future__ import annotations
import random
import math 
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable

#Maze Definition 
move = str #U, D, L, R
coords = Tuple[int, int]

@dataclass
class Maze:
    grid: List[List[int]] #0 free, 1 wall
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
    
    def muscat(self, a: coords, b: coords) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def step(self, pos: coords, move = move) -> coords:
        drc = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        dr, dc = drc[move]
        nxt = (pos[0] + dr, pos[1] + dc)

        #Blocked by wall or out of bounds -> stay in place (counts as bump)
        if not self.is_free(nxt):
            return pos
        return nxt

#GA Components
MOVES: Tuple[move,...] = ('U','D','L','R')

@dataclass
class gene_length:
    pop_size: int = 500
    gene_length: int = 200
    mutation_rate: float = 0.05
    crossover_rate: float = 0.9
    elitism: int = 5
    tournament_k: int = 4
    max_generations: int = 1000
    stagnation_patience: int = 200 #Early stop if no improvement
    wall_penalty: float = 0.2 #Per bump into wall/out of bounds
    revisit_penalty: float = 0.1 #Per revisit of a cell
    goal_bonus: float = 100.0

chromosome = List[move]

@dataclass
class individual:
    dna: chromosome
    fitness: float = float('-inf')
    reached_goal: bool = False
    final_pos: coords = (0,0)
    steps_used: int = 0

#GA Logic
class mazeGA:
    def __init__(self, maze: Maze, cfg: gene_length):
        self.maze = maze
        self.cfg = cfg
        self.rng = random.Random()
        self.population: List[individual] = []
        self.best: Optional[individual] = None
        self.best_history: List[float] = []

    #Initialization
    def random_dna(self) -> chromosome:
        return[self.rng.choice(MOVES) for _ in range(self.cfg.gene_length)]
    
    def init_population(self) -> None:
        self.population = [individual(self.random_dna()) for _ in range(self.cfg.pop_size)]

    #Fitness evaluation
    def evaluate(self, ind: individual) -> None:
        pos = self.maze.start
        visited = {pos: 1}
        bumps = 0
        steps_used = 0
        reached = 0
        score = 0
        prev_dist = self.maze.muscat(pos, self.maze.goal)

        for step, mv in enumerate(ind.dna, 1):
            nxt = self.maze.step(pos, mv)
            new_dist = self.maze.muscat(nxt, self.maze.goal)

            #Reward getting closer to goal
            score += max(0, prev_dist - new_dist) * 20.0  # reward progress
            prev_dist = new_dist

            if nxt == pos: #Bumb
                bumps += 1
            else:
                pos = nxt
                steps_used = step
                visited[pos] = visited.get(pos, 0) + 1

                if pos == self.maze.goal:
                    reached = True
                    break

        #Base score: inverse of distance
        #dist = self.maze.muscat(pos, self.maze.goal)
        #base = -dist

        # Penalties
        score -= self.cfg.wall_penalty * bumps * 5.0
        score -= self.cfg.revisit_penalty * sum(count - 1 for count in visited.values()) * 2.0

        #Bounce for reaching goal, scaled by remaining steps (faster is better)
        if reached:
            score += 10000 + (self.cfg.gene_length - steps_used) * 50.0

        score += 1000 / (1 + self.maze.muscat(pos, self.maze.goal))

        ind.fitness = score
        ind.reached_goal = reached
        ind.final_pos = pos
        ind.steps_used = steps_used if reached else len(ind.dna)


    def evaluate_population(self) -> None:
        for ind in self.population:
            self.evaluate(ind)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        if self.best is None or self.population[0].fitness > self.best.fitness:
            self.best = self.clone(self.population[0])
        self.best_history.append(self.best.fitness)
    
    #Selection
    def tournament_select(self) -> individual:
        K = self.cfg.tournament_k
        candidates = self.rng.sample(self.population, K)
        return max(candidates, key=lambda x: x.fitness)
    
    #Crossover and Mutation
    def crossover(self, a: chromosome, b: chromosome) -> Tuple[chromosome, chromosome]:
        if self.rng.random() > self.cfg.crossover_rate:
            return a[:], b[:]
        point = self.rng.randrange(1, self.cfg.gene_length)
        return a[:point] + b[point:], b[:point] + a[point:]
    
    def mutate(self, dna: chromosome) -> None:
        for i in range(len(dna)):
            if self.rng.random() < self.cfg.mutation_rate:
                dna[i] = self.rng.choice(MOVES)
    
    #Reproduction
    def next_generation(self) -> None:
        new_pop: list[individual] = []

        #Elitism
        elites = [self.clone(ind) for ind in self.population[: self.cfg.elitism]]
        new_pop.extend(elites)

        #Fill the rest with children
        while len(new_pop) < self.cfg.pop_size:
            p1 = self.tournament_select()
            p2 = self.tournament_select()
            c1_dna, c2_dna = self.crossover(p1.dna, p2.dna)
            self.mutate(c1_dna)
            self.mutate(c2_dna)
            new_pop.append(individual(c1_dna))
            if len(new_pop) < self.cfg.pop_size:
                new_pop.append(individual(c2_dna))
        
        self.population = new_pop

    #Utilities
    def clone(self, ind: individual) -> individual:
        return individual(ind.dna[:], ind.fitness, ind.reached_goal, ind.final_pos, ind.steps_used)
    
    def decode_path(self, ind: individual) -> list[coords]:
        pos = self.maze.start
        path = [pos]
        for mv in ind.dna:
            nxt = self.maze.step(pos, mv)
            if nxt == pos:
                #bumb, still record for same pos for clarity
                path.append(pos)
            
            else:
                pos = nxt
                path.append(pos)
            
            if pos == self.maze.goal:
                break
        return path

#Pretty printing
def render_maze_with_path(maze: Maze, path: Iterable[coords]) -> str:
    path_set = set(path)
    chars = []
    for r in range(maze.height):
        row_chars = []
        for c in range(maze.width):
            pos = (r, c)

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

#Demo Maze
def tiny_maze() -> Maze:
    grid = [
        [0,0,0,0,0],
        [1,1,1,1,0],
        [0,0,0,1,0],
        [0,1,0,0,0],
        [0,1,0,1,0],
    ]
    return Maze(grid=grid, start=(0,0), goal=(4,4))

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

#Main
def run_ga(maze: Maze, cfg: gene_length, seed: Optional[int] = None) -> Tuple[individual, int]:
    ga = mazeGA(maze, cfg)
    if seed is not None:
        ga.rng.seed(seed)
        random.seed(seed)
    ga.init_population()

    best = None
    best_gen = 0
    no_improve = 0

    for gen in range(1, cfg.max_generations + 1):
        ga.evaluate_population()
        current_best = ga.population[0]

        if best is None or current_best.fitness > best.fitness + 1e-9:
            best = ga.clone(current_best)
            best_gen = gen
            no_improve = 0
        
        else:
            no_improve += 1
        
        #Early stop if goal reached ir stagnation
        if current_best.reached_goal:
            print(f"Reached goal in generation {gen} with fitness {current_best.fitness:.2f}.")
            best = ga.clone(current_best)
            best_gen = gen
            break

        if no_improve >= cfg.stagnation_patience:
            print(f"No improvement for {cfg.stagnation_patience} generations. Stopping at gen {gen}.")
            break

        if gen % 10 == 0:
            print(f"Gen {gen:4d} | best fitness: {current_best.fitness:.2f} | reached: {current_best.reached_goal}")

        ga.next_generation()
    
    #Final evaluation to make sure best has all fields consistent
    if best is None:
        ga.evaluate_population()
        best = ga.clone(ga.population[0])
        best_gen = 0

    return best, best_gen

def print_solution(maze: Maze, best: individual) -> None:
    path = mazeGA(maze, gene_length()).decode_path(best)
    art = render_maze_with_path(maze, path)
    print("\nBest Individual Summary:")
    print(f" Fitness : {best.fitness:.2f}")
    print(f" Reached goal: {best.reached_goal}")
    print(f" Steps used : {best.steps_used}")
    print(f" Final pos : {best.final_pos}")
    print("\nPath visualization (* marks visited cells):")
    print(art)

if __name__ == "__main__":
    #Choose a maze
    maze = medium_maze() #Or tiny maze()

    #Configure GA
    cfg = gene_length(
        pop_size = 250,
        gene_length = 400,
        mutation_rate = 0.1,
        crossover_rate = 0.9,
        elitism = 8,
        tournament_k = 5,
        max_generations = 600,
        stagnation_patience = 60,
        wall_penalty = 0.2,
        goal_bonus = 60.0,
    )

    #Set a seed for reproducibility
    seed = 42

    best, best_gen = run_ga(maze, cfg, seed=seed)
    print_solution(maze,best)
    print(f"\nBest found at generation: {best_gen}")