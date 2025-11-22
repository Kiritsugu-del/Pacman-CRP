from typing import Tuple, Set, Dict, List
from fol_components import Constant, Variable, Predicate, LocationBasedPredicate, Moved, NextLikely, unify
import random
import os
import sys
import time
import random

Coord = Tuple[int, int]


def get_pressed_key() -> str:
    """Check if an arrow key or 'q' are pressed.
        Returns 'UP', 'DOWN', 'LEFT', 'RIGHT', 'QUIT', or None."""
    # Check OS type first
    if os.name == 'nt':
        # Windows solution
        try:
            import msvcrt
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                # Arrow key prefix
                if ch in [b'\x00', b'\xe0']:
                    ch2 = msvcrt.getch()
                    # Map to direction
                    if ch2 == b'H':
                        return 'UP'
                    elif ch2 == b'P':
                        return 'DOWN'
                    elif ch2 == b'M':
                        return 'RIGHT'
                    elif ch2 == b'K':
                        return 'LEFT'
                elif ch.decode('utf-8', errors='ignore').lower() == 'q':
                    return 'QUIT'
            return None
        except ImportError:
            return None
    else:
        # Unix/Linux/macOS solution
        try:
            import tty
            import termios
            import select
            tty.setcbreak(sys.stdin.fileno(), termios.TCSANOW)
        
            # Check if input is available (non-blocking)
            if not select.select([sys.stdin], [], [], 0)[0]:
                return None

            ch = sys.stdin.read(1)
            # Handle arrow keys (escape sequences)
            if ch == '\x1b':
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    # Map to direction
                    if ch3 == 'A':
                        return 'UP'
                    elif ch3 == 'B':
                        return 'DOWN'
                    elif ch3 == 'C':
                        return 'RIGHT'
                    elif ch3 == 'D':
                        return 'LEFT'
            elif ch.lower() == 'q':
                return 'QUIT'
            return None
        except (ImportError, AttributeError):
            return None
class FOL_KB:
    def __init__(self,env):
        self.facts: Set[Predicate] = set()
        self.env = env

    def add_fact(self, fact: Predicate) -> None:
        assert fact.is_ground()
        self.facts.add(fact)

    def infer_next_likely(self,env) -> Set[NextLikely]:
        """Apply: Moved(Pacman,X,Y,dx,dy) -> NextLikely(X+dx, Y+dy)."""
        results: Set[NextLikely] = set()
        X = Variable("X")
        Y = Variable("Y")
        DX = Variable("DX")
        DY = Variable("DY")

        # rule body pattern
        pattern = Moved(Constant("Pacman"), X, Y, DX, DY)

        for f in list(self.facts):
            c = unify(pattern, f)
            if c is None:
                continue
            x = c[X].value
            y = c[Y].value
            dx = c[DX].value
            dy = c[DY].value
            nx, ny = x + dx, y + dy
            nx, ny = x + dx, y + dy
            if env.blocked((nx, ny)):
                continue
            
            results.add(NextLikely(Constant(nx), Constant(ny)))
        return results
    
class KB:
    def __init__(self):
        self.facts: Set[str] = set()
        self.rules: List[Tuple[str, Set[str]]] = []

    def add_fact(self, atom: str) -> None:
        self.facts.add(atom)

    def add_rule(self, head: str, body: List[str]) -> None:
        """Add a rule: head :- body (if all body atoms are true, infer head)."""
        self.rules.append((head, set(body)))

    def infer_all(self) -> None:
        """Forward chaining: fire all applicable rules until fixpoint."""
        changed = True
        while changed:
            changed = False
            for head, body in self.rules:
                if body.issubset(self.facts) and head not in self.facts:
                    self.facts.add(head)
                    changed = True

    def entails(self, atom: str) -> bool:
        """Does the KB entail this atom?"""
        self.infer_all()
        return atom in self.facts

    def clear(self) -> None:
        self.facts.clear()


class Environment:
    """Grid representing the game environment."""
    def __init__(
        self,
        w: int,
        h: int,
        walls: Set[Coord] = None,
        pellets: Set[Coord] = None,
        start_pos: Coord = (0, 0),
        lives: int = 3
        

    ):
        
        self.w, self.h = w, h
        self.walls: Set[Coord] = set(walls or set())
        self.pellets: Set[Coord] = set(pellets or set())
        self.pacman_pos: Coord = start_pos
        self.start_pos: Coord = start_pos  
        self.time: int = 0
        self.finished: bool = False
        self.ghosts: List['Ghost1','Ghost2'] = []  
        self.score = 0
        self.lives=lives
        self.fol_kb = None


    def in_bounds(self, c: Coord) -> bool:
        """Return True if coordinate c is within grid bounds."""
        x, y = c
        return 0 <= x < self.w and 0 <= y < self.h

    def blocked(self, c: Coord) -> bool:
        """Return True if coordinate c is blocked by walls or bounds."""
        return (not self.in_bounds(c)) or (c in self.walls)

    def sense(self) -> Dict:
        """Return a percept dictionary describing the current state."""
        return dict(
            pos=self.pacman_pos,
            pellet_here=(self.pacman_pos in self.pellets),
            time=self.time,
            finished=self.finished
        )
    
    def step(self, action: str):
        """Advance the environment one step given an action string.
            Supported actions: 'UP', 'DOWN', 'LEFT', 'RIGHT' to move."""
        if self.finished:
            return

        self.time += 1
        self.pacman_prev_pos=self.pacman_pos 

        #hasattr returns True if obj.attr_name exists, otherwise False (without raising an error).
        if hasattr(self, "fol_kb") and self.fol_kb is not None:
            ox, oy = self.pacman_prev_pos
            nx, ny = self.pacman_pos
            dx, dy = nx - ox, ny - oy
            from fol_components import Constant, Moved
            self.fol_kb.add_fact(
                Moved(Constant("Pacman"),
                    Constant(ox), Constant(oy),
                    Constant(dx), Constant(dy))
            )

        # Move according to the action
        moves = {'RIGHT': (1, 0), 'LEFT': (-1, 0), 'DOWN': (0, 1), 'UP': (0, -1)}
        if action in moves:
            dx, dy = moves[action]
            nx, ny = self.pacman_pos[0] + dx, self.pacman_pos[1] + dy
            if not self.blocked((nx, ny)):
                self.pacman_pos = (nx, ny)
            ox, oy = self.pacman_prev_pos

        if (ox, oy) != (nx, ny) and self.fol_kb is not None:
            dx, dy = nx - ox, ny - oy
            self.fol_kb.add_fact(
                Moved(Constant("Pacman"),
                    Constant(ox), Constant(oy),
                    Constant(dx), Constant(dy))
    )
        
        if self.pacman_pos in self.pellets:
            self.pellets.remove(self.pacman_pos)
            self.score += 10

        # Collect pellet if needed
        if self.pacman_pos in self.pellets:
            self.pellets.remove(self.pacman_pos)
        
        for ghost in self.ghosts:
            ghost.perceive_and_update_kb(self)
            ghost.next_action(self)

        if any(ghost.pos == self.pacman_pos for ghost in self.ghosts):
            if self.lives <= 0:
                self.finished = True
            else:
                self.pacman_pos = self.start_pos
                self.lives -= 1
                for ghost in self.ghosts:
                    ghost.pos = ghost.start


        # Check if no pellets are left
        if len(self.pellets) == 0:
            self.finished = True

    def render(self) -> str:
        """Return a multi-line string visualization of the grid.

        Legend:
            'P' - Pac-Man
            '#' - Wall
            '.' - Pellet
            ' ' - Empty space
            'G' - Ghost
        """
        buf: List[str] = []
        status_line = f"t={self.time} | pellets={len(self.pellets)} | score={self.score} | lives = {self.lives}"
        buf.append(status_line)

        ghost_positions = {g.pos: i for i, g in enumerate(self.ghosts)}
        for y in range(self.h):
            row = []
            for x in range(self.w):
                c = (x, y)
                if c in ghost_positions:
                    #ch = 'G'
                    ch = str(ghost_positions[c] + 1)  
                elif c == self.pacman_pos and c not in ghost_positions:
                    ch = 'P'
                elif c in self.walls:
                    ch = '#'
                elif c in ghost_positions and c != self.pacman_pos:
                    ch = 'G'
                elif c in self.pellets:
                    ch = '.'
                else:
                    ch = ' '

                row.append(ch)
            buf.append(''.join(row))

        if self.finished:
            buf.append("GAME FINISHED!")

        return '\n'.join(buf)


class Ghost1:

    Coord = Tuple[int,int]

    def __init__(self,start: Coord, sight: int = 4):
        self.pos = start
        self.sight = sight
        self.kb = KB()
        self.start=start
        self.current_target = None   # <---
        self.adjacent_counter = 0   # nº de turnos consecutivos adjacente ao Pac-Man
        self.rest_turns = 0       # nº de turnos restantes que o fantasma está a descansar
    
    def _can_see_pacman(self, env: Environment) -> Tuple[bool, int, int]:
        #print("G:", self.pos, "P:", env.pacman_pos)
        gx, gy = self.pos
        px, py = env.pacman_pos

        # visão vertical
        if px == gx and abs(py - gy) <= self.sight:
            step = 1 if py > gy else -1
            y = gy + step
            while True:
                if env.blocked((gx, y)):   # parede bloqueia visão
                    return False, px, py
                if y == py:
                    return True, px, py
                y += step

        # visão horizontal
        if py == gy and abs(px - gx) <= self.sight:
            step = 1 if px > gx else -1
            x = gx + step
            while True:
                if env.blocked((x, gy)):
                    return False, px, py
                if x == px:
                    return True, px, py
                x += step

        return False, px, py

    
    def _knows_wall(self, x: int, y: int) -> bool:
        """Verifica se a KB sabe que há uma parede em (x, y)."""
        return f"Wall_{x}_{y}" in self.kb.facts
    
    def perceive_and_update_kb(self, env: Environment) -> None:
        gx, gy = self.pos

        # 1) PERCEIVE
        seen, px, py = self._can_see_pacman(env)
        new_target = (px, py) if seen else None

        arrived = (self.current_target is not None and
                (gx, gy) == self.current_target)

        # 2) DEFINIR NOVO ESTADO INTERNO
        if seen:
            # Caso 1: Viu Pac-Man ➝ atualizar target
            self.current_target = new_target
        elif arrived and not seen:
            # Caso 2: NÃO viu e chegou ao target ➝ esquecer target
            self.current_target = None

        # 3) ATUALIZAR KB — limpar transitórios e adicionar ESTADO INTERNO atual
        self.kb.facts = {
            f for f in self.kb.facts
            if not (f.startswith("Seen_") or f.startswith("Ghost_") or f.startswith("Target_"))
        }

        # se há target novo → adiciona factos e regra
        if self.current_target is not None:
            tx, ty = self.current_target
            seen_atom = f"Seen_{tx}_{ty}"
            target_atom = f"Target_{tx}_{ty}"
            self.kb.add_fact(seen_atom)
            self.kb.add_fact(target_atom)
            self.kb.add_rule(target_atom, [seen_atom])

        # adicionar posição atual do fantasma
        self.kb.add_fact(f"Ghost_{gx}_{gy}")

        # adicionar paredes conhecidas à volta
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = gx + dx, gy + dy
            if env.in_bounds((nx, ny)) and env.blocked((nx, ny)):
                self.kb.add_fact(f"Wall_{nx}_{ny}")

        # fazer inferência
        self.kb.infer_all()


    def next_action(self, env: Environment) -> str:
        gx, gy = self.pos

        # 0.1 LÓGICA DE DESCANSO
        if self.rest_turns > 0:
            self.rest_turns -= 1
            return 'WAIT'
        # 0.2 ATUALIZAÇÃO DO CONTADOR
        px, py = env.pacman_prev_pos
        is_adj = (abs(gx - px) + abs(gy - py) == 1)
        if is_adj:
            self.adjacent_counter += 1
        else:
            self.adjacent_counter = 0
        # 0.3 CONGELAMENTO
        if self.adjacent_counter >= 10:
            self.rest_turns = 2
            self.adjacent_counter = 0
            return 'WAIT'

        # 1) se KB conhece alvos (Target_x_y), perseguir o mais próximo
        targets = []
        for fact in list(self.kb.facts):
            if fact.startswith("Target_"):
                _, sx, sy = fact.split("_")
                targets.append((int(sx), int(sy)))

        if targets:
            # escolher target mais próximo
            tx, ty = min(targets, key=lambda c: abs(c[0] - gx) + abs(c[1] - gy))
            dx = 1 if tx > gx else (-1 if tx < gx else 0)
            dy = 1 if ty > gy else (-1 if ty < gy else 0)

            # tentar mover-se para o target usando apenas conhecimento da KB sobre paredes
            if dx != 0:
                nx, ny = gx + dx, gy
                if env.in_bounds((nx, ny)) and not self._knows_wall(nx, ny):
                    self.pos = (nx, ny)
                    return 'RIGHT' if dx == 1 else 'LEFT'

            if dy != 0:
                nx, ny = gx, gy + dy
                if env.in_bounds((nx, ny)) and not self._knows_wall(nx, ny):
                    self.pos = (nx, ny)
                    return 'DOWN' if dy == 1 else 'UP'

        # 3) fallback: movimento aleatório baseado apenas em paredes conhecidas
        dirs = [
            ('UP', (0, -1)),
            ('DOWN', (0, 1)),
            ('LEFT', (-1, 0)),
            ('RIGHT', (1, 0))
        ]
        random.shuffle(dirs)
        for name, (dx, dy) in dirs:
            nx, ny = gx + dx, gy + dy
            if env.in_bounds((nx, ny)) and not self._knows_wall(nx, ny):
                self.pos = (nx, ny)
                return name

        return 'WAIT'

    
class Ghost2:

    Coord = Tuple[int,int]

    def __init__(self,start: Coord, sight: int = 4):
        self.pos = start
        self.sight = sight
        self.kb = KB()
        self.start = start

     
    def perceive_and_update_kb(self, env) -> None:
        # drop any transient sensors if you want; here we remember seen pacman cells
            px, py = env.pacman_pos
            gx, gy = self.pos


            if abs(px - gx) + abs(py - gy) <= self.sight:
                atom = f"Seen_{px}_{py}"
                self.kb.add_fact(atom)
                # keep a rule that maps Seen -> Target (simple memory)
                self.kb.add_rule(f"Target_{px}_{py}", [atom])

            # always record our current position atom (optional)
            self.kb.add_fact(f"Ghost_{gx}_{gy}")

            # record local walls so the ghost avoids them
            moves = {'UP': (0,-1),'DOWN':(0,1),'LEFT':(-1,0),'RIGHT':(1,0)}
            for d,(dx,dy) in moves.items():
                nx, ny = gx+dx, gy+dy
                if env.blocked((nx, ny)):
                    self.kb.add_fact(f"Wall_{nx}_{ny}")

            self.kb.infer_all()

    def next_action(self, env) -> str:
            gx, gy = self.pos
            # 1) if pacman adjacent -> chase (deterministic)
            for dir_name, (dx,dy) in {'LEFT':(-1,0),'RIGHT':(1,0)}.items():
                nx, ny = gx+dx, gy+dy
                if (nx, ny) == env.pacman_pos:
                    # move into pacman
                    self.pos = (nx, ny)
                    return dir_name

            # 2) if KB knows a Target, move greedily towards closest target
            # build list of remembered targets (atoms like Target_x_y)
            targets = []
            for fact in list(self.kb.facts):
                if fact.startswith("Target_"):
                    _, sx, sy = fact.split("_")
                    targets.append((int(sx), int(sy)))

            if targets:
                tx, ty = min(targets, key=lambda c: abs(c[0] - gx) + abs(c[1] - gy))
                if gy != ty:
                    if ty > gy and not env.blocked((gx, gy + 1)):  # Move down towards row
                        self.pos = (gx, gy + 1)
                        return 'DOWN'
                    elif ty < gy and not env.blocked((gx, gy - 1)):  # Move up towards row
                        self.pos = (gx, gy - 1)
                        return 'UP'
                    # If vertical blocked, try horizontal (to navigate around obstacles)
                    elif tx > gx and not env.blocked((gx + 1, gy)):
                        self.pos = (gx + 1, gy)
                        return 'RIGHT'
                    elif tx < gx and not env.blocked((gx - 1, gy)):
                        self.pos = (gx - 1, gy)
                        return 'LEFT'
                else:
                    # Already in the row: stay here by moving horizontally towards target's x
                    if tx > gx and not env.blocked((gx + 1, gy)):  # Right
                        self.pos = (gx + 1, gy)
                        return 'RIGHT'
                    elif tx < gx and not env.blocked((gx - 1, gy)):  # Left
                        self.pos = (gx - 1, gy)
                        return 'LEFT'
                    # If horizontal blocked, try vertical (rare, but to avoid getting stuck)
                    elif ty > gy and not env.blocked((gx, gy + 1)):
                        self.pos = (gx, gy + 1)
                        return 'DOWN'
                    elif ty < gy and not env.blocked((gx, gy - 1)):
                        self.pos = (gx, gy - 1)
                        return 'UP'

            dirs = [('UP',(0,-1)),('DOWN',(0,1)),('LEFT',(-1,0)),('RIGHT',(1,0))]
            random.shuffle(dirs)
            for name,(dx,dy) in dirs:
                nx, ny = gx+dx, gy+dy
                if not env.blocked((nx, ny)):
                    self.pos = (nx, ny)
                    return name
            return 'WAIT'
    
class FOLGhost3:
    Coord = Tuple[int, int]

    def __init__(self, start: Coord,sight: int):
        self.pos = start
        self.start = start
        self.sight = sight

    def perceive_and_update_kb(self, env: Environment) -> None:
        # This ghost does not add facts itself; it reads env.fol_kb
        pass

    def next_action(self, env: Environment) -> str:
        if env.fol_kb is None:
            return 'WAIT'

        gx, gy = self.pos

        px, py = env.pacman_pos
        if abs(px - gx) + abs(py - gy) > self.sight:
                dirs = [('UP', (0, -1)), ('DOWN', (0, 1)), ('LEFT', (-1, 0)), ('RIGHT', (1, 0))]
                random.shuffle(dirs)  # Randomize order
                for name, (dx, dy) in dirs:
                    nx, ny = gx + dx, gy + dy
                    if not env.blocked((nx, ny)):
                        self.pos = (nx, ny)
                        return name
                # If all directions blocked (rare), wait
                return 'WAIT'

        # 1) ask KB for predicted next position(s)
        predicted = env.fol_kb.infer_next_likely(env)

        candidates = []
        for nl in predicted:
            x = nl.x.value
            y = nl.y.value
            candidates.append((x, y))



                    
        # choose nearest visible predicted cell
        tx, ty = min(candidates, key=lambda c: abs(c[0]-gx) + abs(c[1]-gy))
        dx = 1 if tx > gx else (-1 if tx < gx else 0)
        dy = 1 if ty > gy else (-1 if ty < gy else 0)

        if dx != 0:
            nx, ny = gx + dx, gy
            if not env.blocked((nx, ny)):
                self.pos = (nx, ny)
                return 'RIGHT' if dx == 1 else 'LEFT'
        if dy != 0:
            nx, ny = gx, gy + dy
            if not env.blocked((nx, ny)):
                self.pos = (nx, ny)
                return 'DOWN' if dy == 1 else 'UP'

        return 'WAIT'

def generate_maze(
    w: int,
    h: int,
    wall_density: float = 0.15,
    pellet_density: float = 0.15
) -> Tuple[Set[Coord], Set[Coord], Coord]:
    """Generate walls, pellets, and the Pac-Man start position."""
    # Add random walls
    rng = random.Random()
    all_positions = [(x, y) for y in range(0, h) for x in range(0, w)]
    k_walls = int(wall_density * len(all_positions))
    walls = set(rng.sample(all_positions, k_walls)) if k_walls > 0 else set()

    # Ensure Pac-Man's starting position does not contain a wall
    pacman_start = (0, 0)
    walls.discard(pacman_start)
    ghost1_start = (10,5)
    walls.discard(ghost1_start)
    ghost2_start= (14,10)
    walls.discard(ghost2_start)
    ghost3_start=(8,3)
    walls.discard(ghost3_start)




    # Place pellets in free spaces
    free_cells = [c for c in all_positions if c not in walls and c != pacman_start]
    k_pellets = max(1, int(pellet_density * len(free_cells)))
    pellets = set(rng.sample(free_cells, k_pellets)) if k_pellets > 0 else set()

    return walls, pellets, pacman_start, ghost1_start, ghost2_start, ghost3_start


def run_game(
    env: Environment,
    max_steps: int = 200,
    sleep_s: float = 0.5
):
    """Run the Pac-Man game with keyboard controls."""
    action = "WAIT"

    for _ in range(max_steps):
        if env.finished:
            break

        key = get_pressed_key()
        if key is not None:
            action = key

        if action == 'QUIT':
            break

        env.step(action)
        os.system('cls' if os.name == 'nt' else 'clear')

        print(env.render())
        print()
        time.sleep(sleep_s)


def run_pacman():
    """Game entry point: create a maze, instantiate the environment, run the game."""
    
    width, height = 20, 20 
    walls, pellets, pacman_start, ghost1_start,ghost2_start,ghost3_start = generate_maze(w=width, h=height)
    env = Environment(
        width, height,
        walls=walls,
        pellets=pellets,
        start_pos=pacman_start,
        lives=3,
    )
    env.fol_kb = FOL_KB(env)

    kb_ghost = Ghost1(start=ghost1_start, sight=6)
    kb_ghost2 = Ghost2(start=ghost2_start, sight=6)
    fol_ghost3 = FOLGhost3(start=ghost3_start, sight=6)
    env.ghosts.append(kb_ghost)
    env.ghosts.append(kb_ghost2)
    env.ghosts.append(fol_ghost3)


    run_game(env)


if __name__ == "__main__":
    run_pacman()
