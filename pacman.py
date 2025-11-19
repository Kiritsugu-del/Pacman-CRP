from typing import Tuple, Set, Dict, List
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

    ):
        self.w, self.h = w, h
        self.walls: Set[Coord] = set(walls or set())
        self.pellets: Set[Coord] = set(pellets or set())
        self.pacman_pos: Coord = start_pos
        self.time: int = 0
        self.finished: bool = False
        self.ghosts: List['Ghost'] = []  
        self.score = 0


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

        # Move according to the action
        moves = {'RIGHT': (1, 0), 'LEFT': (-1, 0), 'DOWN': (0, 1), 'UP': (0, -1)}
        if action in moves:
            dx, dy = moves[action]
            nx, ny = self.pacman_pos[0] + dx, self.pacman_pos[1] + dy
            if not self.blocked((nx, ny)):
                self.pacman_pos = (nx, ny)
        
        if self.pacman_pos in self.pellets:
            self.pellets.remove(self.pacman_pos)
            self.score += 10

        # Collect pellet if needed
        if self.pacman_pos in self.pellets:
            self.pellets.remove(self.pacman_pos)
        
        for ghost in self.ghosts:
            ghost.perceive_and_update_kb(self)
            ghost.next_action(self)

        # Check collision with ghosts
        for ghost in self.ghosts:
            if ghost.pos == self.pacman_pos:
                self.finished = True
                break

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
        status_line = f"t={self.time} | pellets={len(self.pellets)} | scoreu={self.score}"
        buf.append(status_line)

        ghost_positions = {g.pos for g in self.ghosts}

        for y in range(self.h):
            row = []
            for x in range(self.w):
                c = (x, y)
                if c in ghost_positions:
                    ch = 'G'
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


class Ghost:

    Coord = Tuple[int,int]

    def __init__(self,start: Coord, sight: int = 4):
        self.pos = start
        self.sight = sight
        self.kb = KB()

     
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
            for dir_name, (dx,dy) in {'UP':(0,-1),'DOWN':(0,1),'LEFT':(-1,0),'RIGHT':(1,0)}.items():
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
                # choose closest target
                tx, ty = min(targets, key=lambda c: abs(c[0]-gx)+abs(c[1]-gy))
                # simple greedy step toward target
                dx = 1 if tx > gx else (-1 if tx < gx else 0)
                dy = 1 if ty > gy else (-1 if ty < gy else 0)
                # prefer horizontal move if both available
                if dx != 0 and not env.blocked((gx+dx, gy)):
                    self.pos = (gx+dx, gy)
                    return 'RIGHT' if dx==1 else 'LEFT'
                if dy != 0 and not env.blocked((gx, gy+dy)):
                    self.pos = (gx, gy+dy)
                    return 'DOWN' if dy==1 else 'UP'

            # 3) fallback random legal move
            dirs = [('UP',(0,-1)),('DOWN',(0,1)),('LEFT',(-1,0)),('RIGHT',(1,0))]
            random.shuffle(dirs)
            for name,(dx,dy) in dirs:
                nx, ny = gx+dx, gy+dy
                if not env.blocked((nx, ny)):
                    self.pos = (nx, ny)
                    return name
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
    ghost1_start = (10,7)
    walls.discard(ghost1_start)



    # Place pellets in free spaces
    free_cells = [c for c in all_positions if c not in walls and c != pacman_start]
    k_pellets = max(1, int(pellet_density * len(free_cells)))
    pellets = set(rng.sample(free_cells, k_pellets)) if k_pellets > 0 else set()

    return walls, pellets, pacman_start, ghost1_start


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

        print(env.render())
        print()
        time.sleep(sleep_s)


def run_pacman():
    """Game entry point: create a maze, instantiate the environment, run the game."""
    
    width, height = 20, 20 
    walls, pellets, pacman_start, ghost1_start = generate_maze(w=width, h=height)

    env = Environment(
        width, height,
        walls=walls,
        pellets=pellets,
        start_pos=pacman_start,


    )

    kb_ghost = Ghost(start=ghost1_start, sight=6)
    env.ghosts.append(kb_ghost)


    run_game(env)


if __name__ == "__main__":
    run_pacman()
