from typing import Tuple, Set, Dict, List
from fol_components import Constant, Variable, Predicate, LocationBasedPredicate, Moved, NextLikely, unify
from collections import deque
import random
import os
import sys
import time
import random

Coord = Tuple[int, int]

def find_shortest_move(start: Coord, target: Coord, env: 'Environment'):
    """
    Encontra o primeiro movimento (str) do caminho mais curto de start para target usando BFS.
    Devolve None se não houver caminho.
    """
    if start == target:
        return 'WAIT'

    # predecessor: mapeia posição -> (pos_anterior, nome_do_movimento_para_pos_anterior)
    # Aqui, predecessor[pos] armazena o *nome do movimento* que levou à pos.
    # Vamos rastrear (pos_anterior, nome_do_movimento_para_pos)
    predecessor: Dict[Coord, Tuple[Coord, str]] = {start: (start, 'START')}
    queue = deque([start])
    
    moves = {'LEFT': (-1, 0), 'RIGHT': (1, 0), 'UP': (0, -1), 'DOWN': (0, 1)}
    
    # Execução da BFS
    target_found = None
    while queue:
        cx, cy = queue.popleft()
        
        # Tentamos movimentos em todas as 4 direções
        shuffled_moves = list(moves.items())
        random.shuffle(shuffled_moves) # Randomizar para evitar vieses em caminhos de igual comprimento
        
        for name, (dx, dy) in shuffled_moves:
            next_pos = (cx + dx, cy + dy)

            if next_pos not in predecessor and not env.blocked(next_pos):
                predecessor[next_pos] = ((cx, cy), name)
                
                if next_pos == target:
                    target_found = next_pos
                    break
                queue.append(next_pos)
        
        if target_found:
            break
            
    if target_found is None:
        return 'WAIT' # Caminho não encontrado

    # Reconstruir o primeiro movimento do caminho mais curto
    current = target_found
    first_move_name = 'WAIT'
    
    while current != start:
        pred_pos, move_name = predecessor[current]
        first_move_name = move_name # move_name é o movimento de pred_pos para current
        current = pred_pos
        
    return first_move_name

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
        self.time: int = 0
        self.finished: bool = False
        self.ghost_targets: Dict['Ghost1', Coord] = {} # <---
        self.ghosts: List['Ghost1','Ghost2'] = []
        self.score = 0
        self.lives=lives
        self.start_pos = start_pos
        self.skip_ghosts_one_turn = False

    def respawn_ghost(self, ghost):
        if hasattr(ghost, "start"):
            ghost.pos = ghost.start

        # RESET TOTAL DO ESTADO DO FANTASMA
        if hasattr(ghost, "kb"):
            ghost.kb.clear()

        if hasattr(ghost, "current_target"):
            ghost.current_target = None

        if hasattr(ghost, "adjacent_counter"):
            ghost.adjacent_counter = 0

        if hasattr(ghost, "rest_turns"):
            ghost.rest_turns = 0

        if hasattr(ghost, "patrol_cycle"):
            ghost.patrol_cycle = 0

        if hasattr(ghost, "pacman_last_seen"):
            ghost.pacman_last_seen = None

        if hasattr(ghost, "visited_cells"):
            ghost.visited_cells.clear()

        if hasattr(ghost, "chase_timer"):
            ghost.chase_timer = 0

        if hasattr(ghost, "turns_without_cluster"):
            ghost.turns_without_cluster = 0

    def set_ghost_targets(self) -> None:
        self.ghost_targets = {ghost: ghost.current_target for ghost in self.ghosts if ghost.current_target is not None}
    
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
        if self.finished:
            return
        self.time += 1
        self.pacman_prev_pos=self.pacman_pos 
        # POSIÇÃO ANTERIOR
        ox, oy = self.pacman_pos
        nx, ny = ox, oy  # por defeito, Pac-Man não se move

        # MOVIMENTO DO PAC-MAN
        moves = {'RIGHT': (1, 0), 'LEFT': (-1, 0), 'DOWN': (0, 1), 'UP': (0, -1)}
        if action in moves:
            dx, dy = moves[action]
            cand = (ox + dx, oy + dy)
            if not self.blocked(cand):
                nx, ny = cand
        # atualizar posição real
        self.pacman_pos = (nx, ny)
        # ATUALIZAR KB SOMENTE SE HOUVE MOVIMENTO
        if (nx, ny) != (ox, oy) and self.fol_kb is not None:
            from fol_components import Constant, Moved
            dx = nx - ox
            dy = ny - oy
            self.fol_kb.add_fact(
                Moved(
                    Constant("Pacman"),
                    Constant(ox), Constant(oy),
                    Constant(dx), Constant(dy)
                )
            )


        # Colisão após o Pac-Man mover
        for ghost in self.ghosts:
            if ghost.pos == self.pacman_pos:
                self.lives -= 1
                if self.lives == 0:
                    self.finished = True
                    return
                # Reset do Pac-Man
                self.pacman_pos = self.start_pos
                # Reset de TODOS os fantasmas
                for g in self.ghosts:
                    self.respawn_ghost(g)
                self.skip_ghosts_one_turn = True   # <--- impedir ghosts de agirem neste frame
                return
        
        # Check if no pellets are left
        if len(self.pellets) == 0:
            self.finished = True
            return

        # recolher pellet
        if self.pacman_pos in self.pellets:
            self.pellets.remove(self.pacman_pos)
            self.score += 10

        # fantasmas não agem se pacman tiver morrido
        if self.skip_ghosts_one_turn:
            self.skip_ghosts_one_turn = False
            return

        # Fantasmas agem (e limpam target se chegaram)
        for ghost in self.ghosts:
            ghost.next_action(self)
        
        # Fantasmas percebem DEPOIS de pacman mover
        for ghost in self.ghosts:
            ghost.perceive_and_update_kb(self)

        # Verificar colisão após Ghost mover
        for ghost in self.ghosts:
            if ghost.pos == self.pacman_pos:
                self.lives -= 1
                if self.lives == 0:
                    self.finished = True
                    return
                self.pacman_pos = self.start_pos
                for g in self.ghosts:
                    self.respawn_ghost(g)
                self.skip_ghosts_one_turn = True
                return

    def render(self) -> str:
        """Return a multi-line string visualization of the grid.

        Legend:
            'P' - Pac-Man
            '#' - Wall
            '.' - Pellet
            ' ' - Empty space
            '1, 2, 3' - Ghosts
        """
        buf: List[str] = []
        status_line = f"t={self.time} | pellets={len(self.pellets)} | score={self.score} | lives = {self.lives}"
        buf.append(status_line)
        
        ghost_positions: Dict[Coord, int] = {g.pos: i for i, g in enumerate(self.ghosts)}
        for y in range(self.h):
            row = []
            for x in range(self.w):
                c = (x, y)
                if c in ghost_positions:
                    ch = str(ghost_positions[c] + 1)
                elif c == self.pacman_pos:
                    ch = 'P'
                # elif c in target_positions:
                #     ch = 'x'                # <--- o target aparece aqui
                elif c in self.walls:
                    ch = '#'
                elif c in self.pellets:
                    ch = '.'
                else:
                    ch = ' '

                row.append(f"[{ch}]")
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
        self.current_target = None   # <---
        self.adjacent_counter = 0   # nº de turnos consecutivos adjacente ao Pac-Man
        self.rest_turns = 0       # nº de turnos restantes que o fantasma está a descansarº
        self.start = start

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
        # Se current_target não for cluster válido, limpa!
        if self.current_target:
            cx, cy = self.current_target
            if f"Cluster_{cx}_{cy}" not in self.kb.facts:
                self.current_target = None

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
    Coord = Tuple[int, int]

    def __init__(self, start: Coord, sight: int = 4):
        self.pos = start
        self.start = start
        self.sight = sight
        self.kb = KB()

        # Estado principal
        self.current_cluster = None        # {"center": (cx,cy), "pellets": [...]}
        self.current_target = None         # (cx,cy) ou pellet isolado

        # Perseguir Pac-Man sem destruir cluster
        self.pacman_last_seen = None
        self.chase_timer = 0

        # Patrulha
        self.patrol_cycle = 0

        # Exploração
        self.visited_cells = set()

        # Pellets & fallback
        self.turns_without_cluster = 0
        self.pellets_seen_now = set()
        self.visited_isolated_pellets = set()
        self.last_isolated_pellet = None

    # ===============================================================
    # PERCEÇÃO
    # ===============================================================
    def perceive_and_update_kb(self, env: Environment) -> None:
        gx, gy = self.pos

        # 1) limpar transitórios
        self.kb.facts = {
            f for f in self.kb.facts
            if not f.startswith("PacmanPos_") and not f.startswith("Ghost_")
        }
        self.kb.add_fact(f"Ghost_{gx}_{gy}")
        self.pacman_last_seen = None

        # 2) perceber LOS
        visible_cells = self._get_line_of_sight(env)
        pellets_seen_now = set()

        for x, y, is_wall, is_pellet, sees_pacman in visible_cells:
            if is_wall:
                self.kb.add_fact(f"Wall_{x}_{y}")

            if is_pellet:
                pellets_seen_now.add((x, y))
                self.kb.add_fact(f"Pellet_{x}_{y}")

            if sees_pacman:
                self.kb.add_fact(f"PacmanPos_{x}_{y}")
                self.pacman_last_seen = (x, y)

        self.pellets_seen_now = pellets_seen_now

        # 3) remover pellets QUE ESTAVAM NA LOS mas desapareceram
        los_coords = {(vx, vy) for (vx, vy, _, _, _) in visible_cells}
        for fact in list(self.kb.facts):
            if fact.startswith("Pellet_"):
                _, sx, sy = fact.split("_")
                px, py = int(sx), int(sy)
                if (px, py) in los_coords and (px, py) not in pellets_seen_now:
                    self.kb.facts.remove(fact)


        # 4) Recalcular clusters
        self.kb.facts = {f for f in self.kb.facts if not f.startswith("Cluster_")}
        clusters = self._infer_4x4_clusters(env)
        for cl in clusters:
            cx, cy = cl["center"]
            self.kb.add_fact(f"Cluster_{cx}_{cy}")

        # Se nada visível → limpar pellet isolado pendente
        if not self.pellets_seen_now:
            self.last_isolated_pellet = None


        self.kb.infer_all()

    # ===============================================================
    # LOS
    # ===============================================================
    def _get_line_of_sight(self, env: Environment):
        gx, gy = self.pos
        visible = []
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            for i in range(1, self.sight + 1):
                nx, ny = gx + dx*i, gy + dy*i
                if not env.in_bounds((nx, ny)):
                    break
                is_wall = (nx, ny) in env.walls
                is_pellet = (nx, ny) in env.pellets
                sees_pacman = (nx, ny) == env.pacman_pos
                visible.append((nx, ny, is_wall, is_pellet, sees_pacman))
                if is_wall:
                    break
        return visible

    # ===============================================================
    # AÇÃO
    # ===============================================================
    def next_action(self, env: Environment) -> str:
        gx, gy = self.pos
        self.visited_cells.add((gx, gy))

        # ------------------------------
        # 0) Lidar com Pac-Man (sem destruir cluster)
        # ------------------------------
        pacman_pos = None
        pac_pos_list = [
            (int(f.split('_')[1]), int(f.split('_')[2]))
            for f in self.kb.facts if f.startswith("PacmanPos_")
        ]
        if pac_pos_list:
            pacman_pos = pac_pos_list[0]

        if pacman_pos:
            self.pacman_last_seen = pacman_pos
            dist = abs(gx - pacman_pos[0]) + abs(gy - pacman_pos[1])

            # perseguição curta (sem destruir target de cluster)
            if dist <= 2 and self.chase_timer < 5:
                self.chase_timer += 1
                act = self._move_towards(env, pacman_pos, ignore_walls=True)
                if act != "WAIT":
                    return act

            self.chase_timer = 0

            # bloqueio entre Pac-Man e o cluster que ele guarda
            if self.current_cluster:
                act = self._move_to_block(
                    env,
                    pacman_pos,
                    self.current_cluster["center"]
                )
                if act != "WAIT":
                    return act
        else:
            self.chase_timer = 0

        # ===============================================================
        # 0.5) Validar cluster atual (antes da patrulha)
        # ===============================================================
        if self.current_cluster:
            still_exists = True
            for (px, py) in self.current_cluster["pellets"]:
                if f"Pellet_{px}_{py}" not in self.kb.facts:
                    still_exists = False
                    break

            if not still_exists:
                #print("    >>> Cluster destruído! Resetando!")
                self.current_cluster = None
                self.patrol_cycle = 0


        # ===============================================================
        # 1) PATRULHA DO CLUSTER ATUAL (se existe)
        # ===============================================================
        if self.current_cluster:
            cx, cy = self.current_cluster["center"]

            # patrulhar SEMPRE que existe cluster
            perimeter = [
                # topo da janela 4×4
                (cx-1, cy-1), (cx, cy-1), (cx+1, cy-1), (cx+2, cy-1),
                # direita
                (cx+2, cy), (cx+2, cy+1),
                # baixo
                (cx+2, cy+2), (cx+1, cy+2), (cx, cy+2), (cx-1, cy+2),
                # esquerda
                (cx-1, cy+1), (cx-1, cy)
            ]

            route = [
                p for p in perimeter
                if env.in_bounds(p) and not self._knows_wall(p[0], p[1])
            ]

            if route:
                idx = self.patrol_cycle % len(route)
                act = self._move_towards(env, route[idx], ignore_walls=False)
                if act != "WAIT":
                    self.patrol_cycle += 1
                    return act


        # ===============================================================
        # 2) Procurar cluster NOVO APENAS SE NÃO TIVER cluster atual
        # ===============================================================
        if self.current_cluster is None:
            clusters = self._infer_4x4_clusters(env)
            if clusters:
                self.turns_without_cluster = 0
                self.current_cluster = min(
                    clusters,
                    key=lambda c: abs(c["center"][0] - gx) + abs(c["center"][1] - gy)
                )

        # ===============================================================
        # 3) SEM CLUSTERS → transformar pellet isolado num cluster falso
        # ===============================================================
        self.turns_without_cluster += 1
        print(f"\n{self.turns_without_cluster}\n")
        if self.turns_without_cluster >= 20 and self.current_cluster is None:

            pellet_coords = [
                (int(f.split('_')[1]), int(f.split('_')[2]))
                for f in self.kb.facts if f.startswith("Pellet_")
            ]

            candidates = [p for p in pellet_coords
                        if p not in self.visited_isolated_pellets]

            if candidates:
                # escolher pellet isolado mais próximo
                if self.last_isolated_pellet is None:
                    self.last_isolated_pellet = min(
                        candidates, key=lambda c: abs(c[0]-gx) + abs(c[1]-gy)
                    )

                isolated = self.last_isolated_pellet

                # Criar CLUSTER FALSO
                self.current_cluster = {
                    "center": isolated,
                    "pellets": [isolated]
                }

                # Preparar patrulha normal
                self.patrol_cycle = 0
                self.current_target = isolated
                self.turns_without_cluster = 0     # <<< MUITO IMPORTANTE

                # Mover para o centro (pellet isolado)
                act = self._move_towards(env, isolated, ignore_walls=False)
                if act != "WAIT":
                    # se chegou lá, marcar como visitado e deixar patrulha assumir depois
                    if self.pos == isolated:
                        self.visited_isolated_pellets.add(isolated)
                        self.last_isolated_pellet = None
                    return act


        # ===============================================================
        # 4) fallback = exploração aleatória
        # ===============================================================
        self.current_target = None
        return self._move_randomly(env)

    # ===============================================================
    # HELPERS DE MOVIMENTO
    # ===============================================================
    def _knows_wall(self, x, y):
        return f"Wall_{x}_{y}" in self.kb.facts

    def _move_randomly(self, env):
        gx, gy = self.pos
        options = []
        for name, (dx, dy) in [
            ('UP',(0,-1)),('DOWN',(0,1)),('LEFT',(-1,0)),('RIGHT',(1,0))
        ]:
            nx, ny = gx+dx, gy+dy
            if env.in_bounds((nx,ny)) and not self._knows_wall(nx,ny):
                cost = 0 if (nx,ny) not in self.visited_cells else 1
                options.append((cost,name,(nx,ny)))
        if not options:
            return "WAIT"
        options.sort(key=lambda x: x[0])
        _, act, pos = random.choice([o for o in options if o[0] == options[0][0]])
        self.pos = pos
        return act

    def _move_towards(self, env, target, ignore_walls=False):
        gx, gy = self.pos
        tx, ty = target

        moves = []
        if abs(tx-gx) >= abs(ty-gy):
            if tx != gx: moves.append((1 if tx>gx else -1, 0))
            if ty != gy: moves.append((0, 1 if ty>gy else -1))
        else:
            if ty != gy: moves.append((0, 1 if ty>gy else -1))
            if tx != gx: moves.append((1 if tx>gx else -1, 0))

        for dx, dy in moves:
            nx, ny = gx+dx, gy+dy
            if not env.in_bounds((nx,ny)):
                continue

            blocked = self._knows_wall(nx,ny)
            if ignore_walls:
                blocked = env.blocked((nx,ny))

            if not blocked:
                self.pos = (nx,ny)
                return {(-1,0):"LEFT",(1,0):"RIGHT",(0,-1):"UP",(0,1):"DOWN"}[(dx,dy)]

        return "WAIT"

    def _move_to_block(self, env, pac, cluster_center):
        gx, gy = self.pos
        px, py = pac
        cx, cy = cluster_center

        best = None
        best_cost = float('inf')

        pc_vec = (cx-px, cy-py)

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = gx+dx, gy+dy
            if not env.in_bounds((nx,ny)) or self._knows_wall(nx,ny):
                continue

            pg_vec = (nx-px, ny-py)
            cost = abs(pg_vec[0]-pc_vec[0]) + abs(pg_vec[1]-pc_vec[1])

            if cost < best_cost:
                best_cost = cost
                best = (dx,dy)

        if best:
            dx,dy = best
            self.pos = (gx+dx, gy+dy)
            return {(-1,0):"LEFT",(1,0):"RIGHT",(0,-1):"UP",(0,1):"DOWN"}[(dx,dy)]
        return "WAIT"

    # ===============================================================
    # CLUSTERS 4×4
    # ===============================================================
    def _infer_4x4_clusters(self, env):
        pellet_coords = {
            (int(f.split('_')[1]), int(f.split('_')[2]))
            for f in self.kb.facts if f.startswith("Pellet_")
        }

        clusters = []
        seen_centers = set()

        if not pellet_coords:
            return clusters

        for wy in range(env.h - 3):
            for wx in range(env.w - 3):

                # pellets dentro da janela 4x4 exata
                window = [
                    p for p in pellet_coords
                    if wx <= p[0] <= wx+3 and wy <= p[1] <= wy+3
                ]

                if len(window) >= 4:
                    # calcular centro REAL (mediana)
                    xs = sorted([p[0] for p in window])
                    ys = sorted([p[1] for p in window])
                    cx = xs[len(xs)//2]
                    cy = ys[len(ys)//2]

                    # evitar duplicados
                    if (cx, cy) not in seen_centers:
                        clusters.append({
                            "center": (cx, cy),
                            "pellets": window
                        })
                        seen_centers.add((cx, cy))

        return clusters

class FOLGhost3:
    Coord = Tuple[int, int]
    
    def perceive_and_update_kb(self, env: Environment) -> None:
        pass

    def __init__(self, start: Coord, sight: int):
        self.pos = start
        self.start = start
        self.sight = sight
        # Ghost 3 precisa de uma KB só para ele se quisermos memória persistente,
        # mas aqui vamos usar a KB principal do ambiente.

    def _random_move(self, env: Environment):
        """Helper para movimento aleatório em caso de fallback."""
        gx, gy = self.pos
        dirs = [('UP', (0, -1)), ('DOWN', (0, 1)), ('LEFT', (-1, 0)), ('RIGHT', (1, 0))]
        random.shuffle(dirs)
        for name, (dx, dy) in dirs:
            nx, ny = gx + dx, gy + dy
            if not env.blocked((nx, ny)):
                self.pos = (nx, ny)
                return name
        return 'WAIT'
    
    def _pacman_in_cross_sight(self, env: Environment):
        """
        Verifica se o Pac-Man está visível (visão em cruz de 4 blocos).
        Devolve a posição do Pac-Man se for visto, caso contrário, None.
        """
        gx, gy = self.pos
        px, py = env.pacman_pos
        
        # 1. Visão horizontal (ao longo da linha x)
        if py == gy and abs(px - gx) <= self.sight:
            # Verifica se há paredes entre o fantasma e o Pac-Man na linha
            dx_step = 1 if px > gx else -1
            for x in range(gx + dx_step, px, dx_step):
                if env.blocked((x, gy)):
                    break
            else:
                return (px, py)
        
        # 2. Visão vertical (ao longo da linha y)
        if px == gx and abs(py - gy) <= self.sight:
            # Verifica se há paredes entre o fantasma e o Pac-Man na coluna
            dy_step = 1 if py > gy else -1
            for y in range(gy + dy_step, py, dy_step):
                if env.blocked((gx, y)):
                    break
            else:
                return (px, py)
                
        return None

    def next_action(self, env: Environment) -> str:
        if env.fol_kb is None:
            return 'WAIT'

        gx, gy = self.pos
        pacman_pos = env.pacman_pos
        
        # 1. Verificar se o Pac-Man está em visão (Cruz de 4 blocos)
        seen_pos = self._pacman_in_cross_sight(env)
        
        if seen_pos is None:
            # Pac-Man não está visível: Patrulha/Movimento aleatório
            return self._random_move(env)

        # Pac-Man Visto:
        
        # 2. Obter Posições Prováveis (FOL)
        predicted = env.fol_kb.infer_next_likely(env)
        
        target_pos = None
        
        if predicted:
            # Se houver previsões FOL, o alvo é a previsão mais próxima.
            candidates = []
            for nl in predicted:
                candidates.append((nl.x.value, nl.y.value))
                
            # Escolher a previsão mais próxima (em distância Manhattan)
            target_pos = min(candidates, key=lambda c: abs(c[0]-gx) + abs(c[1]-gy))
            
        else:
            # Se não houver previsão FOL (ex: Pac-Man parado ou recém-visto),
            # o alvo é a posição ATUAL do Pac-Man (perseguição direta).
            target_pos = seen_pos
        
        # 3. Calcular o Melhor Movimento para o Alvo (Usando BFS)
        
        # A função find_shortest_move deve estar definida globalmente ou importada.
        next_move_name = find_shortest_move(self.pos, target_pos, env)
        
        # 4. Aplicar o Movimento
        if next_move_name == 'WAIT':
            # Não encontrou caminho para o alvo (ex: canto fechado, ou alvo inalcançável)
            return self._random_move(env)
        
        moves_map = {'LEFT': (-1, 0), 'RIGHT': (1, 0), 'UP': (0, -1), 'DOWN': (0, 1)}
        dx, dy = moves_map[next_move_name]
        
        nx, ny = gx + dx, gy + dy
        self.pos = (nx, ny)
        
        return next_move_name

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

    # Place pellets in free spaces
    free_cells = [c for c in all_positions if c not in walls and c != pacman_start]
    k_pellets = max(1, int(pellet_density * len(free_cells)))
    pellets = set(rng.sample(free_cells, k_pellets)) if k_pellets > 0 else set()

    return walls, pellets, pacman_start


def run_game(
    env: Environment,
    max_steps: int = 200,
    sleep_s: float = 0.3
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

def random_spawn_in_quadrant(
    x_min: int, x_max: int,
    y_min: int, y_max: int,
    walls: Set[Coord]
) -> Coord:
    """Devolve uma célula aleatória dentro do quadrante definido,
       que não seja parede."""
    rng = random.Random()
    valid_cells = [
        (x, y)
        for x in range(x_min, x_max + 1)
        for y in range(y_min, y_max + 1)
        if (x, y) not in walls
    ]
    return rng.choice(valid_cells)

def run_pacman():
    """Game entry point: create a maze, instantiate the environment, run the game."""
    
    width, height = 20, 20 
    walls, pellets, pacman_start = generate_maze(w=width, h=height)
    env = Environment(
        width, height,
        walls=walls,
        pellets=pellets,
        start_pos=pacman_start,
        lives=3,
    )
    env.fol_kb = FOL_KB(env)

    # Quadrantes:
    # Q2 = x 10–19, y 0–9
    # Q3 = x 0–9 , y 10–19
    # Q4 = x 10–19, y 10–19

    ghost1_start = random_spawn_in_quadrant(11, 19, 0, 10, walls)   # Q1
    ghost2_start = random_spawn_in_quadrant(0, 10, 11, 19, walls)   # Q3
    ghost3_start = random_spawn_in_quadrant(11, 19, 11, 19, walls)  # Q4
    
    kb_ghost1 = Ghost1(start=ghost1_start, sight=4)
    kb_ghost2 = Ghost2(start=ghost2_start, sight=4)
    fol_ghost3 = FOLGhost3(start=ghost3_start, sight=6)

    env.ghosts.append(kb_ghost1)
    env.ghosts.append(kb_ghost2)
    env.ghosts.append(fol_ghost3)

    run_game(env)


if __name__ == "__main__":
    run_pacman()
