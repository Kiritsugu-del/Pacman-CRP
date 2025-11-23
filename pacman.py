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

#Thid ghost goes around clusters of pellets to protect them 
class Ghost2:
    Coord = Tuple[int,int]

    def __init__(self,start: Coord, sight: int = 4):
        self.pos = start
        self.start = start
        self.sight = sight
        self.kb = KB()
        self.current_target = None       # Coordenada do centro do cluster a proteger
        self.pacman_last_seen = None     # Última posição de Pac-Man vista
        self.patrol_cycle = 0            # Ajuda a definir o movimento de Patrulha
        self.chase_timer = 0             # Contador de turnos de perseguição/confronto
        self.visited_cells = set()       # Para Exploração Aleatória
        self.turns_without_cluster = 0
        self.pellets_seen_now = set()
        self.last_isolated_pellet = None
        self.isolated_visit_done = False

    def perceive_and_update_kb(self, env: Environment) -> None:
        gx, gy = self.pos

        # 0. Limpar factos transitórios (PacmanPos_x_y, Ghost_x_y)
        # O self.kb.clear_transitory() não existe, usamos o padrão de filtragem:
        self.kb.facts = {
            f for f in self.kb.facts 
            if not f.startswith("PacmanPos_") and not f.startswith("Ghost_")
        }
        self.kb.add_fact(f"Ghost_{gx}_{gy}") # Adiciona a posição atual do Ghost
        self.pacman_last_seen = None # Reset do estado interno de Pac-Man

        # 1. PERCEBER PAREDES, PELLETS E PACMAN
        visible_cells = self._get_line_of_sight(env) # Helper function a implementar
        pellet_facts_in_kb = {f for f in self.kb.facts if f.startswith("Pellet_")}
        pellets_seen_now: set[tuple[int, int]] = set()

        for x, y, is_wall, is_pellet, sees_pacman in visible_cells:
            # Perceber Paredes
            if is_wall:
                self.kb.add_fact(f"Wall_{x}_{y}")
            # Perceber Pellets / Confirmação de Consumo
            if is_pellet:
                pellet_atom = f"Pellet_{x}_{y}"
                self.kb.add_fact(pellet_atom)
                pellets_seen_now.add((x, y))
            # Perceber Pac-Man
            if sees_pacman:
                self.kb.add_fact(f"PacmanPos_{x}_{y}")
                self.pacman_last_seen = (x, y) # Atualiza a última posição vista
        
        self.pellets_seen_now = pellets_seen_now
        
        for fact in list(pellet_facts_in_kb):
            _, px, py = fact.split("_")
            p_coord = (int(px), int(py))
            # Se a célula esteve na LOS neste turno...
            if any((cx == p_coord[0] and cy == p_coord[1]) for cx, cy, *_ in visible_cells):
                # ...mas não foi vista como pellet, então foi comida
                if p_coord not in pellets_seen_now:
                    self.kb.facts.remove(fact)

        # 2. INFERÊNCIA DE CLUSTERS
        # (Remover clusters que já não existem antes de inferir novos)
        pellet_facts = [f for f in self.kb.facts if f.startswith("Pellet_")]
        cluster_facts_to_remove = []

        for cluster_fact in [f for f in self.kb.facts if f.startswith("Cluster_")]:
            _, cx, cy = cluster_fact.split("_")
            cx, cy = int(cx), int(cy)
            nearby_count = 0
            for p_fact in pellet_facts:
                _, px, py = p_fact.split("_")
                # Distância de Manhattan <= 2
                if abs(cx - int(px)) + abs(cy - int(py)) <= 2: 
                    nearby_count += 1
            # Lógica de Colapso do Cluster
            if nearby_count < 3: # Limiar de 3
                cluster_facts_to_remove.append(cluster_fact)
        for fact in cluster_facts_to_remove:
            self.kb.facts.remove(fact)
            if self.current_target == (int(fact.split('_')[1]), int(fact.split('_')[2])):
                self.current_target = None # Limpar alvo se cluster colapsar

        # Inferir novos clusters
        # 3. INFERIR CLUSTERS CORRETAMENTE (usando a lógica de 4x4)
        new_cluster_centers = self._infer_4x4_clusters(env)
        
        if new_cluster_centers:
            # Seleciona o centro mais próximo como alvo (ou mantém o alvo existente se for um cluster)
            gx, gy = self.pos
            closest_center = min(new_cluster_centers, key=lambda c: abs(c[0] - gx) + abs(c[1] - gy))
            
            # Adicionar factos de cluster para a KB
            for cx, cy in new_cluster_centers:
                cluster_atom = f"Cluster_{cx}_{cy}"
                self.kb.add_fact(cluster_atom)
                
            # Apenas clusters reais podem definir current_target (se não houver um target melhor)
            current_target_is_cluster = self.current_target and f"Cluster_{self.current_target[0]}_{self.current_target[1]}" in self.kb.facts
            
            if not current_target_is_cluster:
                self.current_target = closest_center
        else:
            # Não há clusters
            if self.current_target and f"Cluster_{self.current_target[0]}_{self.current_target[1]}" in self.kb.facts:
                self.current_target = None

        # Se neste turno não viu nenhum pellet, esquecemos o pellet isolado
        if not self.pellets_seen_now:
            self.last_isolated_pellet = None
            self.isolated_visit_done = False

        self.kb.infer_all() # Executar inferência se tivermos regras mais formais.


    def _get_line_of_sight(self, env: 'Environment') -> List[Tuple[int, int, bool, bool, bool]]:
        """
        Retorna uma lista de tuplos (x, y, is_wall, is_pellet, sees_pacman) 
        para todas as células visíveis dentro do raio de self.sight (4).
        A linha de visão é bloqueada por paredes.
        """
        gx, gy = self.pos
        px, py = env.pacman_pos
        sight_range = self.sight # Deve ser 4, mas usamos self.sight
        # Tuplos de (dx, dy) para as 4 direções cardeais
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        visible_info = []
        for dx, dy in directions:
            for i in range(1, sight_range + 1):
                nx, ny = gx + dx * i, gy + dy * i
                current_coord = (nx, ny)
                if not env.in_bounds(current_coord):
                    # Fim do mapa
                    break 
                is_wall = current_coord in env.walls
                is_pellet = current_coord in env.pellets
                sees_pacman = current_coord == env.pacman_pos
                # Adiciona a informação da célula vista
                visible_info.append((nx, ny, is_wall, is_pellet, sees_pacman))
                if is_wall:
                    # A parede bloqueia a visão para além dela [cite: 22]
                    break       
        return visible_info
        
    def next_action(self, env: 'Environment') -> str:
        gx, gy = self.pos
        self.visited_cells.add((gx, gy)) 

        # --- FASE 0: PACMAN (perseguição + bloqueio) ---
        pacman_coords = [
            (int(f.split('_')[1]), int(f.split('_')[2]))
            for f in self.kb.facts if f.startswith("PacmanPos_")
        ]
        pacman_seen_now = pacman_coords[0] if pacman_coords else None

        if pacman_seen_now:
            self.pacman_last_seen = pacman_seen_now
            dist = abs(gx - pacman_seen_now[0]) + abs(gy - pacman_seen_now[1])

            # 0.1 Perseguição curta (<=2 dist)
            if dist <= 2 and self.chase_timer < 5:
                self.chase_timer += 1
                action = self._move_towards(env, pacman_seen_now, ignore_walls=True)
                self.current_target = pacman_seen_now # Perseguição anula alvo de cluster
                if action != "WAIT":
                    return action

            # 0.2 Bloqueio (Pac-Man visto mas não muito perto ou perseguição falhou)
            self.chase_timer = 0
            cluster_targets = [
                (int(f.split('_')[1]), int(f.split('_')[2]))
                for f in self.kb.facts if f.startswith("Cluster_")
            ]
            if cluster_targets:
                # Bloqueia em direção ao cluster mais próximo
                closest_cluster = min(cluster_targets, key=lambda c: abs(c[0] - gx) + abs(c[1] - gy))
                self.current_target = closest_cluster # Redefine target para cluster
                action = self._move_to_block(env, pacman_seen_now, closest_cluster)
                if action != "WAIT":
                    return action
        else:
            self.chase_timer = 0

        # 🚨 Limpar targets inválidos (ex: pellet isolado, posição antiga, ruído)
        if self.current_target and f"Cluster_{self.current_target[0]}_{self.current_target[1]}" not in self.kb.facts:
            self.current_target = None
            
        # --- FASE 1: PATRULHA DE CLUSTERS (Prioridade 1) ---
        cluster_facts = [
            (int(f.split('_')[1]), int(f.split('_')[2]))
            for f in self.kb.facts if f.startswith("Cluster_")
        ]

        if cluster_facts:
            self.turns_without_cluster = 0
            
            # Escolher cluster mais próximo (se target não estiver já definido ou for target inválido)
            if self.current_target is None or self.current_target not in cluster_facts:
                self.current_target = min(cluster_facts, key=lambda c: abs(c[0] - gx) + abs(c[1] - gy))

            cx, cy = self.current_target

            # --- PERÍMETRO ALARGADO 5x5 ---
            # (Lógica de perímetro inalterada)
            perimeter = [
                (cx-2, cy-2), (cx-1, cy-2), (cx, cy-2), (cx+1, cy-2), (cx+2, cy-2),
                (cx+2, cy-1), (cx+2, cy), (cx+2, cy+1),
                (cx+2, cy+2), (cx+1, cy+2), (cx, cy+2), (cx-1, cy+2), (cx-2, cy+2),
                (cx-2, cy+1), (cx-2, cy), (cx-2, cy-1)
            ]

            route = [p for p in perimeter if env.in_bounds(p) and not self._knows_wall(p[0], p[1])]

            if route:
                patrol_idx = self.patrol_cycle % len(route)
                patrol_target = route[patrol_idx]
                action = self._move_towards(env, patrol_target, ignore_walls=False)
                if action != "WAIT":
                    self.patrol_cycle += 1
                    return action
            # Se o caminho de patrulha não existir, o fantasma desce para a próxima fase.
            # Não precisa de else, simplesmente a função continua.
        else:
            # Não há cluster — incrementar contador
            self.turns_without_cluster += 1
            self.patrol_cycle = 0 # Reiniciar ciclo de patrulha se um novo cluster for encontrado

        # --- FASE 2: Fallback para Pellet Isolado (Prioridade 2: Último Recurso) ---
        # Só executa se não houver clusters E o contador estiver esgotado
        if not cluster_facts and self.turns_without_cluster >= 35:
            # pellets conhecidos pela KB (inclui os que foram vistos mas não foram comidos)
            pellet_facts = [
                (int(f.split('_')[1]), int(f.split('_')[2]))
                for f in self.kb.facts if f.startswith("Pellet_")
            ]
            
            if pellet_facts:
                # Escolher pellet isolado mais próximo
                target_pellet = min(pellet_facts, key=lambda c: abs(c[0] - gx) + abs(c[1] - gy))
                self.current_target = target_pellet
                action = self._move_towards(env, target_pellet, ignore_walls=False)
                if action != "WAIT":
                    return action

        # --- FASE 3: Exploração Aleatória (Prioridade 3: Default) ---
        # É o default se não houver cluster, não houver Pac-Man e o contador de fallback não esgotou
        self.current_target = None
        return self._move_randomly(env)

    # --- Funções Helper de Movimento (A serem adicionadas à classe Ghost2) ---

    def _knows_wall(self, x: int, y: int) -> bool:
        """Verifica se a KB sabe que há uma parede em (x, y)."""
        return f"Wall_{x}_{y}" in self.kb.facts
    
    def _move_towards(self, env: 'Environment', target: Coord, ignore_walls: bool = False) -> str:
        """Move-se uma casa na direção do target (Distância de Manhattan)."""
        gx, gy = self.pos
        tx, ty = target
        moves = []
        # Prioriza movimento que reduz maior distância
        if abs(tx - gx) >= abs(ty - gy):
            dx = 1 if tx > gx else (-1 if tx < gx else 0)
            dy = 0
            if dx != 0: moves.append((dx, dy))
            dx = 0
            dy = 1 if ty > gy else (-1 if ty < gy else 0)
            if dy != 0: moves.append((dx, dy))
        else:
            dy = 1 if ty > gy else (-1 if ty < gy else 0)
            dx = 0
            if dy != 0: moves.append((dx, dy))
            dy = 0
            dx = 1 if tx > gx else (-1 if tx < gx else 0)
            if dx != 0: moves.append((dx, dy))

        for dx, dy in moves:
            nx, ny = gx + dx, gy + dy
            new_pos = (nx, ny)
            is_blocked = self._knows_wall(nx, ny)
            if ignore_walls:
                is_blocked = env.blocked(new_pos) # Usa a informação real do ambiente
            if env.in_bounds(new_pos) and not is_blocked:
                self.pos = new_pos
                return {(-1,0): 'LEFT', (1,0): 'RIGHT', (0,-1): 'UP', (0,1): 'DOWN'}.get((dx, dy), 'WAIT')     
        return 'WAIT'

    def _move_to_block(self, env: 'Environment', pacman_pos: Coord, cluster_target: Coord) -> str:
        """Move-se para a célula que fica entre Pac-Man e o Cluster, se possível."""
        gx, gy = self.pos
        px, py = pacman_pos
        cx, cy = cluster_target
        pc_vec = (cx - px, cy - py)
        best_move = None
        best_cost = float('inf')
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = gx + dx, gy + dy
            new_pos = (nx, ny)
            if not env.in_bounds(new_pos) or self._knows_wall(nx, ny):
                continue
            pg_vec = (nx - px, ny - py)
            cost = abs(pc_vec[0] - pg_vec[0]) + abs(pc_vec[1] - pg_vec[1])
            if cost < best_cost:
                best_cost = cost
                best_move = (dx, dy)

        if best_move:
            dx, dy = best_move
            self.pos = (gx + dx, gy + dy)
            return {(-1,0): 'LEFT', (1,0): 'RIGHT', (0,-1): 'UP', (0,1): 'DOWN'}.get(best_move, 'WAIT')
            
        return 'WAIT' # Se estiver bloqueado

    def _move_randomly(self, env: 'Environment') -> str:
        """Movimento aleatório, favorecendo células menos visitadas."""
        gx, gy = self.pos
        possible_moves = []
        for name, (dx, dy) in [('UP', (0, -1)), ('DOWN', (0, 1)), ('LEFT', (-1, 0)), ('RIGHT', (1, 0))]:
            nx, ny = gx + dx, gy + dy
            new_pos = (nx, ny)
            if env.in_bounds(new_pos) and not self._knows_wall(nx, ny):
                # Custo: número de vezes que a célula foi visitada (menos visitada = menor custo)
                cost = 0 if new_pos not in self.visited_cells else 1
                possible_moves.append((cost, name, new_pos))  
        if not possible_moves:
            return 'WAIT'
        # Escolhe a célula com o menor custo (menos visitada)
        possible_moves.sort(key=lambda x: x[0])
        # Se todas têm o mesmo custo (ou seja, todas foram visitadas), escolhe aleatoriamente
        best_cost = possible_moves[0][0]
        best_moves = [m for m in possible_moves if m[0] == best_cost]
        _, action_name, new_pos = random.choice(best_moves)
        self.pos = new_pos
        return action_name

    def _infer_4x4_clusters(self, env: 'Environment') -> List[Coord]:
        """
        Identifica clusters de 4x4.
        Um cluster existe se houver 3 ou mais pellets em qualquer sub-área 4x4.
        Retorna a lista de centros (x, y) dos clusters.
        """
        all_cluster_centers = []
        
        # Obter todos os pellets conhecidos
        pellet_facts = [f for f in self.kb.facts if f.startswith("Pellet_")]
        pellet_coords = set()
        for fact in pellet_facts:
            try:
                _, px, py = fact.split("_")
                pellet_coords.add((int(px), int(py)))
            except ValueError:
                continue

        # Iterar sobre todas as possíveis janelas 4x4
        # A janela 4x4 tem o canto superior esquerdo em (wx, wy)
        # Ela cobre as coordenadas: [wx, wx+3] x [wy, wy+3]
        for wy in range(env.h - 3):  # y vai de 0 até H-4
            for wx in range(env.w - 3):  # x vai de 0 até W-4
                pellets_in_cluster = []
                for px, py in pellet_coords:
                    # Verifica se o pellet está DENTRO da janela [wx, wx+3] x [wy, wy+3]
                    if wx <= px <= wx + 3 and wy <= py <= wy + 3:
                        pellets_in_cluster.append((px, py))

                if len(pellets_in_cluster) >= 3:
                    # Se for um cluster válido, calcula o centro (mediana)
                    xs = [c[0] for c in pellets_in_cluster]
                    ys = [c[1] for c in pellets_in_cluster]
                    # Centro pela mediana (para ser mais robusto)
                    cx = sorted(xs)[len(xs) // 2]
                    cy = sorted(ys)[len(ys) // 2]
                    all_cluster_centers.append((cx, cy))

        return list(set(all_cluster_centers)) # Remove duplicados (diferentes 4x4 podem ter o mesmo centro)


    
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
