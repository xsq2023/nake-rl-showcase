from __future__ import annotations

from collections import deque
from typing import Dict, List, Set, Tuple

from snake_env import CLOCKWISE, SnakeEnv

State = Tuple[int, ...]
QTable = Dict[State, List[float]]
Point = Tuple[int, int]


class HybridSnakeAgent:
    """RL + planning hybrid policy for stronger and safer gameplay."""

    def __init__(self, q_table: QTable):
        self.q_table = q_table

    def choose_action(self, env: SnakeEnv, state: State) -> int:
        safe_actions = self._safe_actions(env)
        if not safe_actions:
            return 0

        q_values = self.q_table.get(state, [0.0, 0.0, 0.0])
        base_dist = self._manhattan(env.snake[0], env.food)

        best_action = safe_actions[0]
        best_score = float("-inf")

        for action in safe_actions:
            next_head, next_body, ate_food = self._simulate_step(env, action)
            blocked = set(next_body[1:])

            dist_to_food = self._shortest_path_dist(next_head, env.food, blocked, env.width, env.height)
            free_space = self._flood_fill_size(next_head, blocked, env.width, env.height)

            # Trap check: if reachable area is too small, this move is risky.
            trap_penalty = 0.0
            if free_space < len(next_body) + 2:
                trap_penalty = 6.0

            progress = 0.0
            if dist_to_food is not None:
                progress = float(base_dist - dist_to_food)
            elif not ate_food:
                progress = -2.0

            score = (
                q_values[action] * 0.7
                + progress * 1.8
                + free_space * 0.03
                + (6.0 if ate_food else 0.0)
                - trap_penalty
            )

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def choose_action_rl_only(self, state: State) -> int:
        q_values = self.q_table.get(state, [0.0, 0.0, 0.0])
        max_q = max(q_values)
        for i, q in enumerate(q_values):
            if q == max_q:
                return i
        return 0

    def _safe_actions(self, env: SnakeEnv) -> List[int]:
        actions: List[int] = []
        for action in (0, 1, 2):
            next_head, next_body, _ate = self._simulate_step(env, action)
            if self._in_bounds(next_head, env.width, env.height) and next_head not in set(next_body[1:]):
                actions.append(action)
        return actions

    def _simulate_step(self, env: SnakeEnv, action: int) -> Tuple[Point, List[Point], bool]:
        idx = CLOCKWISE.index(env.direction)
        if action == 0:
            next_dir = CLOCKWISE[idx]
        elif action == 1:
            next_dir = CLOCKWISE[(idx + 1) % 4]
        else:
            next_dir = CLOCKWISE[(idx - 1) % 4]

        hx, hy = env.snake[0]
        next_head = (hx + next_dir.x, hy + next_dir.y)
        ate_food = next_head == env.food

        if ate_food:
            next_body = [next_head] + list(env.snake)
        else:
            next_body = [next_head] + list(env.snake[:-1])

        return next_head, next_body, ate_food

    def _shortest_path_dist(
        self,
        start: Point,
        target: Point,
        blocked: Set[Point],
        width: int,
        height: int,
    ) -> int | None:
        if start == target:
            return 0
        if target in blocked:
            return None

        q = deque([(start, 0)])
        visited = {start}

        while q:
            (x, y), d = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                np = (nx, ny)
                if not self._in_bounds(np, width, height):
                    continue
                if np in blocked or np in visited:
                    continue
                if np == target:
                    return d + 1
                visited.add(np)
                q.append((np, d + 1))

        return None

    def _flood_fill_size(self, start: Point, blocked: Set[Point], width: int, height: int) -> int:
        if start in blocked or not self._in_bounds(start, width, height):
            return 0

        q = deque([start])
        visited = {start}

        while q:
            x, y = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                np = (nx, ny)
                if not self._in_bounds(np, width, height):
                    continue
                if np in blocked or np in visited:
                    continue
                visited.add(np)
                q.append(np)

        return len(visited)

    def _in_bounds(self, p: Point, width: int, height: int) -> bool:
        return 0 <= p[0] < width and 0 <= p[1] < height

    def _manhattan(self, a: Point, b: Point) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
