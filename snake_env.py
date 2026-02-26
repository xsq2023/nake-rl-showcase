"""Snake environment for tabular reinforcement learning.

State representation (11 binary features):
1) danger straight
2) danger right
3) danger left
4) moving left
5) moving right
6) moving up
7) moving down
8) food left
9) food right
10) food up
11) food down

Action representation: relative turn
- 0: go straight
- 1: turn right
- 2: turn left
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Tuple

Point = Tuple[int, int]


@dataclass(frozen=True)
class Direction:
    x: int
    y: int


UP = Direction(0, -1)
RIGHT = Direction(1, 0)
DOWN = Direction(0, 1)
LEFT = Direction(-1, 0)
CLOCKWISE = [RIGHT, DOWN, LEFT, UP]


class SnakeEnv:
    def __init__(self, width: int = 12, height: int = 12, seed: int | None = None):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.snake: List[Point] = []
        self.direction = RIGHT
        self.food: Point = (0, 0)
        self.score = 0
        self.frame_count = 0
        self.max_frames_without_food = self.width * self.height * 2
        self.reset()

    def reset(self) -> Tuple[int, ...]:
        cx = self.width // 2
        cy = self.height // 2
        self.direction = RIGHT
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.score = 0
        self.frame_count = 0
        self._place_food()
        return self.get_state()

    def _place_food(self) -> None:
        occupied = set(self.snake)
        candidates = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied
        ]
        if not candidates:
            self.food = (-1, -1)
            return
        self.food = self.rng.choice(candidates)

    def _is_collision(self, pt: Point) -> bool:
        x, y = pt
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _next_direction(self, action: int) -> Direction:
        idx = CLOCKWISE.index(self.direction)
        if action == 0:
            return CLOCKWISE[idx]
        if action == 1:
            return CLOCKWISE[(idx + 1) % 4]
        if action == 2:
            return CLOCKWISE[(idx - 1) % 4]
        raise ValueError(f"invalid action: {action}")

    def _next_head(self, direction: Direction) -> Point:
        hx, hy = self.snake[0]
        return (hx + direction.x, hy + direction.y)

    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool, dict]:
        prev_head = self.snake[0]
        prev_food_dist = abs(self.food[0] - prev_head[0]) + abs(self.food[1] - prev_head[1])
        self.frame_count += 1
        self.direction = self._next_direction(action)
        new_head = self._next_head(self.direction)

        if self._is_collision(new_head):
            return self.get_state(), -12.0, True, {"score": self.score, "reason": "collision"}

        ate_food = new_head == self.food
        self.snake.insert(0, new_head)

        reward = -0.03
        done = False

        if ate_food:
            self.score += 1
            reward = 15.0
            self._place_food()
            self.frame_count = 0
        else:
            self.snake.pop()
            new_food_dist = abs(self.food[0] - new_head[0]) + abs(self.food[1] - new_head[1])
            if new_food_dist < prev_food_dist:
                reward += 0.2
            elif new_food_dist > prev_food_dist:
                reward -= 0.2

        if self.score >= self.width * self.height - 3:
            done = True
            reward = 50.0
            return self.get_state(), reward, done, {"score": self.score, "reason": "win"}

        if self.frame_count > self.max_frames_without_food:
            done = True
            reward = -5.0
            return self.get_state(), reward, done, {"score": self.score, "reason": "stuck"}

        return self.get_state(), reward, done, {"score": self.score, "reason": "running"}

    def get_state(self) -> Tuple[int, ...]:
        head = self.snake[0]

        dir_l = self.direction == LEFT
        dir_r = self.direction == RIGHT
        dir_u = self.direction == UP
        dir_d = self.direction == DOWN

        pt_straight = self._next_head(self.direction)
        right_dir = self._next_direction(1)
        left_dir = self._next_direction(2)
        pt_right = self._next_head(right_dir)
        pt_left = self._next_head(left_dir)

        food_left = int(self.food[0] < head[0])
        food_right = int(self.food[0] > head[0])
        food_up = int(self.food[1] < head[1])
        food_down = int(self.food[1] > head[1])

        state = (
            int(self._is_collision(pt_straight)),
            int(self._is_collision(pt_right)),
            int(self._is_collision(pt_left)),
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),
            food_left,
            food_right,
            food_up,
            food_down,
        )
        return state
