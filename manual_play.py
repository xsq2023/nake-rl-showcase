from __future__ import annotations

import argparse
import tkinter as tk

from snake_env import SnakeEnv, UP, DOWN, LEFT, RIGHT, CLOCKWISE


class ManualSnakeApp:
    def __init__(self, env: SnakeEnv, cell: int = 26, speed_ms: int = 120):
        self.env = env
        self.cell = cell
        self.speed_ms = speed_ms

        self.root = tk.Tk()
        self.root.title("Snake RL - Manual Play")
        w = env.width * cell
        h = env.height * cell + 40
        self.canvas = tk.Canvas(self.root, width=w, height=h, bg="#111")
        self.canvas.pack()

        self.pending_action = 0
        self.state = self.env.reset()

        self.root.bind("<Up>", lambda _e: self._set_direction(UP))
        self.root.bind("<Down>", lambda _e: self._set_direction(DOWN))
        self.root.bind("<Left>", lambda _e: self._set_direction(LEFT))
        self.root.bind("<Right>", lambda _e: self._set_direction(RIGHT))
        self.root.bind("r", self.restart)

    def _relative_action(self, target_dir) -> int:
        idx = CLOCKWISE.index(self.env.direction)
        if target_dir == CLOCKWISE[idx]:
            return 0
        if target_dir == CLOCKWISE[(idx + 1) % 4]:
            return 1
        if target_dir == CLOCKWISE[(idx - 1) % 4]:
            return 2
        return 0

    def _set_direction(self, target_dir):
        self.pending_action = self._relative_action(target_dir)

    def restart(self, _event=None):
        self.state = self.env.reset()
        self.pending_action = 0

    def draw(self, reason: str = ""):
        self.canvas.delete("all")
        c = self.cell

        fx, fy = self.env.food
        self.canvas.create_oval(fx * c + 4, fy * c + 4, (fx + 1) * c - 4, (fy + 1) * c - 4, fill="#ff5d5d", outline="")

        for i, (x, y) in enumerate(self.env.snake):
            color = "#7fb3ff" if i == 0 else "#5087d8"
            self.canvas.create_rectangle(x * c + 2, y * c + 2, (x + 1) * c - 2, (y + 1) * c - 2, fill=color, outline="")

        msg = f"score={self.env.score}  len={len(self.env.snake)}"
        if reason:
            msg += f"  {reason}"
        self.canvas.create_text(8, self.env.height * c + 20, anchor="w", text=msg, fill="#ddd", font=("Menlo", 12))

    def tick(self):
        self.state, _reward, done, info = self.env.step(self.pending_action)
        reason = info.get("reason", "")
        if done:
            reason = f"game over: {reason} (press r)"
        self.draw(reason)
        self.root.after(self.speed_ms, self.tick)

    def run(self):
        self.tick()
        self.root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Manual snake game")
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--height", type=int, default=12)
    p.add_argument("--cell", type=int, default=26)
    p.add_argument("--speed-ms", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    env = SnakeEnv(width=args.width, height=args.height, seed=args.seed)
    app = ManualSnakeApp(env, cell=args.cell, speed_ms=args.speed_ms)
    app.run()
