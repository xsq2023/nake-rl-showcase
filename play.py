from __future__ import annotations

import argparse
import json
from pathlib import Path
import tkinter as tk
from typing import Dict, List, Tuple

from agent import HybridSnakeAgent
from snake_env import SnakeEnv

State = Tuple[int, ...]
QTable = Dict[State, List[float]]


def load_q_table(path: Path) -> QTable:
    payload = json.loads(path.read_text(encoding="utf-8"))
    q_table: QTable = {}
    for key, value in payload.items():
        state = tuple(int(x) for x in key.split("|"))
        q_table[state] = [float(x) for x in value]
    return q_table


class SnakeApp:
    def __init__(
        self,
        env: SnakeEnv,
        q_table: QTable,
        policy_mode: str = "hybrid",
        cell: int = 30,
        speed_ms: int = 70,
    ):
        self.env = env
        self.q_table = q_table
        self.agent = HybridSnakeAgent(q_table)
        self.cell = cell
        self.speed_ms = speed_ms
        self.policy_mode = policy_mode

        self.root = tk.Tk()
        self.root.title("Snake RL - Showcase Edition")
        w = env.width * cell
        h = env.height * cell + 70
        self.canvas = tk.Canvas(self.root, width=w, height=h, bg="#0b0f12", highlightthickness=0)
        self.canvas.pack()

        self.state = self.env.reset()
        self.running = True
        self.best_score = 0
        self.episode = 1

        self.root.bind("<space>", self.toggle_pause)
        self.root.bind("r", self.restart)
        self.root.bind("m", self.toggle_mode)
        self.root.bind("+", self.speed_up)
        self.root.bind("=", self.speed_up)
        self.root.bind("-", self.speed_down)

    def toggle_pause(self, _event=None):
        self.running = not self.running

    def restart(self, _event=None):
        self.state = self.env.reset()
        self.running = True

    def toggle_mode(self, _event=None):
        self.policy_mode = "rl" if self.policy_mode == "hybrid" else "hybrid"

    def speed_up(self, _event=None):
        self.speed_ms = max(15, self.speed_ms - 10)

    def speed_down(self, _event=None):
        self.speed_ms = min(300, self.speed_ms + 10)

    def _pick_action(self) -> int:
        if self.policy_mode == "rl":
            return self.agent.choose_action_rl_only(self.state)
        return self.agent.choose_action(self.env, self.state)

    def draw(self, reason: str = ""):
        self.canvas.delete("all")
        c = self.cell
        gw = self.env.width * c
        gh = self.env.height * c

        self.canvas.create_rectangle(0, 0, gw, gh, fill="#12191f", outline="#293744", width=2)

        # subtle grid
        for x in range(0, gw, c):
            self.canvas.create_line(x, 0, x, gh, fill="#18222c")
        for y in range(0, gh, c):
            self.canvas.create_line(0, y, gw, y, fill="#18222c")

        fx, fy = self.env.food
        self.canvas.create_oval(
            fx * c + 5,
            fy * c + 5,
            (fx + 1) * c - 5,
            (fy + 1) * c - 5,
            fill="#ff6b57",
            outline="#ffac9d",
            width=2,
        )

        for i, (x, y) in enumerate(self.env.snake):
            if i == 0:
                fill, outline = "#5ff0bf", "#c8ffe9"
            else:
                fill, outline = "#2bbf8f", "#71e8c3"
            self.canvas.create_rectangle(
                x * c + 3,
                y * c + 3,
                (x + 1) * c - 3,
                (y + 1) * c - 3,
                fill=fill,
                outline=outline,
                width=1,
            )

        mode_label = "HYBRID" if self.policy_mode == "hybrid" else "RL"
        status = "RUN" if self.running else "PAUSE"

        hud1 = (
            f"score {self.env.score}   best {self.best_score}   len {len(self.env.snake)}   "
            f"episode {self.episode}   mode {mode_label}   tick {self.speed_ms}ms"
        )
        hud2 = "controls: Space pause | R restart | M mode | +/- speed"
        if reason:
            hud2 += f"   status: {reason}"
        self.canvas.create_text(10, gh + 18, anchor="w", text=hud1, fill="#d7e3ee", font=("Menlo", 12, "bold"))
        self.canvas.create_text(10, gh + 44, anchor="w", text=hud2, fill="#9db3c7", font=("Menlo", 11))
        self.canvas.create_text(gw - 10, gh + 18, anchor="e", text=status, fill="#8cc3ff", font=("Menlo", 12, "bold"))

    def tick(self):
        reason = ""
        if self.running:
            action = self._pick_action()
            self.state, _reward, done, info = self.env.step(action)
            reason = info.get("reason", "")
            if done:
                self.best_score = max(self.best_score, info.get("score", 0))
                self.episode += 1
                self.state = self.env.reset()

        self.draw(reason=reason)
        self.root.after(self.speed_ms, self.tick)

    def run(self):
        self.tick()
        self.root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Play snake using trained Q-table")
    p.add_argument("--model", type=str, default="checkpoints/q_table.json")
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--height", type=int, default=12)
    p.add_argument("--cell", type=int, default=30)
    p.add_argument("--speed-ms", type=int, default=70)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--policy", choices=["hybrid", "rl"], default="hybrid")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}. train first.")

    env = SnakeEnv(width=args.width, height=args.height, seed=args.seed)
    q_table = load_q_table(model_path)
    app = SnakeApp(env, q_table, policy_mode=args.policy, cell=args.cell, speed_ms=args.speed_ms)
    app.run()
