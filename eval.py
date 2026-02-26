from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
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


def evaluate(args: argparse.Namespace) -> None:
    q = load_q_table(Path(args.model))
    env = SnakeEnv(width=args.width, height=args.height, seed=args.seed)
    agent = HybridSnakeAgent(q)

    scores: List[int] = []

    for _ in range(args.episodes):
        state = env.reset()
        done = False
        info = {"score": 0}
        while not done:
            if args.policy == "rl":
                action = agent.choose_action_rl_only(state)
            else:
                action = agent.choose_action(env, state)
            state, _reward, done, info = env.step(action)
        scores.append(info["score"])

    sorted_scores = sorted(scores)
    p50 = sorted_scores[len(scores) // 2]
    p90 = sorted_scores[int(len(scores) * 0.9)]

    print(f"episodes={args.episodes}")
    print(f"policy={args.policy}")
    print(f"avg_score={mean(scores):.3f}")
    print(f"best_score={max(scores)}")
    print(f"min_score={min(scores)}")
    print(f"p50_score={p50}")
    print(f"p90_score={p90}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate trained snake policy without GUI")
    p.add_argument("--model", type=str, default="checkpoints/q_table.json")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--height", type=int, default=12)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--policy", choices=["hybrid", "rl"], default="hybrid")
    return p


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
