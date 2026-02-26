from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from statistics import mean
from typing import Dict, List, Tuple

from snake_env import SnakeEnv

State = Tuple[int, ...]
QTable = Dict[State, List[float]]


def get_q(q_table: QTable, state: State) -> List[float]:
    if state not in q_table:
        q_table[state] = [0.0, 0.0, 0.0]
    return q_table[state]


def choose_action(q_table: QTable, state: State, epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randint(0, 2)
    q_values = get_q(q_table, state)
    max_q = max(q_values)
    best = [i for i, q in enumerate(q_values) if q == max_q]
    return rng.choice(best)


def update_q(
    q_table: QTable,
    state: State,
    action: int,
    reward: float,
    next_state: State,
    alpha: float,
    gamma: float,
) -> None:
    q_values = get_q(q_table, state)
    next_q_values = get_q(q_table, next_state)
    target = reward + gamma * max(next_q_values)
    q_values[action] += alpha * (target - q_values[action])


def save_q_table(q_table: QTable, path: Path) -> None:
    payload = {"|".join(map(str, k)): v for k, v in q_table.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_q_table(path: Path) -> QTable:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: QTable = {}
    for key, value in payload.items():
        state = tuple(int(x) for x in key.split("|"))
        result[state] = [float(x) for x in value]
    return result


def train(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    env = SnakeEnv(width=args.width, height=args.height, seed=args.seed)
    q_path = Path(args.output)

    q_table = load_q_table(q_path) if args.resume else {}

    epsilon = args.epsilon_start
    epsilon_decay = (args.epsilon_start - args.epsilon_end) / max(1, args.episodes)

    scores: List[int] = []

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done = False

        while not done:
            action = choose_action(q_table, state, epsilon, rng)
            next_state, reward, done, info = env.step(action)
            update_q(q_table, state, action, reward, next_state, args.alpha, args.gamma)
            state = next_state

        scores.append(info["score"])
        epsilon = max(args.epsilon_end, epsilon - epsilon_decay)

        if episode % args.log_every == 0 or episode == 1:
            window = scores[-args.log_every :]
            avg_score = mean(window)
            best = max(scores)
            print(
                f"episode={episode:5d} epsilon={epsilon:.3f} "
                f"score={scores[-1]:2d} avg={avg_score:.2f} best={best:2d} states={len(q_table)}"
            )

    q_path.parent.mkdir(parents=True, exist_ok=True)
    save_q_table(q_table, q_path)
    print(f"saved q-table: {q_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Snake with tabular Q-learning")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="checkpoints/q_table.json")
    parser.add_argument("--resume", action="store_true", help="continue training from existing q-table")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    train(parser.parse_args())
