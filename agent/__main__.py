from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np

import flappy_bird_gymnasium  # noqa: F401
from agent.diy import DIYDecisionAgent
from agent.ppo import PPOInferenceAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play Flappy Bird with DIY or PPO agent."
    )
    parser.add_argument("--agent", choices=("diy", "ppo"), default="diy")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true", help="Render game window.")
    parser.add_argument(
        "--use-lidar",
        action="store_true",
        help="Use lidar observations (DIY agent ignores this and always uses feature observations).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to PPO model file (required when --agent ppo).",
    )
    return parser.parse_args()


def create_agent(args: argparse.Namespace):
    if args.agent == "diy":
        return DIYDecisionAgent()

    if args.model_path is None:
        raise ValueError("PPO agent requires --model-path.")
    return PPOInferenceAgent(model_path=args.model_path)


def run_episode(
    env: gym.Env, policy, seed: Optional[int], max_steps: Optional[int]
) -> tuple[float, int, int]:
    observation, info = env.reset(seed=seed)
    policy.reset()

    total_reward = 0.0
    step_count = 0
    score = int(info.get("score", 0))

    while True:
        action = int(policy.act(observation))
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step_count += 1
        score = int(info.get("score", score))

        if (
            (max_steps is not None and step_count >= max_steps)
            or terminated
            or truncated
        ):
            return total_reward, score, step_count


def main() -> None:
    args = parse_args()
    use_lidar = bool(args.use_lidar) if args.agent == "ppo" else False
    render_mode = "human" if args.render else None

    env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=use_lidar)
    policy = create_agent(args)

    try:
        rewards = []
        scores = []
        for episode in range(args.episodes):
            episode_seed = None if args.seed is None else args.seed + episode
            reward, score, steps = run_episode(
                env, policy, episode_seed, args.max_steps
            )
            rewards.append(reward)
            scores.append(score)
            print(
                f"[{args.agent}] episode={episode + 1}/{args.episodes} "
                f"score={score} reward={reward:.2f} steps={steps}"
            )

        print(
            f"[{args.agent}] avg_score={np.mean(scores):.2f} "
            f"avg_reward={np.mean(rewards):.2f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
