from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import flappy_bird_gymnasium  # noqa: F401
from agent.diy import DIYDecisionAgent
from train.features import extract_diy_features_torch, shaped_reward_from_features
from train.model import ActorCriticCNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_PATH = PROJECT_ROOT / "checkpoints" / "ppo_cnn.pt"


@dataclass
class PPOConfig:
    total_timesteps: int
    rollout_steps: int
    update_epochs: int
    minibatch_size: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    ent_coef: float
    vf_coef: float
    imitation_coef: float
    learning_rate: float
    max_grad_norm: float
    seed: int
    save_path: Path
    device: str


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(
        description="Train PPO (Actor-Critic + CNN) for Flappy Bird with DIY baseline guidance."
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=200_000, help="Total environment steps."
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=1024,
        help="Steps collected before each PPO update.",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=8,
        help="Optimization epochs per PPO update.",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=256,
        help="Mini-batch size for PPO updates.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument(
        "--clip-coef", type=float, default=0.2, help="PPO clip coefficient."
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy regularization coefficient.",
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="Value loss coefficient."
    )
    parser.add_argument(
        "--imitation-coef",
        type=float,
        default=0.15,
        help="DIY baseline imitation loss coefficient (behavior cloning auxiliary loss).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Adam learning rate."
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device", type=str, default="cpu", help="PyTorch device, e.g. cpu or cuda."
    )
    args = parser.parse_args()
    return PPOConfig(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        imitation_coef=args.imitation_coef,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        save_path=SAVE_PATH,
        device=args.device,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(config: PPOConfig) -> None:
    set_seed(config.seed)
    device = torch.device(config.device)

    env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)
    observation, _ = env.reset(seed=config.seed)
    obs_dim = int(np.asarray(observation).shape[0])
    feature_dim = 12

    model = ActorCriticCNN(obs_dim=obs_dim, feature_dim=feature_dim, action_dim=2).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    diy_agent = DIYDecisionAgent()
    diy_agent.reset()

    rollout_steps = config.rollout_steps
    num_updates = max(1, config.total_timesteps // rollout_steps)

    episode_rewards: list[float] = []
    episode_scores: list[int] = []
    running_episode_reward = 0.0

    obs = np.asarray(observation, dtype=np.float32)
    global_step = 0
    for update in range(1, num_updates + 1):
        obs_buf = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
        action_buf = np.zeros((rollout_steps,), dtype=np.int64)
        logprob_buf = np.zeros((rollout_steps,), dtype=np.float32)
        value_buf = np.zeros((rollout_steps,), dtype=np.float32)
        reward_buf = np.zeros((rollout_steps,), dtype=np.float32)
        done_buf = np.zeros((rollout_steps,), dtype=np.float32)
        diy_action_buf = np.zeros((rollout_steps,), dtype=np.int64)

        for t in range(rollout_steps):
            obs_buf[t] = obs
            obs_t = torch.from_numpy(obs).to(device)
            feat_t = extract_diy_features_torch(obs_t).to(device)
            with torch.no_grad():
                logits, value = model(obs_t, feat_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

            action_i = int(action.item())
            diy_action_buf[t] = int(diy_agent.act(obs))
            action_buf[t] = action_i
            logprob_buf[t] = float(logprob.item())
            value_buf[t] = float(value.item())

            next_obs, env_reward, terminated, truncated, info = env.step(action_i)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            done = bool(terminated or truncated)
            shaped_reward = shaped_reward_from_features(float(env_reward), next_obs)

            reward_buf[t] = float(shaped_reward)
            done_buf[t] = 1.0 if done else 0.0
            running_episode_reward += float(env_reward)
            global_step += 1

            if done:
                episode_rewards.append(running_episode_reward)
                episode_scores.append(int(info.get("score", 0)))
                running_episode_reward = 0.0
                next_obs, _ = env.reset()
                next_obs = np.asarray(next_obs, dtype=np.float32)
                diy_agent.reset()

            obs = next_obs

        with torch.no_grad():
            next_obs_t = torch.from_numpy(obs).to(device)
            next_feat_t = extract_diy_features_torch(next_obs_t).to(device)
            _, next_value = model(next_obs_t, next_feat_t)
            next_value_f = float(next_value.item())

        advantages = np.zeros((rollout_steps,), dtype=np.float32)
        last_gae_lam = 0.0
        for t in reversed(range(rollout_steps)):
            if t == rollout_steps - 1:
                next_non_terminal = 1.0 - done_buf[t]
                next_values = next_value_f
            else:
                next_non_terminal = 1.0 - done_buf[t + 1]
                next_values = value_buf[t + 1]
            delta = (
                reward_buf[t]
                + config.gamma * next_values * next_non_terminal
                - value_buf[t]
            )
            last_gae_lam = (
                delta
                + config.gamma * config.gae_lambda * next_non_terminal * last_gae_lam
            )
            advantages[t] = last_gae_lam
        returns = advantages + value_buf

        b_obs = torch.from_numpy(obs_buf).to(device)
        b_actions = torch.from_numpy(action_buf).to(device)
        b_logprobs = torch.from_numpy(logprob_buf).to(device)
        b_returns = torch.from_numpy(returns).to(device)
        b_advantages = torch.from_numpy(advantages).to(device)
        b_diy_actions = torch.from_numpy(diy_action_buf).to(device)

        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )
        b_inds = np.arange(rollout_steps)

        for _ in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, rollout_steps, config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]
                mb_obs = b_obs[mb_inds]
                mb_feat = extract_diy_features_torch(mb_obs)
                mb_actions = b_actions[mb_inds]
                mb_old_logprobs = b_logprobs[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_diy_actions = b_diy_actions[mb_inds]

                logits, values = model(mb_obs, mb_feat)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                logratio = new_logprobs - mb_old_logprobs
                ratio = logratio.exp()
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = F.mse_loss(values, mb_returns)
                imitation_loss = F.cross_entropy(logits, mb_diy_actions)
                loss = (
                    pg_loss
                    + config.vf_coef * value_loss
                    - config.ent_coef * entropy
                    + config.imitation_coef * imitation_loss
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

        mean_reward = float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0
        mean_score = float(np.mean(episode_scores[-10:])) if episode_scores else 0.0
        print(
            f"update={update}/{num_updates} global_step={global_step} "
            f"mean_reward_10={mean_reward:.2f} mean_score_10={mean_score:.2f}"
        )

    config.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": obs_dim,
            "feature_dim": feature_dim,
            "action_dim": 2,
            "algorithm": "ppo_actor_critic_cnn_with_diy_baseline",
            "config": config.__dict__,
        },
        str(config.save_path),
    )
    env.close()
    print(f"saved_model={config.save_path}")


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
