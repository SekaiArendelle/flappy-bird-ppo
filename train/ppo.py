from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.vector import AsyncVectorEnv
from torch.distributions import Categorical

import flappy_bird_gymnasium  # noqa: F401
from agent.diy import DIYDecisionAgent
from train.features import extract_diy_features_torch, shaped_reward_from_features
from train.model import ActorCriticLSTM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LATEST_SAVE_PATH = CHECKPOINT_DIR / "latest.pt"
BEST_SAVE_PATH = CHECKPOINT_DIR / "best.pt"


@dataclass
class PPOConfig:
    rollout_steps: int
    num_envs: int
    update_epochs: int
    minibatch_size: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    ent_coef: float
    vf_coef: float
    imitation_coef: float
    diy_baseline_episodes: int
    imitation_stop_score_gap: float
    learning_rate: float
    max_grad_norm: float
    total_timesteps: int | None
    target_kl: float | None
    seed: int
    device: str


class DeferredLatestCheckpointSaver:
    def __init__(self, save_path: Path) -> None:
        self._save_path = save_path
        self._latest_payload: dict[str, object] | None = None

    def __enter__(self) -> "DeferredLatestCheckpointSaver":
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def stage(self, checkpoint_payload: dict[str, object]) -> None:
        self._latest_payload = checkpoint_payload

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._latest_payload is None:
            return
        torch.save(self._latest_payload, str(self._save_path))


def parse_total_timesteps(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized == "inf":
        return None
    try:
        steps = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--total-timesteps must be a positive integer, or 'inf' for unlimited steps."
        ) from exc
    if steps <= 0:
        raise argparse.ArgumentTypeError(
            "--total-timesteps must be a positive integer, or 'inf' for unlimited steps."
        )
    return steps


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(
        description="Train PPO (Actor-Critic + CNN) for Flappy Bird with DIY baseline guidance."
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=1024,
        help="Steps collected before each PPO update.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of subprocess environments for parallel rollout collection.",
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
        "--diy-baseline-episodes",
        type=int,
        default=8,
        help="Episodes used to estimate DIY baseline performance before training.",
    )
    parser.add_argument(
        "--imitation-stop-score-gap",
        type=float,
        default=0.5,
        help="Stop imitation once mean_score_10 is within this gap from DIY baseline score.",
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
    parser.add_argument(
        "--total-timesteps",
        type=parse_total_timesteps,
        default=5_000_000,
        help="Total environment steps to train. Use 'inf' for unlimited steps.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Target KL divergence threshold. Stop update epochs early if exceeded.",
    )
    args = parser.parse_args()
    return PPOConfig(
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        imitation_coef=args.imitation_coef,
        diy_baseline_episodes=args.diy_baseline_episodes,
        imitation_stop_score_gap=args.imitation_stop_score_gap,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=args.device,
        total_timesteps=args.total_timesteps,
        target_kl=args.target_kl,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_diy_baseline(seed: int, episodes: int) -> tuple[float, float]:
    episode_count = max(1, episodes)
    diy_agent = DIYDecisionAgent()
    episode_rewards: list[float] = []
    episode_scores: list[int] = []
    with gym.make("FlappyBird-v0", render_mode=None, use_lidar=False) as env:
        for episode in range(episode_count):
            observation, info = env.reset(seed=seed + episode)
            diy_agent.reset()
            episode_reward = 0.0
            score = int(info.get("score", 0))

            while True:
                action = int(diy_agent.act(observation))
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                score = int(info.get("score", score))
                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            episode_scores.append(score)

    return float(np.mean(episode_scores)), float(np.mean(episode_rewards))


def make_flappy_env():
    return gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)


def _extract_vector_info_int(
    info: dict[str, np.ndarray], key: str, index: int, default: int = 0
) -> int:
    values = info.get(key)
    if values is None:
        return default

    mask_key = f"_{key}"
    mask = info.get(mask_key)
    if mask is not None and not bool(mask[index]):
        return default
    return int(values[index])


def train(config: PPOConfig) -> None:
    set_seed(config.seed)
    if config.num_envs <= 0:
        raise ValueError("--num-envs must be a positive integer.")
    if config.rollout_steps % config.num_envs != 0:
        raise ValueError("--rollout-steps must be divisible by --num-envs.")

    device = torch.device(config.device)
    latest_save_path = LATEST_SAVE_PATH
    best_save_path = BEST_SAVE_PATH
    update = 0
    rollout_steps_per_env = config.rollout_steps // config.num_envs
    diy_baseline_score, diy_baseline_reward = evaluate_diy_baseline(
        seed=config.seed, episodes=config.diy_baseline_episodes
    )
    imitation_active = config.imitation_coef > 0.0
    print(
        f"diy_baseline episodes={max(1, config.diy_baseline_episodes)} "
        f"mean_reward={diy_baseline_reward:.2f} mean_score={diy_baseline_score:.2f} "
        f"imitation_active={imitation_active}"
    )
    env_fns = [make_flappy_env for _ in range(config.num_envs)]
    envs = AsyncVectorEnv(env_fns)
    try:
        observation, _ = envs.reset(
            seed=[config.seed + i for i in range(config.num_envs)]
        )
        obs = np.asarray(observation, dtype=np.float32)
        obs_dim = int(obs.shape[1])
        feature_dim = 12
        model = ActorCriticLSTM(
            obs_dim=obs_dim, feature_dim=feature_dim, action_dim=2
        ).to(device)
        hidden_size = model.hidden_size
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        diy_agents = [DIYDecisionAgent() for _ in range(config.num_envs)]
        for diy_agent in diy_agents:
            diy_agent.reset()
        hidden_h = np.zeros((config.num_envs, hidden_size), dtype=np.float32)
        hidden_c = np.zeros((config.num_envs, hidden_size), dtype=np.float32)
        episode_rewards: list[float] = []
        episode_scores: list[int] = []
        running_episode_rewards = np.zeros((config.num_envs,), dtype=np.float32)
        global_step = 0
        best_mean_score = float("-inf")
        best_mean_reward = float("-inf")

        print(
            f"collector=async_vector_env num_envs={config.num_envs} "
            f"rollout_steps_per_env={rollout_steps_per_env} batch_size={config.rollout_steps}"
        )

        with DeferredLatestCheckpointSaver(latest_save_path) as latest_checkpoint_saver:
            while True:
                update += 1
                if config.total_timesteps is not None:
                    progress = min(1.0, global_step / config.total_timesteps)
                    lr_now = config.learning_rate * (1.0 - progress)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_now
                obs_buf = np.zeros(
                    (rollout_steps_per_env, config.num_envs, obs_dim), dtype=np.float32
                )
                action_buf = np.zeros(
                    (rollout_steps_per_env, config.num_envs), dtype=np.int64
                )
                logprob_buf = np.zeros(
                    (rollout_steps_per_env, config.num_envs), dtype=np.float32
                )
                value_buf = np.zeros(
                    (rollout_steps_per_env, config.num_envs), dtype=np.float32
                )
                reward_buf = np.zeros(
                    (rollout_steps_per_env, config.num_envs), dtype=np.float32
                )
                done_buf = np.zeros(
                    (rollout_steps_per_env, config.num_envs), dtype=np.float32
                )
                diy_action_buf = np.zeros(
                    (rollout_steps_per_env, config.num_envs), dtype=np.int64
                )
                hidden_h_buf = np.zeros(
                    (rollout_steps_per_env, config.num_envs, hidden_size),
                    dtype=np.float32,
                )
                hidden_c_buf = np.zeros(
                    (rollout_steps_per_env, config.num_envs, hidden_size),
                    dtype=np.float32,
                )

                for t in range(rollout_steps_per_env):
                    obs_buf[t] = obs
                    hidden_h_buf[t] = hidden_h
                    hidden_c_buf[t] = hidden_c
                    obs_t = torch.from_numpy(obs).to(device)
                    feat_t = extract_diy_features_torch(obs_t).to(device)
                    h_t = torch.from_numpy(hidden_h).to(device)
                    c_t = torch.from_numpy(hidden_c).to(device)
                    with torch.no_grad():
                        logits, value, (new_h, new_c) = model(
                            obs_t, feat_t, (h_t.unsqueeze(0), c_t.unsqueeze(0))
                        )
                        hidden_h = new_h.squeeze(0).cpu().numpy()
                        hidden_c = new_c.squeeze(0).cpu().numpy()
                        dist = Categorical(logits=logits)
                        action = dist.sample()
                        logprob = dist.log_prob(action)

                    action_np = action.cpu().numpy().astype(np.int64)
                    diy_action_buf[t] = np.asarray(
                        [agent.act(obs[i]) for i, agent in enumerate(diy_agents)],
                        dtype=np.int64,
                    )
                    action_buf[t] = action_np
                    logprob_buf[t] = logprob.cpu().numpy().astype(np.float32)
                    value_buf[t] = value.cpu().numpy().astype(np.float32)

                    next_obs, env_reward, terminated, truncated, info = envs.step(
                        action_np
                    )
                    next_obs = np.asarray(next_obs, dtype=np.float32)
                    env_reward = np.asarray(env_reward, dtype=np.float32)
                    done = np.logical_or(terminated, truncated)
                    shaped_reward = np.asarray(
                        [
                            shaped_reward_from_features(
                                float(env_reward[i]), next_obs[i]
                            )
                            for i in range(config.num_envs)
                        ],
                        dtype=np.float32,
                    )

                    reward_buf[t] = shaped_reward
                    done_buf[t] = done.astype(np.float32)
                    running_episode_rewards += env_reward
                    global_step += config.num_envs

                    done_indices = np.flatnonzero(done)
                    for env_index in done_indices:
                        episode_rewards.append(
                            float(running_episode_rewards[env_index])
                        )
                        episode_scores.append(
                            _extract_vector_info_int(
                                info, "score", int(env_index), default=0
                            )
                        )
                        running_episode_rewards[env_index] = 0.0
                        hidden_h[env_index] = 0.0
                        hidden_c[env_index] = 0.0
                        diy_agents[int(env_index)].reset()

                    obs = next_obs

                with torch.no_grad():
                    next_obs_t = torch.from_numpy(obs).to(device)
                    next_feat_t = extract_diy_features_torch(next_obs_t).to(device)
                    _, next_value, _ = model(next_obs_t, next_feat_t)
                    next_value_np = next_value.cpu().numpy().astype(np.float32)

                advantages = np.zeros(
                    (rollout_steps_per_env, config.num_envs), dtype=np.float32
                )
                last_gae_lam = np.zeros((config.num_envs,), dtype=np.float32)
                for t in reversed(range(rollout_steps_per_env)):
                    if t == rollout_steps_per_env - 1:
                        next_non_terminal = 1.0 - done_buf[t]
                        next_values = next_value_np
                    else:
                        next_non_terminal = 1.0 - done_buf[t]
                        next_values = value_buf[t + 1]
                    delta = (
                        reward_buf[t]
                        + config.gamma * next_values * next_non_terminal
                        - value_buf[t]
                    )
                    last_gae_lam = (
                        delta
                        + config.gamma
                        * config.gae_lambda
                        * next_non_terminal
                        * last_gae_lam
                    )
                    advantages[t] = last_gae_lam
                returns = advantages + value_buf

                batch_size = rollout_steps_per_env * config.num_envs
                b_obs = torch.from_numpy(obs_buf.reshape(batch_size, obs_dim)).to(
                    device
                )
                b_actions = torch.from_numpy(action_buf.reshape(batch_size)).to(device)
                b_logprobs = torch.from_numpy(logprob_buf.reshape(batch_size)).to(
                    device
                )
                b_returns = torch.from_numpy(returns.reshape(batch_size)).to(device)
                b_advantages = torch.from_numpy(advantages.reshape(batch_size)).to(
                    device
                )
                b_hidden_h = torch.from_numpy(
                    hidden_h_buf.reshape(batch_size, hidden_size)
                ).to(device)
                b_hidden_c = torch.from_numpy(
                    hidden_c_buf.reshape(batch_size, hidden_size)
                ).to(device)
                imitation_coef = config.imitation_coef if imitation_active else 0.0
                if imitation_coef > 0.0:
                    b_diy_actions = torch.from_numpy(
                        diy_action_buf.reshape(batch_size)
                    ).to(device)

                b_advantages = (b_advantages - b_advantages.mean()) / (
                    b_advantages.std() + 1e-8
                )
                b_inds = np.arange(batch_size)

                approx_kl = 0.0
                early_stop = False
                for epoch in range(config.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, batch_size, config.minibatch_size):
                        end = start + config.minibatch_size
                        mb_inds = b_inds[start:end]
                        mb_obs = b_obs[mb_inds]
                        mb_feat = extract_diy_features_torch(mb_obs)
                        mb_actions = b_actions[mb_inds]
                        mb_old_logprobs = b_logprobs[mb_inds]
                        mb_returns = b_returns[mb_inds]
                        mb_advantages = b_advantages[mb_inds]
                        if imitation_coef > 0.0:
                            mb_diy_actions = b_diy_actions[mb_inds]

                        mb_hidden_h = b_hidden_h[mb_inds]
                        mb_hidden_c = b_hidden_c[mb_inds]
                        mb_hidden = (
                            mb_hidden_h.unsqueeze(0),
                            mb_hidden_c.unsqueeze(0),
                        )
                        logits, values, _ = model(mb_obs, mb_feat, mb_hidden)
                        dist = Categorical(logits=logits)
                        new_logprobs = dist.log_prob(mb_actions)
                        entropy = dist.entropy().mean()

                        logratio = new_logprobs - mb_old_logprobs
                        ratio = logratio.exp()

                        if config.target_kl is not None:
                            with torch.no_grad():
                                approx_kl = ((ratio - 1) - logratio).mean()
                            if approx_kl > config.target_kl:
                                early_stop = True
                                break

                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(
                            ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
                        )
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        value_loss = F.mse_loss(values, mb_returns)
                        loss = (
                            pg_loss
                            + config.vf_coef * value_loss
                            - config.ent_coef * entropy
                        )
                        if imitation_coef > 0.0:
                            imitation_loss = F.cross_entropy(logits, mb_diy_actions)
                            loss = loss + imitation_coef * imitation_loss

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.max_grad_norm
                        )
                        optimizer.step()

                    if early_stop:
                        print(
                            f"early_stop update={update} epoch={epoch} "
                            f"reason=kl_divergence approx_kl={approx_kl:.4f} "
                            f"target_kl={config.target_kl}"
                        )
                        break

                mean_reward = (
                    float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0
                )
                mean_score = (
                    float(np.mean(episode_scores[-10:])) if episode_scores else 0.0
                )
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"update={update} global_step={global_step} "
                    f"mean_reward_10={mean_reward:.2f} mean_score_10={mean_score:.2f} "
                    f"lr={current_lr:.2e} imitation_coef={imitation_coef:.4f}"
                )
                if imitation_active and (
                    mean_score >= diy_baseline_score - config.imitation_stop_score_gap
                ):
                    imitation_active = False
                    print(
                        f"imitation_disabled update={update} mean_score_10={mean_score:.2f} "
                        f"diy_mean_score={diy_baseline_score:.2f} "
                        f"stop_gap={config.imitation_stop_score_gap:.2f}"
                    )

                checkpoint_payload = {
                    "model_state_dict": model.state_dict(),
                    "obs_dim": obs_dim,
                    "feature_dim": feature_dim,
                    "action_dim": 2,
                    "algorithm": "ppo_actor_critic_cnn_with_diy_baseline",
                    "config": config.__dict__,
                    "update": update,
                    "global_step": global_step,
                    "mean_reward_10": mean_reward,
                    "mean_score_10": mean_score,
                    "diy_baseline_mean_reward": diy_baseline_reward,
                    "diy_baseline_mean_score": diy_baseline_score,
                    "imitation_active": imitation_active,
                }
                latest_checkpoint_saver.stage(checkpoint_payload)

                is_better = (mean_score > best_mean_score) or (
                    mean_score == best_mean_score and mean_reward > best_mean_reward
                )
                if is_better:
                    best_mean_score = mean_score
                    best_mean_reward = mean_reward
                    torch.save(checkpoint_payload, str(best_save_path))
                    print(
                        f"best_mean_reward_10={best_mean_reward:.2f} "
                        f"best_mean_score_10={best_mean_score:.2f}"
                    )

                if (
                    config.total_timesteps is not None
                    and global_step >= config.total_timesteps
                ):
                    break
    finally:
        envs.close()

    print(
        f"training_done latest_model={latest_save_path} " f"best_model={best_save_path}"
    )


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
