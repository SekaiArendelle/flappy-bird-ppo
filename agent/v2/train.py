"""
Ray Distributed PPO for Flappy Bird
"""

import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
import os

# ==================== 配置 ====================


@dataclass
class PPOConfig:
    # 环境
    env_name: str = "FlappyBird-v0"
    num_workers: int = 4  # Ray worker 数量
    num_envs_per_worker: int = 4  # 每个worker的并行环境数

    # 观察空间设置
    use_lidar: bool = False  # False=12维状态, True=180维LIDAR
    state_dim: int = 12  # 根据use_lidar自动调整: 12或180
    action_dim: int = 2  # [不动, 拍翅膀]
    hidden_dim: int = 256

    # PPO参数
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # 训练
    steps_per_update: int = 2048  # 每个worker收集的步数
    num_epochs: int = 10  # 每次数据训练轮数
    batch_size: int = 64
    total_timesteps: int = 10_000_000

    # 设备
    use_cuda: bool = True  # Parameter Server是否使用GPU


# ==================== 神经网络 ====================


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # 根据输入维度调整网络结构
        if state_dim == 180:  # LIDAR模式使用更大的网络
            feature_dim = hidden_dim * 2
            self.feature = nn.Sequential(
                nn.Linear(state_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(),
            )
        else:
            feature_dim = hidden_dim
            self.feature = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )

        # Actor: 策略网络
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic: 价值网络
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action_and_value(
        self, state: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(state)
        return value.squeeze(-1)


# ==================== 经验缓冲区 ====================


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def get(self) -> Dict[str, torch.Tensor]:
        return {
            "states": torch.stack(self.states),
            "actions": torch.stack(self.actions),
            "log_probs": torch.stack(self.log_probs),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32),
            "values": torch.tensor(self.values, dtype=torch.float32),
            "dones": torch.tensor(self.dones, dtype=torch.float32),
        }


# ==================== Ray Worker: 环境交互与数据收集 ====================


@ray.remote(num_cpus=2)  # 每个Worker分配2个CPU核心
class RolloutWorker:
    """
    分布式Worker: 负责环境交互、数据收集
    关键修复：在远程进程中导入flappy_bird_gymnasium以注册环境
    """

    def __init__(self, worker_id: int, config: Dict):
        # 必须在远程进程中导入以注册环境！
        import flappy_bird_gymnasium
        import gymnasium as gym

        self.worker_id = worker_id
        self.config = PPOConfig(**config)
        self.device = torch.device("cpu")  # Worker使用CPU

        # 根据配置设置状态维度
        if self.config.use_lidar:
            self.config.state_dim = 180
        else:
            self.config.state_dim = 12

        # 创建向量化环境
        self.envs = self._create_envs(gym)
        self.states, _ = self.envs.reset()

        # 本地模型（定期从Parameter Server同步）
        self.model = ActorCritic(
            self.config.state_dim, self.config.action_dim, self.config.hidden_dim
        ).to(self.device)

        self.buffer = RolloutBuffer()
        self.episode_rewards = deque(maxlen=100)
        self.current_episode_rewards = np.zeros(self.config.num_envs_per_worker)

    def _create_envs(self, gym):
        """创建向量化环境"""

        def make_env():
            def _init():
                # 使用use_lidar参数创建环境
                env = gym.make(self.config.env_name, use_lidar=self.config.use_lidar)
                return env

            return _init

        # 使用SyncVectorEnv进行向量化
        env_fns = [make_env() for _ in range(self.config.num_envs_per_worker)]
        return gym.vector.SyncVectorEnv(env_fns)

    def set_weights(self, weights: Dict[str, np.ndarray]):
        """从Parameter Server同步参数"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(torch.from_numpy(weights[name]))

    def collect_rollout(self, weights: Optional[Dict[str, np.ndarray]] = None) -> Dict:
        """
        收集一个完整的rollout数据
        """
        if weights is not None:
            self.set_weights(weights)

        self.buffer.clear()
        step_count = 0

        while step_count < self.config.steps_per_update:
            # 转换为tensor
            state_tensor = torch.from_numpy(self.states).float().to(self.device)

            # 获取动作
            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action_and_value(
                    state_tensor
                )

            # 执行动作
            actions_np = action.cpu().numpy()
            next_states, rewards, terminated, truncated, infos = self.envs.step(
                actions_np
            )
            dones = np.logical_or(terminated, truncated)

            # 存储transition
            for i in range(self.config.num_envs_per_worker):
                self.buffer.add(
                    state_tensor[i],
                    action[i],
                    log_prob[i],
                    rewards[i],
                    value[i],
                    dones[i],
                )
                self.current_episode_rewards[i] += rewards[i]

                if dones[i]:
                    self.episode_rewards.append(self.current_episode_rewards[i])
                    self.current_episode_rewards[i] = 0

            self.states = next_states
            step_count += self.config.num_envs_per_worker

        # 计算returns和advantages (GAE)
        data = self._compute_gae()

        return {
            "data": data,
            "stats": {
                "mean_reward": (
                    np.mean(self.episode_rewards) if self.episode_rewards else 0
                ),
                "max_reward": (
                    np.max(self.episode_rewards) if self.episode_rewards else 0
                ),
                "steps": step_count,
            },
        }

    def _compute_gae(self) -> Dict[str, torch.Tensor]:
        """计算GAE优势和回报"""
        data = self.buffer.get()

        with torch.no_grad():
            next_state_tensor = torch.from_numpy(self.states).float().to(self.device)
            next_values = self.model.get_value(next_state_tensor).cpu().numpy()

        rewards = data["rewards"].numpy()
        values = data["values"].numpy()
        dones = data["dones"].numpy()

        advantages = np.zeros_like(rewards)
        last_gae = 0

        # 反向计算GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_values.mean()  # 简化处理
            else:
                next_val = values[t + 1]

            delta = (
                rewards[t] + self.config.gamma * next_val * (1 - dones[t]) - values[t]
            )
            last_gae = (
                delta
                + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
            )
            advantages[t] = last_gae

        returns = advantages + values

        data["advantages"] = torch.from_numpy(advantages).float()
        data["returns"] = torch.from_numpy(returns).float()

        return data


# ==================== Ray Parameter Server ====================


@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0, num_cpus=2)
class ParameterServer:
    """
    参数服务器: 维护全局模型，执行参数更新
    """

    def __init__(self, config: Dict):
        self.config = PPOConfig(**config)

        # 根据配置设置状态维度
        if self.config.use_lidar:
            self.config.state_dim = 180

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.use_cuda else "cpu"
        )
        print(f"[ParameterServer] Using device: {self.device}")

        self.model = ActorCritic(
            self.config.state_dim, self.config.action_dim, self.config.hidden_dim
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

        self.update_count = 0
        self.total_steps = 0

    def get_weights(self) -> Dict[str, np.ndarray]:
        """获取当前模型参数"""
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

    def update(self, all_workers_data: List[Dict[str, torch.Tensor]]) -> Dict:
        """
        使用所有worker的数据进行PPO更新
        """
        # 合并所有worker的数据
        states = torch.cat([d["states"] for d in all_workers_data]).to(self.device)
        actions = torch.cat([d["actions"] for d in all_workers_data]).to(self.device)
        old_log_probs = torch.cat([d["log_probs"] for d in all_workers_data]).to(
            self.device
        )
        advantages = torch.cat([d["advantages"] for d in all_workers_data]).to(
            self.device
        )
        returns = torch.cat([d["returns"] for d in all_workers_data]).to(self.device)

        # 标准化advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0

        # 多轮epoch训练
        dataset_size = len(states)
        indices = np.arange(dataset_size)

        for epoch in range(self.config.num_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.config.batch_size):
                end = min(start + self.config.batch_size, dataset_size)
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # 前向传播
                _, new_log_probs, entropy, new_values = self.model.get_action_and_value(
                    batch_states, batch_actions
                )

                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # PPO损失
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.config.clip_epsilon,
                        1 + self.config.clip_epsilon,
                    )
                    * batch_advantages
                )
                policy_loss_batch = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss_batch = nn.functional.mse_loss(new_values, batch_returns)

                # 熵奖励
                entropy_loss_batch = -entropy.mean()

                # 总损失
                loss = (
                    policy_loss_batch
                    + self.config.value_coef * value_loss_batch
                    + self.config.entropy_coef * entropy_loss_batch
                )

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                policy_loss += policy_loss_batch.item()
                value_loss += value_loss_batch.item()
                entropy_loss += entropy_loss_batch.item()

        self.update_count += 1
        self.total_steps += dataset_size

        num_batches = dataset_size // self.config.batch_size + (
            1 if dataset_size % self.config.batch_size else 0
        )
        total_updates = self.config.num_epochs * num_batches

        return {
            "loss": total_loss / total_updates,
            "policy_loss": policy_loss / total_updates,
            "value_loss": value_loss / total_updates,
            "entropy": -entropy_loss / total_updates,
            "update_count": self.update_count,
            "total_steps": self.total_steps,
        }

    def save_checkpoint(self, path: str):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "update_count": self.update_count,
                "config": self.config.__dict__,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.update_count = checkpoint["update_count"]
        return True


# ==================== 训练管理器 ====================


class DistributedPPOTrainer:
    def __init__(self, config: Optional[PPOConfig] = None):
        self.config = config or PPOConfig()

        # 根据use_lidar调整state_dim
        if self.config.use_lidar:
            self.config.state_dim = 180

        self.config_dict = {
            "env_name": self.config.env_name,
            "num_workers": self.config.num_workers,
            "num_envs_per_worker": self.config.num_envs_per_worker,
            "use_lidar": self.config.use_lidar,
            "state_dim": self.config.state_dim,
            "action_dim": self.config.action_dim,
            "hidden_dim": self.config.hidden_dim,
            "lr": self.config.lr,
            "gamma": self.config.gamma,
            "gae_lambda": self.config.gae_lambda,
            "clip_epsilon": self.config.clip_epsilon,
            "entropy_coef": self.config.entropy_coef,
            "value_coef": self.config.value_coef,
            "max_grad_norm": self.config.max_grad_norm,
            "steps_per_update": self.config.steps_per_update,
            "num_epochs": self.config.num_epochs,
            "batch_size": self.config.batch_size,
            "total_timesteps": self.config.total_timesteps,
            "use_cuda": self.config.use_cuda,
        }

        # 初始化Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=True)

        print(f"Ray initialized. Available resources: {ray.available_resources()}")

        # 创建Parameter Server
        self.ps = ParameterServer.remote(self.config_dict)

        # 创建Workers
        print(f"Creating {self.config.num_workers} workers...")
        self.workers = [
            RolloutWorker.remote(i, self.config_dict)
            for i in range(self.config.num_workers)
        ]

        self.history = {
            "rewards": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }

    def train(self):
        """主训练循环"""
        print(f"\nStarting distributed training:")
        print(f"  Workers: {self.config.num_workers}")
        print(f"  Envs per worker: {self.config.num_envs_per_worker}")
        print(
            f"  Total environments: {self.config.num_workers * self.config.num_envs_per_worker}"
        )
        print(
            f"  State dim: {self.config.state_dim} ({'LIDAR' if self.config.use_lidar else 'Standard'})"
        )
        print(
            f"  Device: {'CUDA' if self.config.use_cuda and torch.cuda.is_available() else 'CPU'}"
        )

        iteration = 0
        weights = None

        try:
            while True:
                start_time = time.time()

                # 1. 所有Worker并行收集数据
                rollout_futures = [
                    worker.collect_rollout.remote(weights) for worker in self.workers
                ]
                rollouts = ray.get(rollout_futures)

                # 2. 聚合数据
                all_data = [r["data"] for r in rollouts]
                all_stats = [r["stats"] for r in rollouts]

                # 3. Parameter Server更新
                update_future = self.ps.update.remote(all_data)

                # 4. 统计信息
                mean_rewards = [s["mean_reward"] for s in all_stats]
                max_rewards = [s["max_reward"] for s in all_stats]
                total_steps = sum(s["steps"] for s in all_stats)

                # 等待更新完成并获取新权重
                update_info = ray.get(update_future)
                weights = ray.get(self.ps.get_weights.remote())

                iteration += 1

                # 记录历史
                self.history["rewards"].append(np.mean(mean_rewards))
                self.history["policy_loss"].append(update_info["policy_loss"])
                self.history["value_loss"].append(update_info["value_loss"])
                self.history["entropy"].append(update_info["entropy"])

                # 打印日志
                fps = total_steps / (time.time() - start_time)
                print(
                    f"[Iter {iteration:4d}] Steps: {update_info['total_steps']:>9,} | "
                    f"Reward: {np.mean(mean_rewards):>6.2f}±{np.std(mean_rewards):<5.2f} | "
                    f"Max: {np.max(max_rewards):>5.1f} | "
                    f"Policy: {update_info['policy_loss']:>7.4f} | "
                    f"Value: {update_info['value_loss']:>7.4f} | "
                    f"Ent: {update_info['entropy']:>5.3f} | "
                    f"FPS: {fps:>5.0f}"
                )

                # 保存检查点
                if iteration % 50 == 0:
                    os.makedirs("checkpoints", exist_ok=True)
                    path = f"checkpoints/flappy_bird_ppo_iter{iteration}.pt"
                    ray.get(self.ps.save_checkpoint.remote(path))
                    print(f"  -> Checkpoint saved: {path}")

                # 停止条件
                if update_info["total_steps"] >= self.config.total_timesteps:
                    print("\nTraining completed!")
                    break

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")

        # 保存最终模型
        os.makedirs("checkpoints", exist_ok=True)
        final_path = "checkpoints/flappy_bird_ppo_final.pt"
        ray.get(self.ps.save_checkpoint.remote(final_path))
        print(f"Final model saved: {final_path}")

        return self.history

    def evaluate(self, num_episodes: int = 10, render: bool = True):
        """评估训练好的模型"""
        import flappy_bird_gymnasium
        import gymnasium as gym

        weights = ray.get(self.ps.get_weights.remote())

        model = ActorCritic(
            self.config.state_dim, self.config.action_dim, self.config.hidden_dim
        )
        model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
        model.eval()

        env = gym.make(
            self.config.env_name,
            render_mode="human" if render else None,
            use_lidar=self.config.use_lidar,
        )

        rewards = []
        for ep in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                    action, _, _, _ = model.get_action_and_value(state_tensor)

                state, reward, terminated, truncated, _ = env.step(action.item())
                episode_reward += reward
                done = terminated or truncated

            rewards.append(episode_reward)
            print(f"Episode {ep+1}: Reward = {episode_reward:.1f}")

        print(f"\nMean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        env.close()
        return rewards

    def shutdown(self):
        """关闭Ray"""
        ray.shutdown()


# ==================== 主入口 ====================

if __name__ == "__main__":
    # 创建配置
    config = PPOConfig(
        num_workers=4,  # 根据CPU核心数调整
        num_envs_per_worker=4,  # 每个worker的环境数
        steps_per_update=2048,  # 每个worker收集的步数
        total_timesteps=5_000_000,  # 总训练步数（测试时可调小）
        lr=3e-4,
        use_lidar=False,  # False=12维状态, True=180维LIDAR
        use_cuda=torch.cuda.is_available(),
    )

    # 创建训练器
    trainer = DistributedPPOTrainer(config)

    try:
        # 训练
        history = trainer.train()

        # 评估
        print("\n" + "=" * 60)
        print("Final Evaluation")
        print("=" * 60)
        trainer.evaluate(num_episodes=5, render=True)

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        trainer.shutdown()
