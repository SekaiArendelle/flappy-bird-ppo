import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import flappy_bird_gymnasium
import gymnasium as gym

# import normal
import constant
from model import ActorCritic
from typing import List, Optional, Tuple, Dict


# ==================== 经验缓冲区 ====================
class RolloutBuffer:
    def __init__(self) -> None:
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[torch.Tensor] = []
        self.dones: List[float] = []

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: bool,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(float(value))
        self.log_probs.append(log_prob)
        self.dones.append(float(done))

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def get(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return (
            torch.stack(self.states).to(constant.DEVICE),
            torch.stack(self.actions).to(constant.DEVICE),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.values, dtype=torch.float32),
            torch.stack(self.log_probs),
            torch.tensor(self.dones, dtype=torch.float32),
        )


# ==================== GAE 优势估计 ====================
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    next_value: float = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """广义优势估计 (Generalized Advantage Estimation)"""
    advantages = torch.zeros_like(rewards)
    last_advantage = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = (
                values[t + 1].item()
                if isinstance(values[t + 1], torch.Tensor)
                else values[t + 1]
            )

        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        advantages[t] = last_advantage = (
            delta + gamma * lambda_ * (1 - dones[t]) * last_advantage
        )

    returns = advantages + values
    return advantages, returns


# ==================== PPO 训练器 ====================
class PPOTrainer:
    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.policy: ActorCritic = ActorCritic(
            state_dim, action_dim, constant.HIDDEN_DIM
        ).to(constant.DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=constant.LEARNING_RATE)

        # 创建学习率调度器
        self.scheduler = self._create_scheduler()

        self.buffer: RolloutBuffer = RolloutBuffer()
        self.global_step = 0

    def _create_scheduler(self):
        """创建学习率调度器"""
        if constant.LR_SCHEDULE == "linear":

            def lr_lambda(step):
                progress = min(step / constant.LR_DECAY_STEPS, 1.0)
                return max(
                    1.0 - progress, constant.MIN_LEARNING_RATE / constant.LEARNING_RATE
                )

        elif constant.LR_SCHEDULE == "exponential":

            def lr_lambda(step):
                decay_rate = (
                    np.log(constant.MIN_LEARNING_RATE / constant.LEARNING_RATE)
                    / constant.LR_DECAY_STEPS
                )
                return np.exp(decay_rate * step)

        elif constant.LR_SCHEDULE == "cosine":

            def lr_lambda(step):
                progress = min(step / constant.LR_DECAY_STEPS, 1.0)
                return (1 + np.cos(np.pi * progress)) / 2 * (
                    1 - constant.MIN_LEARNING_RATE / constant.LEARNING_RATE
                ) + constant.MIN_LEARNING_RATE / constant.LEARNING_RATE

        else:

            def lr_lambda(step):
                return 1.0

        return LambdaLR(self.optimizer, lr_lambda)

    def update(self, last_state: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """PPO 策略更新"""
        states, actions, rewards, old_values, old_log_probs, dones = self.buffer.get()

        # 转移到设备
        states = states.to(constant.DEVICE)
        actions = actions.to(constant.DEVICE)
        rewards = rewards.to(constant.DEVICE)
        old_values = old_values.to(constant.DEVICE)
        old_log_probs = old_log_probs.to(constant.DEVICE)
        dones = dones.to(constant.DEVICE)

        # 计算最后一个状态的 value 用于 GAE
        next_value = 0.0
        if last_state is not None:
            with torch.no_grad():
                _, _, _, next_val = self.policy.get_action(last_state.unsqueeze(0))
                next_value = float(next_val.squeeze().item())

        # 计算 GAE
        advantages, returns = compute_gae(
            rewards,
            old_values,
            dones,
            constant.GAMMA,
            constant.GAE_LAMBDA,
            next_value,
        )

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多轮 epoch 更新
        dataset_size = len(states)
        indices = np.arange(dataset_size)

        for epoch in range(constant.NUM_EPOCHS):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, constant.BATCH_SIZE):
                end = start + constant.BATCH_SIZE
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # 评估当前策略
                log_probs, entropy, values = self.policy.evaluate(
                    batch_states, batch_actions
                )

                # 计算比率
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # PPO 裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - constant.CLIP_EPSILON,
                        1 + constant.CLIP_EPSILON,
                    )
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值函数损失
                value_loss = nn.MSELoss()(values, batch_returns)

                # 熵奖励 (鼓励探索)
                entropy_loss = -entropy.mean()

                # 总损失
                loss = (
                    policy_loss
                    + constant.VALUE_COEF * value_loss
                    + constant.ENTROPY_COEF * entropy_loss
                )

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), constant.MAX_GRAD_NORM
                )
                self.optimizer.step()

        # 步进学习率调度器
        self.scheduler.step()

        self.buffer.clear()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(-entropy_loss.item()),
        }


# ==================== 主训练循环 ====================
def train() -> ActorCritic:
    # 创建环境 (使用向量状态而非图像)
    env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)
    state_dim = env.observation_space.shape[0]  # 通常是12维向量状态
    action_dim = env.action_space.n  # 2个动作: 0=不操作, 1=跳跃

    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Device: {constant.DEVICE}")

    trainer = PPOTrainer(state_dim, action_dim)

    episode_rewards = []
    global_step = 0

    for episode in range(constant.NUM_EPISODES):
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(constant.DEVICE)

        episode_reward = 0
        episode_length = 0

        while True:
            # 采样动作
            with torch.no_grad():
                action, log_prob, entropy, value = trainer.policy.get_action(
                    state.unsqueeze(0)
                )

            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            if reward == 0.1 and next_state[3] != 1 and next_state[4] != 0:
                player_y = next_state[0]
                next_pipe_top = next_state[3]
                next_pipe_bottom = next_state[4]
                # gap_center = (next_pipe_top + next_pipe_bottom) / 2
                # reward = normal.normal_pdf(player_y, gap_center, 0.4)
                if next_pipe_bottom < player_y < next_pipe_top:
                    reward = 0.25
                else:
                    reward = -0.2
            elif (
                reward == 0.1
                and len(trainer.buffer.actions) >= 3
                and trainer.buffer.actions[-1]
                == trainer.buffer.actions[-2]
                == trainer.buffer.actions[-3]
                == 1
            ):
                reward = -0.2
            elif reward == 0.1:
                reward = 0.1

            # 存储经验
            trainer.buffer.add(state, action, reward, value.squeeze(), log_prob, done)

            episode_reward += reward
            episode_length += 1
            global_step += 1

            state = torch.FloatTensor(next_state).to(constant.DEVICE)

            # 达到更新步数时执行 PPO 更新
            if len(trainer.buffer.states) >= constant.STEPS_PER_UPDATE:
                # 传入 last_state 用于计算 GAE 的 bootstrap value
                losses = trainer.update(last_state=state if not done else None)

                if episode % 10 == 0:
                    print(
                        f"Episode {episode} | Step {global_step} | "
                        f"Reward: {np.mean(episode_rewards[-10:]):.2f} | "
                        f"Policy Loss: {losses['policy_loss']:.4f} | "
                        f"Entropy: {losses['entropy']:.4f}"
                    )

            if done:
                break

        episode_rewards.append(episode_reward)

        # 每100个 episode 保存模型并打印统计
        if episode % 100 == 0 and episode > 0:
            avg_reward = np.mean(episode_rewards[-100:])
            max_reward = np.max(episode_rewards[-100:])
            print(
                f"=== Episode {episode} | Avg Reward: {avg_reward:.2f} | Max: {max_reward:.2f} ==="
            )

            # 保存模型
            torch.save(trainer.policy.state_dict(), f"ppo_flappy_{episode}.pth")

    env.close()
    return trainer.policy


if __name__ == "__main__":
    # 训练
    policy = train()

    # 测试 (取消注释以运行)
    # test("ppo_flappy_4000.pth")
