import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import flappy_bird_gymnasium
import gymnasium as gym
# import normal
from torch.distributions import Categorical


# ==================== 超参数配置 ====================
class Config:
    # 环境
    env_name = "FlappyBird-v0"
    render_mode = None  # "human" 用于可视化
    # render_mode = "human"

    # 网络
    hidden_dim = 256

    # PPO
    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.1
    max_grad_norm = 0.5

    # 训练
    num_episodes = 5000
    steps_per_update = 2048
    num_epochs = 10
    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== Actor-Critic 网络 ====================
class ActorCritic(nn.Module):
    """共享 backbone 的 Actor-Critic 架构"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Actor: 策略网络 (输出动作概率)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic: 价值网络 (输出状态价值)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

    def get_action(self, state, deterministic=False):
        """采样动作并返回 log_prob"""
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)

        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, state_value

    def evaluate(self, state, action):
        """评估给定状态下动作的对数概率和价值"""
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, state_value.squeeze(-1)


# ==================== 经验缓冲区 ====================
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def get(self):
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.values, dtype=torch.float32),
            torch.stack(self.log_probs),
            torch.tensor(self.dones, dtype=torch.float32),
        )


# ==================== GAE 优势估计 ====================
def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95, next_value=0):
    """广义优势估计 (Generalized Advantage Estimation)"""
    advantages = torch.zeros_like(rewards)
    last_advantage = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        advantages[t] = last_advantage = (
            delta + gamma * lambda_ * (1 - dones[t]) * last_advantage
        )

    returns = advantages + values
    return advantages, returns


# ==================== PPO 训练器 ====================
class PPOTrainer:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.policy = ActorCritic(state_dim, action_dim, config.hidden_dim).to(
            config.device
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.buffer = RolloutBuffer()

    def update(self, last_state=None):
        """PPO 策略更新"""
        states, actions, rewards, old_values, old_log_probs, dones = self.buffer.get()

        # 转移到设备
        states = states.to(self.config.device)
        actions = actions.to(self.config.device)
        rewards = rewards.to(self.config.device)
        old_values = old_values.to(self.config.device)
        old_log_probs = old_log_probs.to(self.config.device)
        dones = dones.to(self.config.device)

        # 计算最后一个状态的 value 用于 GAE
        next_value = 0
        if last_state is not None:
            with torch.no_grad():
                _, _, _, next_val = self.policy.get_action(last_state.unsqueeze(0))
                next_value = next_val.squeeze().item()

        # 计算 GAE
        advantages, returns = compute_gae(
            rewards,
            old_values,
            dones,
            self.config.gamma,
            self.config.gae_lambda,
            next_value,
        )

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多轮 epoch 更新
        dataset_size = len(states)
        indices = np.arange(dataset_size)

        for epoch in range(self.config.num_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.config.batch_size):
                end = start + self.config.batch_size
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
                        1 - self.config.clip_epsilon,
                        1 + self.config.clip_epsilon,
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
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

        self.buffer.clear()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
        }


# ==================== 主训练循环 ====================
def train():
    config = Config()

    # 创建环境 (使用向量状态而非图像)
    env = gym.make(config.env_name, render_mode=config.render_mode, use_lidar=False)
    state_dim = env.observation_space.shape[0]  # 通常是12维向量状态
    action_dim = env.action_space.n  # 2个动作: 0=不操作, 1=跳跃

    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Device: {config.device}")

    trainer = PPOTrainer(state_dim, action_dim, config)

    episode_rewards = []
    global_step = 0

    for episode in range(config.num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state)

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
            elif reward == 0.1 and len(trainer.buffer.actions) >= 3 and \
                  trainer.buffer.actions[-1] ==  \
                 trainer.buffer.actions[-2] == \
                    trainer.buffer.actions[-3] == 1:
                reward = -0.2
            elif reward == 0.1:
                reward = 0.1

            # 存储经验
            trainer.buffer.add(state, action, reward, value.squeeze(), log_prob, done)

            episode_reward += reward
            episode_length += 1
            global_step += 1

            state = torch.FloatTensor(next_state)

            # 达到更新步数时执行 PPO 更新
            if len(trainer.buffer.states) >= config.steps_per_update:
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


# ==================== 测试/可视化 ====================
def test(model_path, num_episodes=5):
    """测试训练好的模型"""
    config = Config()
    env = gym.make(config.env_name, render_mode="human", use_lidar=False)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = ActorCritic(state_dim, action_dim, config.hidden_dim)
    policy.load_state_dict(torch.load(model_path))
    policy.to(config.device)
    policy.eval()

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.device)
                action, _, _, _ = policy.get_action(state_tensor, deterministic=True)

            state, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward

            if terminated or truncated:
                print(f"Test Episode {episode + 1}: Reward = {episode_reward}")
                break

    env.close()


if __name__ == "__main__":
    # 训练
    policy = train()

    # 测试 (取消注释以运行)
    # test("ppo_flappy_4000.pth")
