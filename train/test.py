import torch
from constant import Config
import flappy_bird_gymnasium
import gymnasium as gym
from model import ActorCritic

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
