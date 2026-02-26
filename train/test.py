import os
import torch
import constant
import flappy_bird_gymnasium
import gymnasium as gym
from model import ActorCritic

# ==================== 测试/可视化 ====================
def test(model_path, num_episodes=5):
    """测试训练好的模型"""
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = ActorCritic(state_dim, action_dim, constant.HIDDEN_DIM)
    policy.load_state_dict(torch.load(model_path))
    policy.to(constant.DEVICE)
    policy.eval()

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(constant.DEVICE)
                action, _, _, _ = policy.get_action(state_tensor, deterministic=True)

            state, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward

            if terminated or truncated:
                print(f"Test Episode {episode + 1}: Reward = {episode_reward}")
                break

    env.close()

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(SCRIPT_DIR, "ppo_flappy_3400.pth")
    test(model_path)
