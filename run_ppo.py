import gymnasium as gym
import torch
from gymnasium.spaces import Discrete

from PPO import ActorNetwork, CriticNetwork, load_model


def run_model(model_path: str, env_name: str = "CartPole-v1", episodes: int = 10):
    # 環境の設定
    env = gym.make(env_name, render_mode="human")
    state_dim = (
        env.observation_space.shape[0] if env.observation_space.shape is not None else 1
    )
    action_dim = (
        int(env.action_space.n) if isinstance(env.action_space, Discrete) else 1
    )

    # モデルの初期化
    actor = ActorNetwork(state_dim, action_dim, hidden_dim=212)
    critic = CriticNetwork(state_dim, hidden_dim=212)

    # モデルの読み込み
    start_episode, best_reward = load_model(actor, critic, model_path)
    print(f"モデルを読み込みました: {model_path}")
    print(f"開始エピソード: {start_episode}, 最高報酬: {best_reward}")

    # GPUの使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor.to(device)
    critic.to(device)
    actor.eval()  # 推論モードに設定

    # エピソードの実行
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward: float = 0.0
        done = False

        while not done:
            # 状態をテンソルに変換
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # アクションの選択
            with torch.no_grad():
                action_probs = actor(state_tensor)
                action = torch.argmax(action_probs).item()

            # 環境でステップを実行
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += float(reward)
            state = next_state

        print(f"エピソード {episode + 1}: 合計報酬 = {total_reward}")

    env.close()


if __name__ == "__main__":
    # モデルファイルのパスを指定
    model_path = "models/final/ppo_20250423_023611_ep200_reward500.0.pt"
    run_model(model_path, episodes=5)
