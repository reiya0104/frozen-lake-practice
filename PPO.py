# PPO (Proximal Policy Optimization) アルゴリズムの実装サンプル

import json
import os
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.trial import Trial

# [LaTeX コード開始]
# \[
# L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]
# \]
# \[
# r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
# \]
# \[
# \hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}
# \]
# \[
# \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
# \]


class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(self.fc3(x))


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class TrainingMetrics:
    def __init__(self) -> None:
        self.episodes: list[int] = []
        self.rewards: list[float] = []
        self.actor_losses: list[float] = []
        self.critic_losses: list[float] = []
        self.avg_rewards: list[float] = []

    def update(
        self, episode: int, reward: float, actor_loss: float, critic_loss: float
    ) -> None:
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)

        # 移動平均の計算（直近10エピソード）
        window_size = min(10, len(self.rewards))
        avg_reward = sum(self.rewards[-window_size:]) / window_size
        self.avg_rewards.append(avg_reward)

    def plot(self, save_path: str | None = None) -> None:
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # 報酬のプロット
        ax1.plot(self.episodes, self.rewards, label="Reward", alpha=0.3)
        ax1.plot(
            self.episodes, self.avg_rewards, label="Moving Average (10)", color="red"
        )
        ax1.set_title("Training Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.legend()
        ax1.grid(True)

        # 損失のプロット
        ax2.plot(self.episodes, self.actor_losses, label="Actor Loss", color="blue")
        ax2.plot(self.episodes, self.critic_losses, label="Critic Loss", color="orange")
        ax2.set_title("Training Losses")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Training curves saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def check_early_stopping(self, patience: int = 50, min_delta: float = 0.0) -> bool:
        """早期停止の判定を行う

        Args:
            patience: 改善が見られないエピソード数の閾値
            min_delta: 改善とみなす最小変化量

        Returns:
            bool: 学習を停止すべきかどうか
        """
        if len(self.avg_rewards) < patience:
            return False

        # 過去patience回分の移動平均報酬を確認
        recent_rewards = self.avg_rewards[-patience:]
        best_reward = max(recent_rewards[:-1])  # 最新のrewardを除いた最大値

        # 最新のrewardが十分な改善を示しているかチェック
        return recent_rewards[-1] <= best_reward + min_delta


def save_model(
    actor: nn.Module,
    critic: nn.Module,
    episode: int,
    reward: float,
    optimizer_actor: torch.optim.Optimizer | None = None,
    optimizer_critic: torch.optim.Optimizer | None = None,
    metrics: TrainingMetrics | None = None,
    save_dir: str = "models",
) -> str:
    """モデルを保存する関数

    Args:
        actor: アクターネットワーク
        critic: クリティックネットワーク
        episode: 現在のエピソード数
        reward: 獲得した報酬
        optimizer_actor: アクターの最適化器
        optimizer_critic: クリティックの最適化器
        metrics: 学習の指標
        save_dir: 保存先ディレクトリ

    Returns:
        str: 保存したファイルのパス
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ppo_{timestamp}_ep{episode}_reward{reward:.1f}.pt"
    save_path = os.path.join(save_dir, filename)

    save_dict = {
        "episode": episode,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "reward": reward,
    }

    if optimizer_actor is not None:
        save_dict["optimizer_actor_state_dict"] = optimizer_actor.state_dict()
    if optimizer_critic is not None:
        save_dict["optimizer_critic_state_dict"] = optimizer_critic.state_dict()
    if metrics is not None:
        save_dict["metrics"] = {
            "rewards": metrics.rewards,
            "actor_losses": metrics.actor_losses,
            "critic_losses": metrics.critic_losses,
            "avg_rewards": metrics.avg_rewards,
        }

    torch.save(save_dict, save_path)
    print(f"Model saved to {save_path}")
    return save_path


def load_model(
    actor: nn.Module, critic: nn.Module, model_path: str
) -> tuple[int, float]:
    """モデルを読み込む関数"""
    checkpoint = torch.load(model_path)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    return checkpoint["episode"], checkpoint["reward"]


def objective(trial: Trial) -> float:
    """Optunaの目的関数"""
    # ハイパーパラメータの探索範囲を定義
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.99),
        "epsilon": trial.suggest_float("epsilon", 0.1, 0.3),
        "epochs": int(trial.suggest_int("epochs", 5, 20)),
        "batch_size": int(trial.suggest_int("batch_size", 32, 256)),
        "hidden_dim": int(trial.suggest_int("hidden_dim", 32, 256)),
    }

    # 評価用の短いトレーニング実行
    metrics = TrainingMetrics()
    ppo(
        env_name="CartPole-v1",
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        epsilon=params["epsilon"],
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        hidden_dim=int(params["hidden_dim"]),
        episodes=200,  # 短い評価用エピソード数
        metrics=metrics,
    )

    # 最後の10エピソードの平均報酬を評価指標として使用
    return sum(metrics.rewards[-10:]) / 10


def ppo(
    env_name: str = "CartPole-v1",
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    epsilon: float = 0.2,
    epochs: int = 10,
    batch_size: int = 64,
    episodes: int = 1000,
    load_model_path: str | None = None,
    metrics: TrainingMetrics | None = None,
    early_stopping_patience: int = 50,
    early_stopping_delta: float = 1.0,
    hidden_dim: int = 64,
) -> None:
    env = gym.make(env_name)
    state_dim = (
        env.observation_space.shape[0] if env.observation_space.shape is not None else 1
    )
    action_dim = int(env.action_space.n)  # type: ignore

    actor = ActorNetwork(state_dim, action_dim, hidden_dim=hidden_dim)
    critic = CriticNetwork(state_dim, hidden_dim=hidden_dim)
    optimizer_actor = optim.Adam(actor.parameters(), lr=learning_rate)
    optimizer_critic = optim.Adam(critic.parameters(), lr=learning_rate)

    # GPUの使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor.to(device)
    critic.to(device)

    best_reward = float("-inf")

    # モデルの読み込み
    if load_model_path:
        start_episode, best_reward = load_model(actor, critic, load_model_path)
        print(f"Loaded model from {load_model_path}")
        print(f"Starting from episode {start_episode} with best reward {best_reward}")
    else:
        start_episode = 0

    if metrics is None:
        metrics = TrainingMetrics()

    for episode in range(start_episode, episodes):
        state, _ = env.reset()
        states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        done: bool = False

        # データ収集
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_probs = actor(state_tensor)
                value = critic(state_tensor)

            m = torch.distributions.Categorical(action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action.item())
            rewards.append(float(reward))
            log_probs.append(log_prob)
            values.append(value)

            state = next_state

        # アドバンテージの計算
        returns: list[float] = []
        G: float = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns_tensor = torch.FloatTensor(returns).to(device)
        values_tensor = torch.cat(values).squeeze().to(device)
        advantages_tensor = returns_tensor - values_tensor

        # 正規化
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )

        # PPOの更新
        for _ in range(epochs):
            for idx in range(0, len(states), batch_size):
                batch_states = torch.FloatTensor(states[idx : idx + batch_size]).to(
                    device
                )
                batch_actions = torch.LongTensor(actions[idx : idx + batch_size]).to(
                    device
                )
                batch_old_log_probs = torch.cat(log_probs[idx : idx + batch_size]).to(
                    device
                )
                batch_advantages = advantages_tensor[idx : idx + batch_size].to(device)
                batch_returns = returns_tensor[idx : idx + batch_size].to(device)

                # アクターの更新
                action_probs = actor(batch_states)
                m = torch.distributions.Categorical(action_probs)
                new_log_probs = m.log_prob(batch_actions)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # クリティックの更新
                value_pred = critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(value_pred, batch_returns)

                # 最適化
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                actor_loss.backward()
                critic_loss.backward()

                # 勾配の監視
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)

                # 最適化
                optimizer_actor.step()
                optimizer_critic.step()

                # 学習の進捗監視
                if (episode + 1) % 10 == 0:
                    print(
                        f"Gradient norm - Actor: {torch.norm(actor.fc1.weight.grad):.3f}"
                    )

        # メトリクスの更新
        metrics.update(episode + 1, sum(rewards), actor_loss.item(), critic_loss.item())

        if (episode + 1) % 100 == 0:
            print(
                "Episode {}\tTotal Reward: {}\tActor Loss: {:.3f}\tCritic Loss: {:.3f}".format(
                    episode + 1, sum(rewards), actor_loss.item(), critic_loss.item()
                )
            )
            # モデルの保存（報酬が改善した場合）
            if episode == 0 or sum(rewards) > best_reward:
                best_reward = sum(rewards)
                best_model_path = save_model(
                    actor,
                    critic,
                    episode + 1,
                    best_reward,
                    optimizer_actor,
                    optimizer_critic,
                    metrics,
                    save_dir="models/best",
                )
                print(f"New best model saved to {best_model_path}")
                metrics.plot(f"training_curves_ep{episode + 1}.png")

            # 早期停止の判定
            if metrics.check_early_stopping(
                early_stopping_patience, early_stopping_delta
            ):
                print(f"Early stopping triggered at episode {episode + 1}")
                break

    # 最終モデルの保存
    final_model_path = save_model(
        actor,
        critic,
        episodes,
        sum(rewards),
        optimizer_actor,
        optimizer_critic,
        metrics,
        save_dir="models/final",
    )
    print(f"Training completed. Final model saved to {final_model_path}")
    env.close()


def optimize_hyperparameters(n_trials: int = 100) -> optuna.Study:
    """ハイパーパラメータの最適化を実行"""
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 最適化の結果を保存
    Path("optimization_results").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"optimization_results/results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "best_value": trial.value,
                "best_params": trial.params,
                "n_trials": n_trials,
            },
            f,
            indent=4,
        )
    print(f"Optimization results saved to {results_path}")

    # 最適化の結果を可視化
    optuna.visualization.plot_optimization_history(study)
    plt.savefig("optimization_history.png")
    plt.close()

    optuna.visualization.plot_param_importances(study)
    plt.savefig("param_importances.png")
    plt.close()

    return study


if __name__ == "__main__":
    # ハイパーパラメータの最適化を実行
    study = optimize_hyperparameters(n_trials=50)
    best_params = study.best_trial.params
    ppo(
        learning_rate=best_params["learning_rate"],
        gamma=best_params["gamma"],
        epsilon=best_params["epsilon"],
        epochs=int(best_params["epochs"]),
        batch_size=int(best_params["batch_size"]),
        hidden_dim=int(best_params["hidden_dim"]),
        early_stopping_patience=50,
        early_stopping_delta=1.0,
    )
