# REINFORCEアルゴリズムの実装サンプル

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


# [LaTeX コード開始]
# \[
# \pi_\theta(a\mid s) = P(a\mid s;\theta)
# \]
# \[
# G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k
# \]
# \[
# \nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\bigl[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t\mid s_t) G_t\bigr]
# \]
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


def reinforce(
    env_name: str = "CartPole-v1",
    learning_rate: float = 1e-2,
    gamma: float = 0.99,
    episodes: int = 1000,
) -> None:
    env = gym.make(env_name)
    state_dim = (
        env.observation_space.shape[0] if env.observation_space.shape is not None else 1
    )
    action_dim = int(env.action_space.n)  # type: ignore
    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    for episode in range(episodes):
        state, _ = env.reset()
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        done: bool = False

        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            probs = policy(state_tensor)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(float(reward))
            state = next_state

        # モンテカルロ法でリターンを計算
        returns: list[float] = []
        G: float = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_tensor = torch.tensor(returns)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (
            returns_tensor.std() + 1e-9
        )

        # 損失関数: REINFORCE
        # [LaTeX 損失関数]
        # \[
        # L(\theta) = -\sum_{t=0}^{T-1} \log\pi_\theta(a_t\mid s_t) G_t
        # \]
        loss = torch.tensor(0.0, requires_grad=True)
        for log_prob, G_t in zip(log_probs, returns_tensor):
            loss = loss - log_prob * G_t

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 100 == 0:
            print(
                "Episode {}\tTotal Reward: {}\tLoss: {:.3f}".format(
                    episode + 1, sum(rewards), loss.item()
                )
            )

    env.close()


if __name__ == "__main__":
    reinforce()
