# 方策反復法

from typing import cast

import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from gymnasium.spaces import Discrete

# 環境の作成
env: FrozenLakeEnv = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=True,
    render_mode="ansi",
).unwrapped  # type: ignore

# env.reset(seed=0)
# np.random.seed(0)


class PolicyIteration:
    def __init__(self, env: FrozenLakeEnv, gamma: float = 0.9, threshold: float = 1e-6):
        self.env = env
        self.gamma = gamma
        self.threshold = threshold
        self.nS = cast(Discrete, env.observation_space).n
        self.nA = cast(Discrete, env.action_space).n
        self.V = np.zeros(self.nS)
        self.Q = np.zeros((self.nS, self.nA))
        self.pi = np.ones((self.nS, self.nA)) / self.nA
        self.pi_new = np.ones((self.nS, self.nA)) / self.nA

    def policy_evaluation(self) -> np.ndarray:
        """
        方策評価を行う関数
        pi: 方策(確率分布), pi[s, a] = P(a|s)
        V: 状態価値関数, V[s] = V(s)
        gamma: 割引率
        ベルマン作用素を計算する
        ベルマン作用素: 状態価値関数を更新する作用素
        r'\\hat{V}^{\\pi, new} (s) = \\sum_{a} \\pi(a|s) \\sum_{s'} \\sum_{r} P(s', r|s, a) [r + \\gamma \\hat{V}^{\\pi} (s')]'
        """

        def bellman_operator(V: np.ndarray) -> np.ndarray:
            V_new = np.zeros_like(V)
            for s in range(self.nS):
                for a in range(self.nA):
                    for p, s_prime, r, _ in self.env.P[s][a]:
                        V_new[s] += self.pi[s, a] * p * (r + self.gamma * V[s_prime])
            return V_new

        while True:
            V_new = bellman_operator(self.V)
            if np.max(np.abs(V_new - self.V)) < self.threshold:
                self.V = V_new
                break
            self.V = V_new
        return self.V

    def policy_improvement(self) -> np.ndarray:
        """
        方策改善を行う関数
        pi: 方策(確率分布), pi[s, a] = P(a|s)
        V: 状態価値関数, V[s] = V(s)
        gamma: 割引率
        r'\\hat{Q}^{\\pi} (s, a) = \\sum_{s'} \\sum_{r} P(s', r|s, a) [r + \\gamma \\hat{V}^{\\pi} (s')]'
        r'\\pi_new(a|s) = argmax_{a} \\hat{Q}^{\\pi} (s, a)'
        """
        self.Q = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                for p, s_prime, r, _ in self.env.P[s][a]:
                    self.Q[s, a] += p * (r + self.gamma * self.V[s_prime])
        self.pi_new = np.zeros_like(self.Q)
        for s in range(self.nS):
            self.pi_new[s, np.argmax(self.Q[s])] = 1
        return self.pi_new

    def policy_iteration(self) -> np.ndarray:
        """
        方策反復法を行う関数
        """
        i = 0
        while True:
            self.policy_evaluation()
            self.policy_improvement()
            if np.max(np.abs(self.pi - self.pi_new)) < self.threshold:
                print(f"収束しました。反復回数: {i}")
                break
            self.pi = self.pi_new
            i += 1
        return self.pi

    def play(
        self, render: bool = False, delay: float = 0.0, verbose: bool = False
    ) -> None:
        """
        方策を実行する関数
        """
        import time

        def get_position(state: int) -> tuple[int, int]:
            return state // self.env.nrow, state % self.env.ncol

        def get_action(action: int) -> str:
            return ["←", "↓", "→", "↑"][action]

        if verbose:
            print("ゲーム開始!")

        state, _ = self.env.reset()
        while True:
            action = int(np.argmax(self.pi[state]))
            if verbose:
                print(f"場所: {get_position(state)}, 行動: {get_action(action)}")
            state, reward, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                if verbose:
                    print(f"報酬: {reward}, ゲーム終了!")
                break
            if render:
                self.env.render()
            if delay > 0:
                time.sleep(delay)
        self.env.close()

    def evaluate(self, episodes: int = 100) -> tuple[float, float]:
        """
        方策の定量評価を行う関数
        episodes: 実行するエピソード数
        返り値: (平均報酬, 成功率)
        """
        total_rewards = []
        successes = 0
        for ep in range(episodes):
            state, _ = self.env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            while True:
                action = int(np.argmax(self.pi[state]))
                state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                if terminated or truncated:
                    if reward == 1.0:
                        successes += 1
                    break
            total_rewards.append(total_reward)
        avg_reward = float(np.mean(total_rewards))
        success_rate = successes / episodes
        return avg_reward, success_rate


if __name__ == "__main__":
    # 初期化・実行コード
    print("Policy Iteration: 方策反復法")
    policy_iteration = PolicyIteration(env)
    policy_iteration.policy_iteration()
    print("\n状態価値関数 V:")
    print(pl.DataFrame(policy_iteration.V))
    print("\n行動価値関数 Q:")
    print(pl.DataFrame(policy_iteration.Q))
    print("\n方策 π:")
    print(pl.DataFrame(policy_iteration.pi))

    # 方策性能評価
    avg_reward, success_rate = policy_iteration.evaluate(1000)
    print("\n方策性能評価(1000エピソード)")
    print(f"平均報酬: {avg_reward:.3f}")
    print(f"成功率: {success_rate:.2%}")

    # ゲーム進行を高速化: レンダリング・ログ出力を無効化
    policy_iteration.play(render=False, verbose=False)
