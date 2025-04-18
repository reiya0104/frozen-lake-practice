from typing import cast
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from gymnasium.spaces import Discrete
from pprint import pprint
import pandas as pd  # type: ignore

# 環境の作成
env: FrozenLakeEnv = gym.make(
    "FrozenLake-v1", 
    map_name="4x4", 
    is_slippery=True, 
    render_mode="human",
).unwrapped  # type: ignore

print("状態数(nS) = 観測空間の要素数:", cast("Discrete", env.observation_space).n)
print("行動数(nA) = 行動空間の要素数:", cast("Discrete", env.action_space).n)

# 各 H, G, F の説明
# H: 穴
# G: ゴール
# F: 滑る床

class Planner:
    def __init__(self, env: FrozenLakeEnv) -> None:
        self.env = env

    def s_to_loc(self, s: int) -> tuple[int, int]:
        return divmod(s, self.env.ncol)

    def reward(self, s: int) -> float:
        row, col = self.s_to_loc(s)

        match self.env.desc[row, col]:
            case b"H":
                return -1.0
            case b"G":
                return 10.0
            case _:
                return 0.0

    def dict_to_grid(self, V: dict[int, float]) -> pd.DataFrame:
        grid = [[0.0 for _ in range(self.env.ncol)] for _ in range(self.env.nrow)]
        
        for s in V:
            r, c = self.s_to_loc(s)
            grid[r][c] = V[s]

        return pd.DataFrame(grid)

    def plan(self, gamma: float = 0.9, threshold: float = 0.0001) -> pd.DataFrame:
        """
        次の状態の価値を計算する
        V_1(s) = R(s)
        V_{k+1}(s) = R(s) + gamma * max_{a} (sum_{s'} T(s'|s,a) * V_k(s'))
        """
        self.env.reset()
        nS = cast("Discrete", self.env.observation_space).n
        nA = cast("Discrete", self.env.action_space).n
        V: dict[int, float] = {}
        
        for s in range(nS):
            V[s] = self.reward(s)

        while True:
            delta = 0.0
            for s in V:
                row, col = self.s_to_loc(s)
                if self.env.desc[row, col] in [b"H", b"G"]:
                    continue

                expected_reward: list[float] = []
                for a in range(nA):
                    r = 0.0
                    for prob, next_s, _, _ in self.env.P[s][a]:
                        r += float(prob) * V[int(next_s)]
                    expected_reward.append(r)
                
                max_reward = max(expected_reward)
                new_V = self.reward(s) + gamma * max_reward

                # 収束判定
                delta = max(delta, abs(new_V - V[s]))
                V[s] = new_V
                
            if delta < threshold:
                break

        return self.dict_to_grid(V)

    def get_optimal_action(self, s: int, V: pd.DataFrame) -> tuple[int, str]:
        """評価値に基づいて最適な行動を選択"""
        row, col = self.s_to_loc(s)
        if self.env.desc[row, col] in [b"H", b"G"]:
            return 0, "終了"  # 終了状態では行動を選択しない

        best_action = 0
        max_value = float("-inf")
        action_names = ["左", "下", "右", "上"]
        
        for a in range(cast("Discrete", self.env.action_space).n):
            total = 0.0
            for prob, next_s, _, _ in self.env.P[s][a]:
                r, c = self.s_to_loc(int(next_s))
                total += float(prob) * V.iloc[r, c]
            
            if total > max_value:
                max_value = total
                best_action = a
        
        return best_action, action_names[best_action]

    def run_episode(self, V: pd.DataFrame) -> float:
        """評価値に基づいて1エピソードを実行"""
        observation, _ = self.env.reset()
        total_reward = 0.0
        
        while True:
            action, action_name = self.get_optimal_action(observation, V)
            print(f"状態: {observation}, 選択した行動: {action_name} ({action})")
            observation, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"最終状態: {observation}, 報酬: {reward}")
                break
        
        return total_reward

pl = Planner(env)
pl.env.reset()
print(pl.env.render())

# plan の結果をもとに、最適な行動を処理する

V = pl.plan()
print("\n評価値:")
pprint(V)

# 最適な行動でエピソードを実行
print("\nエピソード開始:")
total_reward = pl.run_episode(V)
print(f"\n総報酬: {total_reward}")
