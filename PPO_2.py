from typing import List, TypedDict

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class ModelConfig(TypedDict):
    fcnet_hiddens: List[int]
    fcnet_activation: str


class Config(TypedDict):
    env: str
    framework: str
    model: ModelConfig


# 学習環境設定
config: Config = {
    "env": "CartPole-v1",  # 環境の名前
    "framework": "torch",  # フレームワーク
    "model": {
        "fcnet_hiddens": [32],  # ニューラルネットワークの隠れ層のノード数
        "fcnet_activation": "linear",  # 活性化関数
    },
}

# 環境の設定
env = gym.make(config["env"], render_mode="rgb_array")

print(f"環境の名前: {env.unwrapped.spec.id if env.unwrapped.spec else 'Unknown'}")
print(f"環境の観察空間: {env.observation_space}")
print(f"環境の行動空間: {env.action_space}")


def before_training(env: gym.Env) -> None:
    # 学習前の動画を保存するための設定
    before_training_path = "before_training"
    env = RecordVideo(env, before_training_path, episode_trigger=lambda x: True)

    env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            env.reset()

    env.close()


# before_training(env)


def main_training(env: gym.Env) -> None:
    # 学習
    vec_env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=30000)

    # 学習後の動画を保存
    after_training_path = "after_training"
    video_env = RecordVideo(
        vec_env.envs[0], after_training_path, episode_trigger=lambda x: True
    )

    observation, _ = video_env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation)
        observation, _, terminated, truncated, _ = video_env.step(action)
        done = terminated or truncated

    video_env.close()


if __name__ == "__main__":
    main_training(env)
