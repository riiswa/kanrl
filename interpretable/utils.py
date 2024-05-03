import os

import torch
import numpy as np
import gymnasium as gym
from huggingface_sb3 import load_from_hub

from stable_baselines3 import PPO, A2C, TD3
from gymnasium.wrappers import RecordVideo

ALGOS = {"ppo": PPO, "a2c": A2C, "td3": TD3}


def rollouts(env, policy, num_episodes=1):
    observations = []
    actions = []

    for episode in range(num_episodes):
        done = False
        observation, _ = env.reset()
        while not done:
            action = policy(observation)
            observations.append(observation)
            actions.append(action)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()

    observations = np.vstack(observations)
    actions = np.hstack(actions)
    if actions.ndim == 1:
        actions = actions[:, None]
    return observations, actions


def generate_dataset_from_expert(algo, env_name, num_train_episodes=90, num_test_episodes=10, force=False):
    dataset_path = os.path.join("datasets", f"{algo}-{env_name}.pt")
    if os.path.exists(dataset_path) and not force:
        return
    repo_id = f"sb3/{algo}-{env_name}"
    policy_file = f"{algo}-{env_name}.zip"

    checkpoint = load_from_hub(repo_id, policy_file)

    expert = ALGOS[algo].load(checkpoint)
    train_env = gym.make(env_name)
    test_env = gym.make(env_name, render_mode="rgb_array")
    test_env = RecordVideo(test_env, video_folder="videos", episode_trigger=lambda x: True, name_prefix=f"{algo}-{env_name}")

    def policy(obs):
        return expert.predict(obs, deterministic=True)[0]

    train_observations, train_actions = rollouts(train_env, policy, num_train_episodes)
    test_observations, test_actions = rollouts(test_env, policy, num_test_episodes)

    dataset = {
        "train_input": torch.from_numpy(train_observations),
        "test_input": torch.from_numpy(test_observations),
        "train_label": torch.from_numpy(train_actions),
        "test_label": torch.from_numpy(test_actions)
    }

    os.makedirs("datasets", exist_ok=True)
    torch.save(dataset, dataset_path)


if __name__ == "__main__":
    generate_dataset_from_expert("ppo", "CartPole-v1")
