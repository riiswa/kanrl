import os
import requests

import torch
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO, A2C, TD3
from gymnasium.wrappers import RecordVideo

# Define a mapping betwen algo name and algos:(could also prevent from typos / non existing algos args)
ALGOS = {"ppo": PPO, "a2c": A2C, "td3": TD3}
SAVING_DIR = "policy"
# Could do the same for environments to get the labels of their obs / actions (what we discussed on discord)

def load_from_hub(repo_id, file_name, saving_dir):
    hf_policy_url = f"https://huggingface.co/{repo_id}/resolve/main/{file_name}"
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    target_file_path = os.path.join(saving_dir, file_name)
    response = requests.get(hf_policy_url, stream=True)

    if response.status_code == 200:
        with open(target_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"File downloaded and saved to: {target_file_path}")
    else:
        # Should maybe raise a value error
        print(f"Failed to download the file. Status code: {response.status_code}")

    return

# Change to use num_episodes instead of num_steps
def rollout(env_name, policy, num_steps, render=None):
    # Initialize the env and the policy
    env = gym.make(env_name, render_mode=render)
    expert = ALGOS[algo].load(os.path.join(SAVING_DIR, policy))

    # Generate a dataset 
    observations = []
    actions = []

    s, _ = env.reset()
    for _ in range(num_steps):
        a = expert.predict(s, deterministic=True)[0]
        observations.append(s)
        actions.append(a)
        s, r, term, trunc, _ = env.step(a)
        if term or trunc:
            s, _ = env.reset()

    observations = np.vstack(observations)
    actions = np.hstack(actions)

    return observations, actions


# Should be better to save all data (policy, video, datase) in the same directory, I'll add that later
def record_episode(env_name, policy):
    expert = ALGOS[algo].load(os.path.join(SAVING_DIR, policy))
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=SAVING_DIR, name_prefix="expert", episode_trigger=lambda x: True)
    s, _ = env.reset()

    episode_over = False
    while not episode_over:
        a = expert.predict(s, deterministic=True)[0]
        s, r, term, trunc, _ = env.step(a)

        episode_over = term or trunc
    env.close() 



def get_expert_dataset(algo, env_name, dataset_length):
    repo_id = f"sb3/{algo}-{env_name}"
    policy_file = f"{algo}-{env_name}.zip"

    # Get the expert policy on huggingface 
    checkpoint = load_from_hub(repo_id, policy_file, SAVING_DIR)

    # Get a trajectory of a trained agent
    observations, actions = rollout(env_name, policy=policy_file, num_steps=dataset_length)

    # Transform it to a dataset
    dataset = {}
    dataset['train_input'] = torch.from_numpy(observations)
    dataset['test_input'] = torch.from_numpy(observations)
    dataset['train_label'] = torch.from_numpy(actions)[:, None]
    dataset['test_label'] = torch.from_numpy(actions)[:, None]

    # Record video of 1 episode
    record_episode(env_name, policy=policy_file)
    
    return dataset

if __name__ == "__name__":
    algo = "ppo"
    env_name = "CartPole-v1"
    # Change to use num_episodes instead of num_steps
    dataset_length = 10_000

    dataset = get_expert_dataset(algo, env_name, dataset_length)

    print(dataset['train_input'].shape)