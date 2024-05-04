import glob
import os
import pickle

import torch
import numpy as np
import gymnasium as gym
from huggingface_hub.utils import EntryNotFoundError
from huggingface_sb3 import load_from_hub
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip
from rl_zoo3 import ALGOS
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.running_mean_std import RunningMeanStd

import os
import tarfile
import urllib.request


def install_mujoco():
    mujoco_url = "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
    mujoco_file = "mujoco210-linux-x86_64.tar.gz"
    mujoco_dir = "mujoco210"

    # Check if the directory already exists
    if not os.path.exists("mujoco210"):
        # Download Mujoco if not exists
        print("Downloading Mujoco...")
        urllib.request.urlretrieve(mujoco_url, mujoco_file)

        # Extract Mujoco
        print("Extracting Mujoco...")
        with tarfile.open(mujoco_file, "r:gz") as tar:
            tar.extractall()

        # Clean up the downloaded tar file
        os.remove(mujoco_file)

        print("Mujoco installed successfully!")
    else:
        print("Mujoco already installed.")

    # Set environment variable MUJOCO_PY_MUJOCO_PATH
    os.environ["MUJOCO_PY_MUJOCO_PATH"] = os.path.abspath(mujoco_dir)

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    mujoco_bin_path = os.path.join(os.path.abspath(mujoco_dir), "bin")
    if mujoco_bin_path not in ld_library_path:
        os.environ["LD_LIBRARY_PATH"] = ld_library_path + ":" + mujoco_bin_path



class NormalizeObservation(gym.Wrapper):
    def __init__(self, env: gym.Env, clip_obs: float, obs_rms: RunningMeanStd, epsilon: float):
        gym.Wrapper.__init__(self, env)
        self.clip_obs = clip_obs
        self.obs_rms = obs_rms
        self.epsilon = epsilon

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self.normalize(np.array([observation]))[0]
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self.normalize(np.array([observation]))[0], info

    def normalize(self, obs):
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)


class CreateDataset(gym.Wrapper):
    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.observations = []
        self.actions = []
        self.last_observation = None

    def step(self, action):
        self.observations.append(self.last_observation)
        self.actions.append(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.last_observation = observation
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.last_observation = observation
        return observation, info

    def get_dataset(self):
        if isinstance(self.env.action_space, gym.spaces.Box) and self.env.action_space.shape != (1,):
            actions = np.vstack(self.actions)
        else:
            actions = np.hstack(self.actions)
        return np.vstack(self.observations), actions


def rollouts(env, policy, num_episodes=1):
    for episode in range(num_episodes):
        done = False
        observation, _ = env.reset()
        while not done:
            action = policy(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()


def generate_dataset_from_expert(algo, env_name, num_train_episodes=5, num_test_episodes=2, force=False):
    if env_name.startswith("Swimmer") or env_name.startswith("Hopper"):
        install_mujoco()
    dataset_path = os.path.join("datasets", f"{algo}-{env_name}.pt")
    video_path = os.path.join("videos", f"{algo}-{env_name}.mp4")
    if os.path.exists(dataset_path) and os.path.exists(video_path) and not force:
        return dataset_path, video_path
    repo_id = f"sb3/{algo}-{env_name}"
    policy_file = f"{algo}-{env_name}.zip"

    expert_path = load_from_hub(repo_id, policy_file)
    try:
        vec_normalize_path = load_from_hub(repo_id, "vec_normalize.pkl")
        with open(vec_normalize_path, "rb") as f:
            vec_normalize = pickle.load(f)
            if vec_normalize.norm_obs:
                vec_normalize_params = {"clip_obs": vec_normalize.clip_obs, "obs_rms": vec_normalize.obs_rms, "epsilon": vec_normalize.epsilon}
            else:
                vec_normalize_params = None
    except EntryNotFoundError:
        vec_normalize_params = None

    expert = ALGOS[algo].load(expert_path)
    train_env = gym.make(env_name)
    train_env = CreateDataset(train_env)
    if vec_normalize_params is not None:
        train_env = NormalizeObservation(train_env, **vec_normalize_params)
    test_env = gym.make(env_name, render_mode="rgb_array")
    test_env = CreateDataset(test_env)
    if vec_normalize_params is not None:
        test_env = NormalizeObservation(test_env, **vec_normalize_params)
    test_env = RecordVideo(test_env, video_folder="videos", episode_trigger=lambda x: True, name_prefix=f"{algo}-{env_name}")

    def policy(obs):
        return expert.predict(obs, deterministic=True)[0]

    os.makedirs("videos", exist_ok=True)
    rollouts(train_env, policy, num_train_episodes)
    rollouts(test_env, policy, num_test_episodes)

    train_observations, train_actions = train_env.get_dataset()
    test_observations, test_actions = test_env.get_dataset()

    dataset = {
        "train_input": torch.from_numpy(train_observations),
        "test_input": torch.from_numpy(test_observations),
        "train_label": torch.from_numpy(train_actions),
        "test_label": torch.from_numpy(test_actions)
    }

    os.makedirs("datasets", exist_ok=True)
    torch.save(dataset, dataset_path)

    video_files = glob.glob(os.path.join("videos", f"{algo}-{env_name}-episode*.mp4"))
    clips = [VideoFileClip(file) for file in video_files]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(video_path, codec="libx264", fps=24)

    return dataset_path, video_path


if __name__ == "__main__":
    generate_dataset_from_expert("ppo", "CartPole-v1", force=True)
