# Adapted from https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py

import argparse
from itertools import count
from collections import deque
from tqdm import tqdm
from functools import partial
import time
import os
import random
from collections import deque

import hydra
from omegaconf import DictConfig

import gymnasium as gym
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions.normal import Normal
from hydra.core.hydra_config import HydraConfig
from kan import KAN
from efficient_kan import EfficientKAN
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from utils.mpi_tools import mpi_fork, mpi_statistics_scalar, num_procs, proc_id
from utils.mpi_torch import (
    average_gradients,
    setup_pytorch_for_mpi,
    sync_all_params,
)


def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

class Policy(nn.Module):
    def __init__(self, config, env):
        super(Policy, self).__init__()
        self.mlp = False
        self.efficient = False

        if config.method == "KAN":
            self.network = KAN(
                width=[env.observation_space.shape[0], config.width, env.action_space.n],
                grid=config.grid,
                k=3,
                bias_trainable=False,
                sp_trainable=False,
                sb_trainable=False,
            )
        elif config.method == "EfficientKAN":
            self.efficient = True
            self.network = EfficientKAN(
                layers_hidden=[env.observation_space.shape[0], config.width, env.action_space.n],
            )
        elif config.method == "MLP":
            self.mlp = True
            self.affine1 = nn.Linear(4, config.width, dtype=torch.float32)
            self.dropout = nn.Dropout(p=0.6)
            self.affine2 = nn.Linear(config.width, 2, dtype=torch.float32)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        if self.mlp:
            x = self.affine1(x)
            x = self.dropout(x)
            x = F.relu(x)
            action_scores = self.affine2(x)
        else:
            if self.efficient: x = x.double()
            action_scores = self.network(x)
        return F.softmax(action_scores, dim=1)


def select_action(state, policy, env):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode(config, policy, optimizer, eps):
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + config.gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


@hydra.main(config_path=".", config_name="config-reinforce", version_base=None)
def main(config: DictConfig):
    env = gym.make('CartPole-v1')
    env.reset(seed=config.seed)
    torch.manual_seed(config.seed)

    policy = Policy(config, env)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    eps = np.finfo(np.float32).eps.item()

    run_name = f"{config.method}_{config.env_id}_{config.seed}_{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")

    os.makedirs("results", exist_ok=True)
    with open(f"results/{run_name}.csv", "w") as f:
        f.write("episode,length\n")

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )

    pbar_position = 0 if HydraConfig.get().mode == HydraConfig.get().mode.RUN else HydraConfig.get().job.num

    running_reward = 10
    for i_episode in tqdm(count(1), desc=f"{run_name}", position=pbar_position):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state, policy, env)
            state, reward, done, _, _ = env.step(action)
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                with open(f"results/{run_name}.csv", "a") as f:
                    f.write(f"{i_episode},{t}\n")
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode(config, policy, optimizer, eps)
        if i_episode % config.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
