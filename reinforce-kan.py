# Adapted from https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py
# also https://github.com/CppMaster/SC2-AI/blob/master/minigames/move_to_beacon/src/optuna_search.py
# and https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py
# and https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/2_rtg_pg.py

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

# from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.policies import BaseModel #BasePolicy
# from stable_baselines3.ppo.policies import MlpPolicy

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate


def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

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
            # print("HELLOW")
            self.mlp = True
            # self.network = mlp([env.observation_space.shape[0], config.width, env.action_space.n])
            # print(self.network)
            self.affine1 = nn.Linear(env.observation_space.shape[0], config.width)
            self.dropout = nn.Dropout(p=0.6)
            self.affine2 = nn.Linear(config.width, env.action_space.n)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        if self.mlp:
            x = x.double()
            # print(f"x.dtype is {x.dtype}")
            # print(f"AFFINE ONE DTYPE is {self.affine1.weight.dtype}")
            x = self.affine1(x)
            x = self.dropout(x)
            x = F.relu(x)
            action_scores = self.affine2(x)
            # action_scores = self.network(x)
        else:
            if self.efficient: 
                x = x.double()
            action_scores = self.network(x)
        return F.softmax(action_scores, dim=1)


def select_action(state, policy, env):
    state = torch.from_numpy(state).float().unsqueeze(0)
    # print(f"SELECT ACTION STATE DTYPE IS {state.dtype}")
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode(config, policy, optimizer, eps):
    R = 0
    policy_loss = []
    returns = deque()
    if config.rtg == True:
        returns = reward_to_go(policy.rewards)
    else:
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
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)
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
    for i_episode in tqdm(range(config.n_episodes), desc=f"{run_name}", position=pbar_position):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            # print(f"STATE DTYPE {state.dtype}")
            action = select_action(state, policy, env)
            state, reward, done, _, _ = env.step(action)
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                with open(f"results/{run_name}.csv", "a") as f:
                    f.write(f"{i_episode},{t}\n")
                break

        running_reward = config.episode_discount * ep_reward + (1 - config.episode_discount) * running_reward
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
    # sampler = TPESampler(n_startup_trials=10, multivariate=True)
    # pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=10)
    # study = optuna.create_study(
    #     sampler=sampler,
    #     pruner=pruner,
    #     load_if_exists=True,
    #     direction="maximize",
    # )

    # try:
    #     study.optimize(main, n_jobs=4, n_trials=128)
    # except KeyboardInterrupt:
    #     pass

    # print("Number of finished trials: ", len(study.trials))

    # trial = study.best_trial
    # print(f"Best trial: {trial.number}")
    # print("Value: ", trial.value)

    # print("Params: ")
    # for key, value in trial.params.items():
    #     print(f"    {key}: {value}")

    # study.trials_dataframe().to_csv(f"{study_path}/report.csv")

    # with open(f"{study_path}/study.pkl", "wb+") as f:
    #     pkl.dump(study, f)

    # try:
    #     fig1 = plot_optimization_history(study)
    #     fig2 = plot_param_importances(study)
    #     fig3 = plot_parallel_coordinate(study)

    #     fig1.show()
    #     fig2.show()
    #     fig3.show()

    # except (ValueError, ImportError, RuntimeError) as e:
    #     print("Error during plotting")
    #     print(e)
