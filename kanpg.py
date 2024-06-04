# Implementation of simple policy gradient
import os
import time
import random

import hydra
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from gymnasium import spaces 
from omegaconf import DictConfig
from kan import KAN

from utils.kan_utils import reg

# Train a policy on 'training_steps' steps in an environment, with updates every 'batch_size' steps 
def train_policy_gradient(
        method="mlp", 
        env_name='CartPole-v0', 
        lr=1e-2, 
        width=8,
        grid=5,
        training_steps=300_000, 
        batch_size=5000,
        lamb=0.0,
        lamb_l1=1.0,
        lamb_entropy=2.0,
        lamb_coef=0.0,
        lamb_coefdiff=0.0,
        small_mag_threshold=1e-16,
        small_reg_factor=1.0,
        seed=0):
    
    print(f"Training simple {method} policy on {env_name} for {training_steps} timesteps")
    run_name = f"Simple_PG_{method}_{env_name}_{seed}_{int(time.time())}"

    # prepare savings
    writer = SummaryWriter(f"runs/{run_name}")
    os.makedirs("results", exist_ok=True)
    with open(f"results/{run_name}.csv", "w") as f:
        f.write("timestep,avg_return\n")
    
    env = gym.make(env_name)
    assert isinstance(env.observation_space, spaces.Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, spaces.Discrete), \
        "This example only works for envs with discrete action spaces."
    
    if method == "MLP":
        logits_net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], width),
            nn.ReLU(),
            nn.Linear(width, env.action_space.n),
        )
    elif method == "KAN":
        logits_net = KAN(
            width=[env.observation_space.shape[0], width, env.action_space.n],
            grid=grid,
            k=3,
            bias_trainable=False,
            sp_trainable=False,
            sb_trainable=False,
        )
    else:
        raise(ValueError(f"Unknown method {method}"))
    

    def get_policy(obs):
        logits = logits_net(obs.unsqueeze(0).double()).squeeze()
        return Categorical(logits=logits)
    
    def get_batch_policy(obs):
        logits = logits_net(obs.double())
        return Categorical(logits=logits)
    
    def get_action(obs):
        action = get_policy(obs).sample().item()
        return action
    
    def reward_to_go(rewards):
        n = len(rewards)
        rtg = np.zeros_like(rewards)
        for i in reversed(range(n)):
            rtg[i] = rewards[i] + (rtg[i+1] if i+1 < n else 0)
        return rtg
    
    def compute_loss(obs, act, weights):
        logits = get_batch_policy(obs)
        logp = logits.log_prob(act) 
        return -(logp * weights).mean()
    
    optimizer = Adam(logits_net.parameters(), lr=lr)

    def train_one_epoch():
        # could maybe clean that by using Buffer class in buffer.py
        batch_obs = []          
        batch_acts = []        
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         

        obs, _ = env.reset()       
        done, truncated = False, False           
        ep_rews = []           

        while True:
            batch_obs.append(obs.copy())
            action = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, done, truncated, _ = env.step(action)
            batch_acts.append(action)
            ep_rews.append(reward)
            if done or truncated:
                # use a simple sum for the return (no reward decay)
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += list(reward_to_go(ep_rews))

                obs, _= env.reset()
                done, truncated, ep_rews = False, False, []
                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        
        # TODO : not really clean, see how to refactor this (add reg term to the loss if KAN)
        if method == "KAN":
            reg_ = reg(logits_net,
               lamb_l1=lamb_l1,
               lamb_entropy=lamb_entropy,
               lamb_coef=lamb_coef,
               lamb_coefdiff=lamb_coefdiff,
               small_mag_threshold=small_mag_threshold,
               small_reg_factor=small_reg_factor)
            loss = batch_loss + lamb * reg_
        else:
            loss = batch_loss

        loss.backward()
        optimizer.step()
        avg_return = np.mean(batch_rets)
        return avg_return, batch_lens

    # divide training process in n_epochs
    n_epochs = training_steps // batch_size

    # training loop
    n_steps = 0
    for i in range(n_epochs):
        avg_return, batch_lens = train_one_epoch()
        n_steps += batch_size
        # record results
        writer.add_scalar('return', avg_return, n_steps)
        writer.add_scalar('timestep', n_steps, n_steps)
        with open(f"results/{run_name}.csv", "a") as f:
            f.write(f"{n_steps},{avg_return}\n")
        if n_steps % 10 * batch_size == 0:
            print(f"training_steps: {n_steps} - return: {avg_return:.3f}")
            # TODO : should also maybe add logits_net.update_grid_from_samples

    print(f"\nFinal results - training_steps: {n_steps} - return: {avg_return:.3f}")


def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@hydra.main(config_path="conf", config_name="pg_config", version_base=None)
def main(config: DictConfig):
    set_all_seeds(config.seed)
    
    start = time.time()
    train_policy_gradient(method=config.method, 
                    env_name=config.env_name, 
                    training_steps=config.training_steps, 
                    batch_size=config.batch_size,
                    lr=config.learning_rate,
                    grid=config.grid,
                    width=config.width,
                    seed=config.seed
                    )
    end = time.time()

    # KAN ~ 30/40x slower than MLP with this config
    print(f"Training time : {end - start:.2f} seconds")

if __name__ == '__main__':
    main()  