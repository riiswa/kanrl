# Implementation of a really simple policy gradient algorithm
import os
import time

import hydra
import numpy as np
import torch
import torch.nn as nn 
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from gymnasium import spaces 
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

from utils import set_all_seeds
from networks import initialize_network, reg


class Agent(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.logits_net = initialize_network(
            input_size=env.observation_space.shape[0],
            output_size=env.action_space.n,
            **config
        )
        
    def get_policy(self, obs):
        logits = self.logits_net(obs.unsqueeze(0).float()).squeeze()
        return Categorical(logits=logits)

    def get_batch_policy(self, obs):
        logits = self.logits_net(obs.float())
        return Categorical(logits=logits)

    def get_action(self, obs):
        action = self.get_policy(obs).sample().item()
        return action


@hydra.main(config_path="conf", config_name="pg_config", version_base=None)
def main(config: DictConfig):
    set_all_seeds(config.seed)
    
    start = time.time()

    run_name = f"Simple_PG_{config.method}_{config.env_name}_{config.seed}_{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    os.makedirs("results", exist_ok=True)
    with open(f"results/{run_name}.csv", "w") as f:
        f.write("timestep,avg_return\n")
    
    env = gym.make(config.env_name)
    assert isinstance(env.observation_space, spaces.Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, spaces.Discrete), \
        "This example only works for envs with discrete action spaces."
    
    agent = Agent(env, config)
    
    optimizer = Adam(agent.parameters(), config.learning_rate)
    
    # divide training process in n_rollouts
    n_rollouts = config.training_steps // config.batch_size

    # training loop
    n_steps = 0

    pbar_position = 0 if HydraConfig.get().mode == HydraConfig.get().mode.RUN else HydraConfig.get().job.num
    for rollout in tqdm(range(n_rollouts), desc=f"{run_name}", position=pbar_position):
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
            action = agent.get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, done, truncated, _ = env.step(action)
            batch_acts.append(action)
            ep_rews.append(reward)
            if done or truncated:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # Implement reward to go : the weight for each logprob(a|s) is R(tau)
                n = len(ep_rews)
                rtg = np.zeros_like(ep_rews)
                for i in reversed(range(n)):
                    rtg[i] = ep_rews[i] + (rtg[i+1] if i+1 < n else 0)

                batch_weights += list(rtg)

                obs, _= env.reset()
                done, truncated, ep_rews = False, False, []
                if len(batch_obs) > config.batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()

        # Compute loss 
        logits = agent.get_batch_policy(torch.as_tensor(batch_obs, dtype=torch.float32))
        log_p = logits.log_prob(torch.as_tensor(batch_acts, dtype=torch.int32))
        loss = -(log_p * torch.as_tensor(batch_weights, dtype=torch.float32)).mean()
        
        # Add regularization term if using KAN
        if config.method == "KAN":
            reg_ = reg(net=agent.logits_net)
            loss += config.lamb * reg_

        loss.backward()
        optimizer.step()
        avg_return = np.mean(batch_rets)
        
        n_steps += config.batch_size

        # record results
        writer.add_scalar('return', avg_return, n_steps)
        writer.add_scalar('timestep', n_steps, n_steps)
        with open(f"results/{run_name}.csv", "a") as f:
            f.write(f"{n_steps},{avg_return}\n")

    end = time.time()
    # KAN ~ 30/40x slower than MLP with this config
    print(f"\nFinal results - training_steps: {n_steps} - return: {avg_return:.3f}")
    print(f"Training time : {end - start:.2f} seconds")

if __name__ == '__main__':
    main()  