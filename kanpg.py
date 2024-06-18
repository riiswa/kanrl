# Implementation of a really simple policy gradient algorithm
import os
import time

import hydra
import numpy as np
import torch
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


def get_policy(logits_net, obs):
    logits = logits_net(obs.unsqueeze(0).float()).squeeze()
    return Categorical(logits=logits)

def get_batch_policy(logits_net, obs):
    logits = logits_net(obs.float())
    return Categorical(logits=logits)

def get_action(logits_net, obs):
    action = get_policy(logits_net, obs).sample().item()
    return action

def reward_to_go(rewards):
    n = len(rewards)
    rtg = np.zeros_like(rewards)
    for i in reversed(range(n)):
        rtg[i] = rewards[i] + (rtg[i+1] if i+1 < n else 0)
    return rtg

def compute_loss(logits_net, obs, act, weights):
    logits = get_batch_policy(logits_net, obs)
    logp = logits.log_prob(act) 
    return -(logp * weights).mean()


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
    
    logits_net = initialize_network(
        input_size=env.observation_space.shape[0],
        output_size=env.action_space.n,
        **config)
    
    optimizer = Adam(logits_net.parameters(), config.learning_rate)

    def train_one_epoch():
        # TODO : could maybe clean that by using Buffer class in buffer.py
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
            action = get_action(logits_net, torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, done, truncated, _ = env.step(action)
            batch_acts.append(action)
            ep_rews.append(reward)
            if done or truncated:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += list(reward_to_go(ep_rews))

                obs, _= env.reset()
                done, truncated, ep_rews = False, False, []
                if len(batch_obs) > config.batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        loss = compute_loss(
            logits_net,
            obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=torch.int32),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32)
        )
        
        if config.method == "KAN":
            reg_ = reg(net=logits_net)
            loss += config.lamb * reg_

        loss.backward()
        optimizer.step()
        avg_return = np.mean(batch_rets)

        return avg_return, batch_lens
    
    # divide training process in n_epochs
    n_epochs = config.training_steps // config.batch_size

    # training loop
    n_steps = 0

    pbar_position = 0 if HydraConfig.get().mode == HydraConfig.get().mode.RUN else HydraConfig.get().job.num
    for epoch in tqdm(range(n_epochs), desc=f"{run_name}", position=pbar_position):
    # for i in range(n_epochs):
        avg_return, batch_lens = train_one_epoch()
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