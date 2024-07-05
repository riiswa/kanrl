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

from utils.utils import set_all_seeds
from utils.networks import initialize_network, reg

class Agent(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.critic = initialize_network(
            input_size=env.observation_space.shape[0],
            output_size=1,
            **config.critic
            )
        self.actor = initialize_network(
            input_size=env.observation_space.shape[0],
            output_size=env.action_space.n,
            **config.actor)
        
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x):
        x = x.unsqueeze(0).float()
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()

        return action.item(), probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def get_batch_actions_and_values(self, x, actions):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return actions, probs.log_prob(actions), probs.entropy(), self.critic(x)
    

@hydra.main(config_path="../conf", config_name="ppo_config", version_base=None)
def main(config: DictConfig):
    set_all_seeds(config.seed)
    
    print(config)
    print(config.actor)
    print(config.critic)
    
    actor_name = f"actor_{config.actor.width}_{config.actor.method}"
    critic_name = f"critic_{config.critic.width}_{config.critic.method}"
    run_name = f"PPO_{actor_name}_{critic_name}_{config.env_name}_{config.seed}_{int(time.time())}"

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
    print(agent.actor)
    print(agent.critic)
    optimizer = Adam(agent.parameters(), config.learning_rate)

    # storage setup
    obs = torch.zeros((config.batch_size,) + env.observation_space.shape)
    actions = torch.zeros((config.batch_size,) + env.action_space.shape)
    logprobs = torch.zeros((config.batch_size))
    rewards = torch.zeros((config.batch_size))
    dones = torch.zeros((config.batch_size))
    values = torch.zeros((config.batch_size))

    # start the training procedure
    n_steps = 0
    n_rollouts = config.training_steps // config.batch_size
    mini_batch_size = config.batch_size // config.num_minibatches
    start_time = time.time()
    next_obs, _ = env.reset()
    next_obs = torch.Tensor(next_obs)
    next_done = torch.zeros(1)

    # logging metrics for performance
    ep_rews = [] 
    ep_lens = []
    ep_returns = []

    pbar_position = 0 if HydraConfig.get().mode == HydraConfig.get().mode.RUN else HydraConfig.get().job.num
    for rollout_id in tqdm(range(n_rollouts), desc=f"{run_name}", position=pbar_position):
        if config.anneal_lr:
            frac = 1.0 - (rollout_id - 1.0) / n_rollouts
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, config.batch_size):
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, truncated, _ = env.step(action)
            next_done = np.logical_or(done, truncated)
            rewards[step] = torch.tensor(reward)
            
            n_steps += 1
            ep_rews.append(reward)

            # Check if episode is done
            if next_done:
                ep_ret, ep_len = np.sum(ep_rews), len(ep_rews)
                ep_returns.append(ep_ret)
                ep_lens.append(ep_len)
                next_obs, _ = env.reset()
                ep_rews = [] 

            # convert back the numpy elements to torch to pass them to the actor and critic networks 
            next_obs, next_done = torch.Tensor(next_obs), torch.tensor(np.int8(next_done))

            # Log metrics if log_interval
            if n_steps % config.log_interval == 0:
                
                mean_ep_return = np.mean(ep_returns)
                mean_ep_len = np.mean(ep_lens)

                ep_returns = []
                ep_lens = []

                writer.add_scalar('return', mean_ep_return, n_steps)
                writer.add_scalar("length", mean_ep_len, n_steps)

                with open(f"results/{run_name}.csv", "a") as f:
                    f.write(f"{n_steps},{mean_ep_return}\n")

        # TODO : Check how the returns are computed 
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0).float())
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(config.batch_size)):
                if t == config.batch_size - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_inds = np.arange(config.batch_size)
        clip_fracs = []

        # Update the actor and the critic networks 
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_batch_actions_and_values(obs[mb_inds], actions[mb_inds])
                logratio = newlogprob - logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                mb_advantages = advantages[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_advantages * ratio 
                pg_loss2 = - mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                    v_clipped = values[mb_inds] + torch.clamp(
                        newvalue - values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef
                    ) 
                    v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_clipped, v_loss_unclipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                optimizer.zero_grad()

                # if config.method == "KAN":
                #     # Should we only add reg term on critic here ? 
                #     reg_ = reg(net=agent.critic)
                #     loss += config.lamb * reg_

                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

        y_pred, y_true = values.numpy(), returns.numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], n_steps)
        writer.add_scalar("losses/value_loss", v_loss.item(), n_steps)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), n_steps)
        writer.add_scalar("losses/entropy", entropy_loss.item(), n_steps)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), n_steps)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), n_steps)
        writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), n_steps)
        writer.add_scalar("losses/explained_variance", explained_var, n_steps)
        writer.add_scalar("charts/SPS", int(n_steps / (time.time() - start_time)), n_steps)

    end_time = time.time()
    print(f"\nFinal results - training_steps: {n_steps} - return: {mean_ep_return:.3f}")
    print(f"Training time : {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()  