import os
import time

import hydra
import torch
import torch.nn as nn
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter
from hydra.core.hydra_config import HydraConfig

from buffer import ReplayBuffer
from omegaconf import DictConfig
from tqdm import tqdm

from utils import set_all_seeds
from networks import initialize_network, reg

def compute_loss(
    net,
    target,
    data,
    gamma=0.99,
):
    observations, actions, next_observations, rewards, terminations = data

    with torch.no_grad():
        next_q_values = net(next_observations)
        next_actions = next_q_values.argmax(dim=1)
        next_q_values_target = target(next_observations)
        target_max = next_q_values_target[range(len(next_q_values)), next_actions]
        td_target = rewards.flatten() + gamma * target_max * (
            1 - terminations.flatten()
        )

    old_val = net(observations).gather(1, actions).squeeze()
    loss = nn.functional.mse_loss(td_target, old_val)
    return loss 


@hydra.main(config_path="conf", config_name="ddqn_config", version_base=None)
def main(config: DictConfig):
    set_all_seeds(config.seed)
    env = gym.make(config.env_id)

    # TODO : Might be a cleaner way to initialize the networks
    q_network = initialize_network(
        input_size=env.observation_space.shape[0],
        output_size=env.action_space.n,
        **config)
    target_network = initialize_network(
        input_size=env.observation_space.shape[0],
        output_size=env.action_space.n,
        **config)

    target_network.load_state_dict(q_network.state_dict())
    
    run_name = f"DDQN_{config.method}_{config.env_id}_{config.seed}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    os.makedirs("results", exist_ok=True)
    with open(f"results/{run_name}.csv", "w") as f:
        f.write("episode,length\n")

    optimizer = torch.optim.Adam(q_network.parameters(), config.learning_rate)
    buffer = ReplayBuffer(config.replay_buffer_capacity, env.observation_space.shape[0])

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )

    pbar_position = 0 if HydraConfig.get().mode == HydraConfig.get().mode.RUN else HydraConfig.get().job.num

    for episode in tqdm(range(config.n_episodes), desc=f"{run_name}", position=pbar_position):
        observation, info = env.reset()
        observation = torch.from_numpy(observation)
        finished = False
        episode_length = 0
        while not finished:
            if episode < config.warm_up_episodes:
                action = env.action_space.sample()
            else:
                action = (
                    # Changed .double to .float()
                    q_network(observation.unsqueeze(0).float())
                    .argmax(axis=-1)
                    .squeeze()
                    .item()
                )
            next_observation, reward, terminated, truncated, info = env.step(action)
            # TODO : Do we wanna keep this specific condition for CartPole env ? 
            if config.env_id == "CartPole-v1":
                reward = -1 if terminated else 0
            next_observation = torch.from_numpy(next_observation)

            buffer.add(observation, action, next_observation, reward, terminated)

            observation = next_observation
            finished = terminated or truncated
            episode_length += 1

        # When an episode is finished:
        with open(f"results/{run_name}.csv", "a") as f:
            f.write(f"{episode},{episode_length}\n")

        # TODO : Could maybe add all the code below in a train function ? 
        if len(buffer) >= config.batch_size:
            for _ in range(config.train_steps):
                loss = compute_loss(
                    q_network,
                    target_network,
                    buffer.sample(config.batch_size),
                    config.gamma,
                )
                
                if config.method == "KAN":
                    # TODO : should we enable specifying args for reg (lambs ...) ? Keep it like that ? Or just remove it ? 
                    reg_ = reg(net=q_network)
                    loss = loss + config.lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            writer.add_scalar("episode_length", episode_length, episode)
            writer.add_scalar("loss", loss, episode)
            # TODO : See how we should handle that specific update of KAN networks 
            if (
                episode % 25 == 0
                and config.method == "KAN"
                and episode < int(config.n_episodes * (1 / 2))
            ):
                q_network.update_grid_from_samples(buffer.observations[: len(buffer)])
                target_network.update_grid_from_samples(buffer.observations[: len(buffer)])

            if episode % config.target_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())


if __name__ == "__main__":
    main()
