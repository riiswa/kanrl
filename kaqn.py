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


class Agent(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.q_network = initialize_network(
            input_size=env.observation_space.shape[0],
            output_size=env.action_space.n,
            **config
        )
        self.target_network = initialize_network(
            input_size=env.observation_space.shape[0],
            output_size=env.action_space.n,
            **config
        )

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
                
    def update_grid_from_samples(self, observations):
         self.q_network.update_grid_from_samples(observations)
         self.target_network.update_grid_from_samples(observations)


@hydra.main(config_path="conf", config_name="ddqn_config", version_base=None)
def main(config: DictConfig):
    set_all_seeds(config.seed)
    env = gym.make(config.env_id)

    # Create a ddqn agent
    agent = Agent(env, config)
    agent.update_target()

    run_name = f"DDQN_{config.method}_{config.env_id}_{config.seed}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    os.makedirs("results", exist_ok=True)
    with open(f"results/{run_name}.csv", "w") as f:
        f.write("episode,length\n")

    optimizer = torch.optim.Adam(agent.q_network.parameters(), config.learning_rate)
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
                    agent.q_network(observation.unsqueeze(0).float())
                    .argmax(axis=-1)
                    .squeeze()
                    .item()
                )
            next_observation, reward, terminated, truncated, info = env.step(action)
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

        if len(buffer) >= config.batch_size:
            for _ in range(config.train_steps):
                # Compute the loss
                observations, actions, next_observations, rewards, terminations = buffer.sample(config.batch_size)

                with torch.no_grad():
                        next_q_values = agent.q_network(next_observations)
                        next_actions = next_q_values.argmax(dim=1)
                        next_q_values_target = agent.target_network(next_observations)
                        target_max = next_q_values_target[range(len(next_q_values)), next_actions]
                        td_target = rewards.flatten() + config.gamma * target_max * (1 - terminations.flatten())

                old_val = agent.q_network(observations).gather(1, actions).squeeze()
                loss = nn.functional.mse_loss(td_target, old_val)

                # Add additional reg term to the loss with KANs
                if config.method == "KAN":
                    reg_ = reg(net=agent.q_network)
                    loss += config.lamb * reg_

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            writer.add_scalar("episode_length", episode_length, episode)
            writer.add_scalar("loss", loss, episode)

            if episode % 25 == 0 and config.method == "KAN" and episode < int(config.n_episodes * (1 / 2)):
                agent.update_grid_from_samples(buffer.observations[: len(buffer)])

            if episode % config.target_update_freq == 0:
                agent.update_target()

if __name__ == "__main__":
    main()
