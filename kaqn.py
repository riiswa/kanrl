import argparse
from typing import Literal

import gymnasium as gym
import torch
import torch.nn as nn
from kan import KAN
from torch.utils.tensorboard import SummaryWriter

from buffer import ReplayBuffer


def kan_train(
    net,
    target,
    data,
    optimizer,
    gamma=0.99,
    lamb=0.0,
    lamb_l1=1.0,
    lamb_entropy=2.0,
    lamb_coef=0.0,
    lamb_coefdiff=0.0,
    small_mag_threshold=1e-16,
    small_reg_factor=1.0,
):
    def reg(acts_scale):
        def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
            return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

        reg_ = 0.0
        for i in range(len(acts_scale)):
            vec = acts_scale[i].reshape(
                -1,
            )

            p = vec / torch.sum(vec)
            l1 = torch.sum(nonlinear(vec))
            entropy = -torch.sum(p * torch.log2(p + 1e-4))
            reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

        # regularize coefficient to encourage spline to be zero
        for i in range(len(net.act_fun)):
            coeff_l1 = torch.sum(torch.mean(torch.abs(net.act_fun[i].coef), dim=1))
            coeff_diff_l1 = torch.sum(
                torch.mean(torch.abs(torch.diff(net.act_fun[i].coef)), dim=1)
            )
            reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

        return reg_

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
    reg_ = reg(net.acts_scale)
    loss = loss + lamb * reg_
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def mlp_train(
    net,
    target,
    data,
    optimizer,
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
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for training Kolmogorov-Arnold Q-Network (KAQN) agent on CartPole-v1"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=500, help="Number of episodes for training"
    )
    parser.add_argument(
        "--warm_up_episodes", type=int, default=50, help="Number of warm-up episodes"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for future rewards"
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=5,
        help="Number of training steps per episode",
    )
    parser.add_argument(
        "--target_update_freq",
        type=int,
        default=10,
        help="Frequency of updating target network",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--replay_buffer_capacity",
        type=int,
        default=10000,
        help="Capacity of the replay buffer",
    )
    parser.add_argument(
        "--width", type=int, default=5, help="KAN width of the hidden layer"
    )
    parser.add_argument("--grid", type=int, default=3, help="KAN grid hyperparameter")
    parser.add_argument("--method", type=str, default="KAN", help="Wether to use MLP or KAN")
    args = parser.parse_args()

    env = gym.make("CartPole-v1")
    if args.method == "KAN":
        q_network = KAN(
            width=[env.observation_space.shape[0], args.width, env.action_space.n],
            grid=args.grid,
            k=3,
            # bias_trainable=False,
            # sp_trainable=False,
            # sb_trainable=False,
        )
        target_network = KAN(
            width=[env.observation_space.shape[0], args.width, env.action_space.n],
            grid=args.grid,
            k=3,
            # bias_trainable=False,
            # sp_trainable=False,
            # sb_trainable=False,
        )
        train = kan_train
    elif args.method == "MLP":
        q_network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, env.action_space.n)
        )
        target_network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, env.action_space.n)
        )
        train = mlp_train
    else:
        raise Exception(f"Method {args.method} don't exist, choose between MLP and KAN." )

    target_network.load_state_dict(q_network.state_dict())

    writer = SummaryWriter()

    optimizer = torch.optim.Adam(q_network.parameters(), args.learning_rate)
    buffer = ReplayBuffer(args.replay_buffer_capacity, env.observation_space.shape[0])

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    for episode in range(args.n_episodes):
        observation, info = env.reset()
        observation = torch.from_numpy(observation)
        finished = False
        episode_length = 0
        while not finished:
            if episode < args.warm_up_episodes:
                action = env.action_space.sample()
            else:
                action = (
                    q_network(observation.unsqueeze(0).double())
                    .argmax(axis=-1)
                    .squeeze()
                    .item()
                )
            next_observation, reward, terminated, truncated, info = env.step(action)
            reward = -1 if terminated else 0
            next_observation = torch.from_numpy(next_observation)

            buffer.add(observation, action, next_observation, reward, terminated)

            observation = next_observation
            finished = terminated or truncated
            episode_length += 1
        if len(buffer) >= args.batch_size:
            for _ in range(args.train_steps):
                loss = train(
                    q_network,
                    target_network,
                    buffer.sample(args.batch_size),
                    optimizer,
                    args.gamma,
                )
            print(f"Episode: {episode}, Loss: {loss}, Episode Length: {episode_length}")
            writer.add_scalar("episode_length", episode_length, episode)
            writer.add_scalar("loss", loss, episode)
            if episode % 25 == 0 and args.method == "KAN":
                q_network.update_grid_from_samples(buffer.observations[:len(buffer)])

            if episode % args.target_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())
