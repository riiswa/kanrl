import torch
from typing import Dict, Tuple, Optional, Callable, Union
import gymnasium as gym
from kan import KAN
import numpy as np


def extract_dim(space: gym.Space):
    if isinstance(space, gym.spaces.Box) and len(space.shape) == 1:
        return space.shape[0], False
    elif isinstance(space, gym.spaces.Discrete):
        return space.n, True
    else:
        raise NotImplementedError(f"There is no support for space {space}.")


class InterpretablePolicyExtractor:
    lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']

    def __init__(self, env_name: str, hidden_widths: Optional[Tuple[int]]=None):
        self.env = gym.make(env_name)
        if hidden_widths is None:
            hidden_widths = []
        observation_dim, self._observation_is_discrete = extract_dim(self.env.observation_space)
        action_dim, self._action_is_discrete = extract_dim(self.env.action_space)
        self.policy = KAN(width=[observation_dim, *hidden_widths, action_dim])
        self.loss_fn = torch.nn.MSELoss() if not self._action_is_discrete else torch.nn.CrossEntropyLoss()

    def train_from_dataset(self, dataset: Union[Dict[str, torch.Tensor], str], steps: int = 20):
        if isinstance(dataset, str):
            dataset = torch.load(dataset)
        if dataset["train_label"].ndim == 1 and not self._action_is_discrete:
            dataset["train_label"] = dataset["train_label"][:, None]
        if dataset["train_label"].ndim == 1 and not self._action_is_discrete:
            dataset["test_label"] = dataset["test_label"][:, None]
        return self.policy.train(dataset, opt="LBFGS", steps=steps, loss_fn=self.loss_fn)

    def forward(self, observation):
        observation = torch.from_numpy(observation)
        action = self.policy(observation.unsqueeze(0))
        if self._action_is_discrete:
            return action.argmax(axis=-1).squeeze().item()
        else:
            return action.squeeze(0).detach().numpy()

    def train_from_policy(self, policy: Callable[[np.ndarray], Union[np.ndarray, int, float]], steps: int):
        raise NotImplementedError()  # TODO
