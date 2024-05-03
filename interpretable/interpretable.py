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

    def __init__(self, env_id: str, hidden_widths: Optional[Tuple[int]]=None):
        self.env = gym.make(env_id)
        if hidden_widths is None:
            hidden_widths = []
        observation_dim, self._observation_is_discrete = extract_dim(self.env.observation_space)
        action_dim, self._action_is_discrete = extract_dim(self.env.action_space)
        self.policy = KAN(width=[observation_dim, *hidden_widths, action_dim])
        self.symbolic_policy = None
        self.loss_fn = torch.nn.MSELoss() if not self._action_is_discrete else torch.nn.CrossEntropyLoss()

    def _regression_accuracy(self, prefix, dataset: Dict[str, torch.Tensor]):
        return lambda: torch.mean(((self.policy(dataset[prefix + '_input']) - dataset[prefix + '_label']) ** 2).float())

    def _classification_accuracy(self, prefix, dataset: Dict[str, torch.Tensor]):
        return lambda: torch.mean(
            (self.policy(dataset[prefix + '_input']).argmax(dim=-1) == dataset[prefix + '_label']).float())

    def train_from_dataset(self, dataset: Dict[str, torch.Tensor], steps: int = 20):
        if self._action_is_discrete:
            train_acc = self._classification_accuracy("train", dataset)
            test_acc = self._classification_accuracy("test", dataset)
        else:
            train_acc = self._regression_accuracy("train", dataset)
            test_acc = self._regression_accuracy("test", dataset)
        results = self.policy.train(dataset, opt="LBFGS", steps=steps, loss_fn=self.loss_fn, metrics=(train_acc, test_acc))

        return results['train_acc'], results['test_acc']

    def plot(self):
        self.policy.prune()
        self.policy.plot(mask=True)

    def train_from_policy(self, policy: Callable[[np.ndarray], Union[np.ndarray, int, float]], steps: int):
        raise NotImplementedError()  # TODO
