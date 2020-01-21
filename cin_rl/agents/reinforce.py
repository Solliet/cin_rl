
from base import PolicyGradient

from torch.distributions import Categorical
from operator import attrgetter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D
from scipy.signal import lfilter
from torchvision import transforms

def discount(x, gamma): return lfilter(
    [1], [1, -gamma], x[::-1])[::-1]


sar_getter = attrgetter('obs', 'action', 'reward')


class Reinforce(nn.Module):
    def __init__(self, model, gamma):
        super(Reinforce, self).__init__()
        self.model = model
        self.gamma = gamma
        pass

    def __call__(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        single_action = False
        if len(obs.shape) == 1:
            obs.unsqueeze_(0)
            single_action = True

        probs = F.softmax(self.model(obs), dim=1)

        distribution = D.Categorical(probs=probs)
        action = distribution.sample()
        logits = distribution.log_prob(action)
        if single_action:
            return action.item(), logits
        return action.numpy(), logits

    def actiondist(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        single_action = False
        if len(obs.shape) == 1:
            obs.unsqueeze_(0)
            single_action = True

        return D.Categorical(logits=F.log_softmax(self.model(obs), dim=1))

    def logprobs(self, obs, actions, returnd=False):
        dist = self.actiondist(obs)
        if isinstance(actions, np.ndarray):
            logits = dist.log_prob(torch.from_numpy(actions))
        elif isinstance(actions, list):
            logits = dist.log_prob(torch.tensor(actions))
        elif isinstance(actions, torch.Tensor):
            logits = dist.log_prob(actions)
        else:
            raise ValueError(f'Unknown type {type(actions)}')
        return logits, dist

    def loss(self, trajectory):
        obs, action, rewards = sar_getter(trajectory)

        rewards = torch.from_numpy(np.array(discount(rewards, self.gamma)))
        rewards = (rewards-rewards.mean())/(rewards.std()+1e-8)
        logprobs, dist = self.logprobs(obs, action)
        
        loss = (-rewards*logprobs).sum() + dist.entropy().mean()
        return loss
