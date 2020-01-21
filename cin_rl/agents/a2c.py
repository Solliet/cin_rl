
from base import PolicyGradient

from torch.distributions import Categorical
from operator import attrgetter

import numpy as np
import torch
from torch import nn
from torch import from_numpy as np_to_torch
from torch.nn import functional as F
from torch import distributions as D
from scipy.signal import lfilter
from collections import namedtuple

def discount(x, gamma): return lfilter(
    [1], [1, -gamma], x[::-1])[::-1]


Experience = namedtuple(
    "Experience", ["obs", "action", "reward", "newobs", "done"])
sarsd_getter = attrgetter("obs", "action", "reward", "newobs", "done")


class A2C(nn.Module):
    def __init__(self, model, gamma, lambd, beta):
        super(A2C, self).__init__()
        self.model = model
        self.gamma = gamma
        self.lambd = lambd
        self.beta = beta
        pass

    def __call__(self, obs):
        if isinstance(obs, np.ndarray):
            obs = np_to_t(obs).float()

        single_action = False
        if len(obs.shape) == 1:
            obs.unsqueeze_(0)
            single_action = True
        _, probs = self.model(obs)
        probs = F.softmax(probs, dim=1)

        distribution = D.Categorical(probs=probs)
        action = distribution.sample()
        logits = distribution.log_prob(action)
        if single_action:
            return action.item(), logits
        return action.numpy(), logits

    def estimate(self, obs):
        if isinstance(obs, np.ndarray):
            obs = np_to_torch(obs).float()

        single_action = False
        if len(obs.shape) == 1:
            obs.unsqueeze_(0)
            single_action = True
        v, _ = self.model(obs)
        return v

    def actiondist(self, obs):
        if isinstance(obs, np.ndarray):
            obs = np_to_torch(obs).float()

        single_action = False
        if len(obs.shape) == 1:
            obs.unsqueeze_(0)
            single_action = True

        return D.Categorical(logits=F.log_softmax(self.model(obs)[1], dim=1))

    def logprobs(self, obs, actions, returnd=False):
        dist = self.actiondist(obs)
        logits = None
        if isinstance(actions, np.ndarray):
            logits = dist.log_prob(np_to_torch(actions))
        elif isinstance(actions, list):
            logits = dist.log_prob(torch.tensor(actions))
        elif isinstance(actions, torch.Tensor):
            logits = dist.log_prob(actions)
        else:
            raise ValueError(f'Unknown type {type(actions)}')
        return logits, dist

    def loss(self, trajectories):
        if not isinstance(trajectories, list):
            trajectories = [trajectories]

        def internal(trajectory):

            obs, action, rewards, nobs, done = sarsd_getter(trajectory)

            values = self.estimate(obs).squeeze()
            with torch.no_grad():
                advantage = self.gamma * \
                    self.estimate(nobs).squeeze() * torch.from_numpy(done==0.0) + \
                    torch.from_numpy(rewards) - values.detach()
                
                advantage = discount(advantage.numpy(), self.gamma*self.lambd)
                advantage = torch.from_numpy(advantage.copy())
                advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)

            rewards = torch.from_numpy(
                np.array(discount(rewards, self.gamma*self.lambd)))
            rewards = (rewards-rewards.mean())/(rewards.std()+1e-8)
            value_loss = F.mse_loss(rewards, values)

            logprobs, dist = self.logprobs(obs, action)
            policy_loss = (-advantage*logprobs).sum()
            entropy_loss = - dist.entropy().sum()
            return policy_loss, value_loss, entropy_loss

        policyl, criticl, entropyl = zip(*[internal(traj) for traj in trajectories])
        policyl = torch.stack(policyl).sum()
        criticl = torch.stack(criticl).sum()
        entropyl = self.beta* torch.stack(entropyl).mean()

        return  policyl + criticl + entropyl 