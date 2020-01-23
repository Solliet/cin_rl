

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


def A2C(ac, beta, gamma, lambd):

    def A2C(trajectories):
        if not isinstance(trajectories, list):
            trajectories = [trajectories]

        def internal(trajectory):

            obs, action, rewards, nobs, done = sarsd_getter(trajectory)
            action = np_to_torch(action)
            dist, v = ac(obs)
            v = v.squeeze()
            with torch.no_grad():
                _, vprime = ac(nobs)
                vprime = vprime.squeeze()
                advantage = gamma * vprime * np_to_torch(done == 0.0) + \
                    np_to_torch(rewards) - v.detach()

                advantage = discount(advantage.numpy(), gamma*lambd)
                advantage = np_to_torch(advantage.copy())
                advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)

            rewards = np_to_torch(
                np.array(discount(rewards, gamma*lambd)))
            rewards = (rewards-rewards.mean())/(rewards.std()+1e-8)
            value_loss = F.smooth_l1_loss(rewards, v)

            logprobs = dist.log_prob(action)
            policy_loss = (-advantage*logprobs).sum()
            entropy_loss = - dist.entropy().sum()
            return policy_loss, value_loss, entropy_loss

        policyl, criticl, entropyl = zip(
            *[internal(traj) for traj in trajectories])
        policyl = torch.stack(policyl).sum()
        criticl = torch.stack(criticl).sum()
        entropyl = beta * torch.stack(entropyl).mean()

        return policyl + criticl + entropyl

    return A2C
