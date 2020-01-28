import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D
import gym
import functools
import itertools


from cin_rl.agents import ppo
from cin_rl.runners import episode_runner, run_single

from collections import namedtuple
from collections.abc import Iterable

import numpy as np
import joblib

env = gym.make('Pendulum-v0')
obs = env.reset()


model = nn.ModuleDict(
    {
        'shared': nn.Sequential(
            nn.Linear(8, 256),
            nn.SELU(),
            nn.Linear(256, 256),
            nn.SELU(),
            nn.Linear(256, 256),
        ),
        'actor': nn.Linear(256, 4),
        'critic': nn.Linear(256, 1),
        
    }
).double()


def ac(obs):
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs).double()
    f1 = model['shared'](obs.double())
    return D.Categorical(logits=F.log_softmax(model['actor'](f1), dim=1)), model['critic'](f1)


def act(obs):
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs).double()
    f1 = model['shared'](obs)
    logits = F.log_softmax(model['actor'](f1), dim=0)
    return D.Categorical(logits=logits).sample().item()


Experience = namedtuple(
    "Experience", ["obs", "action", "reward", "newobs", "done"])

optimizer = torch.optim.Adam([
    dict(params=model['critic'].parameters(), lr=1e-4),
    dict(params=model['shared'].parameters(), lr=1e-4),
    dict(params=model['actor'].parameters(), lr=1e-4)
])

env_fn = functools.partial(gym.make, 'LunarLander-v2')


test_runner = episode_runner(env_fn, act)


ppo.PPO_train(ac, optimizer, lambda: joblib.Parallel(n_jobs=joblib.cpu_count())([joblib.delayed(run_single)(env_fn, act) for _ in range(16)]),
    lambda: next(test_runner).reward.sum(),
    alpha=0.1,
    beta=0.01,
    gamma=0.999,
    lambd=0.999,
    eps_clip=0.20,
    nepisodes=500,
    nppo_iterations=32,
    minibatch_size=2096)
