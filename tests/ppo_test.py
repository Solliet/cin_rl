import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D
import gym
import functools

from cin_rl.agents import PPO
from cin_rl.nn.actor_critic import ActorCritic
from cin_rl.runners import episode_runner, nstep_runner

from collections import namedtuple
import numpy as np

env = gym.make('Pendulum-v0')
obs = env.reset()


model = nn.ModuleDict(
    {
        'shared': nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
        ),
        'actor': nn.Linear(64, 2),
        'critic': nn.Linear(64, 1)
    }
).double()


def ac(obs):
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs)
    f1 = model['shared'](obs)
    return D.Categorical(logits=F.log_softmax(model['actor'](f1), dim=1)), model['critic'](f1)


def act(obs):
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs)
    f1 = model['shared'](obs)
    logits = F.log_softmax(model['actor'](f1), dim=0)
    return D.Categorical(logits=logits).sample().item()


Experience = namedtuple("Experience", ["obs", "action", "reward", "newobs", "done"])

optim = torch.optim.Adam([
    dict(params=model['critic'].parameters(), lr=1e-2),
    dict(params=model['shared'].parameters(), lr=1e-2),
    dict(params=model['actor'].parameters(), lr=1e-2)
])

reward = 10.0
env_fn = functools.partial(gym.make, 'CartPole-v1')



ppo_objective = PPO(ac, alpha=0.999, beta=1e-4, gamma=0.999, lambd=0.999, eps_clip=.2)
    
test_runner = episode_runner(env_fn, act)
runners = [
    episode_runner(env_fn, act)
    for _ in range(16)
]

for _ in range(1000):
    
    trajectories = [next(runner) for runner in runners]
    for i, loss in zip(range(16), ppo_objective(trajectories, 64)):
        optim.zero_grad()
        loss.backward()
        optim.step()
    
    reward = 0.95 * reward + 0.05 * next(test_runner).reward.sum()
    print(_, reward, loss.item())
