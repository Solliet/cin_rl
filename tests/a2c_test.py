import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D
import gym
import functools

from cin_rl.agents.a2c import A2C
from cin_rl.nn.actor_critic import ActorCritic

from collections import namedtuple
import numpy as np

env = gym.make('Pendulum-v0')
obs = env.reset()


model = nn.ModuleDict(
    {
        'shared': nn.Sequential(
            nn.Linear(4, 64),
            nn.ELU(),
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
    return D.Categorical(logits=F.log_softmax(model['actor'](f1), dim=0)).sample().item()

objective = A2C(ac, beta=1e-4, gamma=1-1e-3, lambd=1-1e-3)

Experience = namedtuple(
    "Experience", ["obs", "action", "reward", "newobs", "done"])


class SingleRunner:
    def __init__(self, env, seed=1):
        self.env = env()
        self.env.seed(seed)

    def collect_episode(self, agent):
        done = False
        obs = self.env.reset()
        history = []

        while not done:
            action = agent(obs)
            nobs, reward, done, _ = self.env.step(action)
            history.append(Experience(obs, action, reward, nobs, done))
            obs = nobs

        return Experience(*map(np.array, zip(*history)))


optim = torch.optim.Adam(params=model.parameters(), lr=1e-2)
reward = 10.0
runner = SingleRunner(functools.partial(gym.make, 'CartPole-v1'))

for _ in range(1000):
    trajectories = [runner.collect_episode(act) for _ in range(8)]
    reward = 0.95 * reward + 0.05 * \
        np.array([trajectory.reward.sum()
                  for trajectory in trajectories]).mean()
    optim.zero_grad()
    loss = objective(trajectories)
    loss.backward()
    optim.step()
    print(_, reward, loss)
