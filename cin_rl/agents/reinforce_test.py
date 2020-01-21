import torch
from torch import nn
from torch.nn import functional as F
import gym
import functools
from reinforce import  Reinforce
from collections import namedtuple
import numpy as np

env = gym.make('CartPole-v1')
obs = env.reset()
model = nn.Sequential(
        nn.Linear(4, 128),
        nn.ELU(),
        nn.Linear(128,2)
).float()

agent = Reinforce(model, 0.999)


Experience = namedtuple("Experience", ["obs", "action", "reward", "newobs", "done"])

class SingleRunner:
    def __init__(self, env, seed=1):
        self.env = env()
        self.env.seed(seed)

    def collect_episode(self, agent):
        done = False
        obs = self.env.reset()
        history = []

        while not done:
            action, logits = agent(obs)
            nobs, reward, done, _ = self.env.step(action)
            history.append(Experience(obs, action, reward, nobs, done))
            obs = nobs
        
        return Experience(*map(np.array, zip(*history)))

optim = torch.optim.Adam(agent.parameters())
reward = 10.0
runner = SingleRunner(functools.partial(gym.make,'CartPole-v1'))
for _ in range(10000):
    trajectory = runner.collect_episode(agent)
    reward = 0.95 * reward + 0.05*trajectory.reward.sum()
    optim.zero_grad()
    loss = agent.loss(trajectory)
    loss.backward()
    optim.step()
    print(_, reward, loss.item())