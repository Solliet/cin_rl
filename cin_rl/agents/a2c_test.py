import torch
from torch import nn
from torch.nn import functional as F
import gym
import functools
from a2c import A2C
from collections import namedtuple
import numpy as np
from multiprocessing_env import SubprocVecEnv

env = gym.make('Pendulum-v0')
obs = env.reset()
model = nn.Sequential(
        nn.Linear(4, 128),
        nn.ELU(),
        nn.Linear(128,2)
).float()


class ActorCritic(nn.Module):
    def __init__(self, common, critic, actor):
        super(ActorCritic, self).__init__()
        self.common = common
        self.critic = critic
        self.actor = actor
    
    def forward(self, obs):
        l = self.common(obs)
        return self.critic(l), self.actor(l)


agent = A2C(ActorCritic(
    nn.Sequential(
        nn.Linear(4, 64),
        nn.ELU(),
    ), nn.Linear(64, 1),nn.Linear(64, 2)
), 0.999, 0.99, 1e-4)


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

optim = torch.optim.Adam(agent.parameters(), lr=1e-2)
reward = 10.0
runner = SingleRunner(functools.partial(gym.make,'CartPole-v1'))

for _ in range(1000):
    trajectories = [runner.collect_episode(agent) for _ in range(8)]
    reward = 0.95 * reward + 0.05*np.array([trajectory.reward.sum() for trajectory in trajectories]).mean()
    optim.zero_grad()
    loss = agent.loss(trajectories)
    loss.backward()
    optim.step()
    print(_, reward, loss)
