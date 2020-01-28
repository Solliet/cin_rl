'''

'''

import numpy as np
import torch
from torch import from_numpy as np2torch
from torch.nn import functional as F
from scipy.signal import lfilter
from collections import namedtuple
from operator import attrgetter
from functools import reduce


def discount(x, gamma): return lfilter(
    [1], [1, -gamma], x[::-1])[::-1]


Experience = namedtuple(
    "Experience", ["obs", "action", "reward", "newobs", "done"])
sarsd_getter = attrgetter("obs", "action", "reward", "newobs", "done")


def PPO(ac, alpha, beta, gamma, lambd, eps_clip):

    def prepare(trajectory):
        obs, action, rewards, nobs, done = sarsd_getter(trajectory)

        obs = np2torch(obs)
        action = np2torch(action)
        t_rewards = np2torch(rewards)
        done = np2torch(done)

        with torch.no_grad():
            dist, v = ac(obs)
            _, vprime = ac(nobs)

        v = v.squeeze()
        vprime = vprime.squeeze()
        
        # residuals for GAE
        td_residual = t_rewards + vprime * gamma * done - v
        advantage = np2torch(discount(td_residual.numpy(), gamma * lambd).copy())
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

        returns = np2torch(discount(rewards, gamma).copy())
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # action probabilities with the initial policy
        action_proba = dist.log_prob(action).exp()

        # obs, action, advantage, action_proba, returns
        return obs.squeeze(), action, advantage.squeeze(), action_proba, returns

    def PPOObjective(obs, actions, advantage, action_proba, returns):

        dist, value = ac(obs)

        prob_ratio = dist.log_prob(actions).exp() / action_proba
        lcpi = advantage * prob_ratio
        lclip = torch.clamp(prob_ratio, 1 - eps_clip, 1 + eps_clip) * advantage
        policy_loss = -torch.min(lcpi, lclip).mean()

        value = value.squeeze()
        value_loss = alpha * F.smooth_l1_loss(returns, value)
        entropy_loss = -beta * dist.entropy().mean()

        return policy_loss + value_loss + entropy_loss

    def PPO(trajectories, batch_size):
        if not isinstance(trajectories, list):
            trajectories = [trajectories]
        obs, action, advantage, action_proba, returns = zip(
            *[prepare(trajectory) for trajectory in trajectories])
        obs = torch.cat(obs).squeeze()
        action = torch.cat(action).squeeze()
        advantage = torch.cat(advantage).squeeze()
        action_proba = torch.cat(action_proba).squeeze()
        returns = torch.cat(returns).squeeze()

        while True:
            indices = torch.randint(0, obs.shape[0], (batch_size,))
            yield PPOObjective(obs[indices], action[indices], advantage[indices], action_proba[indices],
                               returns[indices])

    return PPO


def PPO_step(traj_generator, ppo_objective, optimizer, ppo_iterations, minibatch_size):
    trajectories = traj_generator()

    if isinstance(ppo_iterations, int):
        ppo_iterations = range(ppo_iterations)

    losses = []
    for i, loss in zip(ppo_iterations, ppo_objective(trajectories, minibatch_size)):
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

    return np.array(losses).mean()


def PPO_train(ac, optimizer, traj_generator, evaluator, alpha, beta, gamma, lambd, eps_clip, nepisodes, nppo_iterations,
              minibatch_size):
    J = PPO(ac, alpha=alpha, beta=beta,
            gamma=gamma, lambd=lambd, eps_clip=eps_clip)

    if isinstance(nepisodes, int):
        nepisodes = range(nepisodes)

    if isinstance(nppo_iterations, int):
        nppo_iterations = range(nppo_iterations)

    losses = []
    rewards = []
    for i in nepisodes:
        losses.append(PPO_step(traj_generator, J, optimizer,
                           nppo_iterations, minibatch_size))
        rewards.append(evaluator())
        avg = reduce(lambda l, curr: l*0.90+0.1*curr , rewards)
        print(f'epoch::{i}\t\tloss::{losses[-1]}\t\tavgreward::{avg}\t\treward::{rewards[-1]}')

    return losses
