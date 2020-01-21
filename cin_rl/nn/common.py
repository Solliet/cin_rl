import torch
from torch import nn, optim, distributions
from functools import partial


class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.target_shape)


class Flatten(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Residual(nn.Module):
    pass


class InvertedResidual(nn.Module):
    pass


def get_last_dim(module: nn.Module):
    return list(m for m in module.parameters() if isinstance(m, torch.Tensor))[-1].shape

def get_device(module: nn.Module):
    return next(module.parameters()).device


class ActorCritic(nn.Module):
    def __init__(self, common, critic, actor):
        super(ActorCritic, self).__init__()
        self.common = common
        self.critic = critic
        self.actor = actor
    
    def forward(self, obs):
        l = self.common(obs)
        return self.critic(l), self.actor(l)


