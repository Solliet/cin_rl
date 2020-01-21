import torch
from torch import nn, optim
from torch.nn import functional as F
from .common import get_device, get_last_dim
import abc


class VisionModule(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, output_dim):
        super(self, VisionModule).__init__()
        self.output_dim = output_dim

    @abc.abstractmethod
    def encode(self, input):
        raise NotImplementedError()

class VAE(VisionModule):

    def __init__(self, encoder: VisionModule, decoder: nn.Module, latent, kl_weight=1, kl_anneal_steps=1):
        super(VAE, self).__init__(latent)
        self.encoder = encoder
        self.decoder = decoder
        self.to()

        self.mu = nn.Linear(
            in_features=get_last_dim(encoder).out_features,
            out_features=latent
        ).to(get_device(encoder))

        self.logvar = nn.Linear(
            in_features=get_last_dim(encoder).out_features,
            out_features=latent
        ).to(get_device(encoder))

        self.kl_loss = lambda logvar, mu: -0.5 * \
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.kl_weight = kl_weight
        self.kl_anneal_steps = kl_anneal_steps

        self.steps = 1

    def forward(self, x, training=False):
        compressed = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)

        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            xprime = self.decoder(z)
            return xprime, mu, logvar
        else:
            return torch.cat([mu, logvar], dim=1)

    def loss(self, x, xprime, mu, logvar):
        kl = self.kl_loss(logvar, mu) * self.kl_weight * \
            min(self.steps/self.kl_anneal_steps, 1)
        rec = F.mse_loss(x, xprime)
        return kl, rec

    def step_kl(self):
        self.steps += 1

class MMD_VAE(VisionModule):
    def __init__(self, encoder: VisionModule, decoder: nn.Module):
        super(MMD_VAE, self).__init__(encoder.output_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.to(get_device(encoder))

    def compress(self, x):
        return self.encoder(x)

    def forward(self, x, training=False):
        latent = self.compress(x)
        if training:
            return latent, self.decoder(latent)
        return latent

    def loss(self, x, xprime, latent):
        def gaussian_kernel(a, b):
            dim1_1, dim1_2 = a.shape[0], b.shape[0]
            depth = a.shape[1]
            a = a.view(dim1_1, 1, depth)
            b = b.view(1, dim1_2, depth)
            a_core = a.expand(dim1_1, dim1_2, depth)
            b_core = b.expand(dim1_1, dim1_2, depth)
            numerator = (a_core - b_core).pow(2).mean(2)/depth
            return torch.exp(-numerator)

        def MMD(a, b):
            return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()

        reconstruction_error = F.mse_loss(x, xprime)
        mmd = MMD(torch.randn(200, self.encoder.output_dim,
                              requires_grad=False).to(self.device), latent)
        return reconstruction_error, mmd
