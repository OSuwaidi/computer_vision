import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Basis
        self.basis = nn.Parameter(torch.randn(latent_dim + 1, input_dim))

        # Decoder
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encoder(self, x):
        h = self.fc1(x).relu()
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        # Sample epsilon from Normal(0,I)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        coord = mu + eps * std
        return coord

    def linear_comb(self, coord):
        z = coord @ self.basis[:-1]  # (BS, input_dim)
        return z + self.basis[-1]

    def decoder(self, coord):
        z = self.linear_comb(coord)
        h = self.fc2(z).relu()
        return self.fc3(h).tanh()  # Apply sigmoid for [0, 1] normalized data and tanh for [-1, 1]

    def forward(self, x):
        mu, logvar = self.encoder(x)
        coord = self.reparameterize(mu, logvar)
        recon_x = self.decoder(coord)
        return recon_x, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x.view(-1, 784), reduction='sum') / x.size(0)  # Averaged over batch

    # KL Divergence term
    kl_div = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum() / x.size(0)  # Averaged over batch

    return recon_loss + kl_div
