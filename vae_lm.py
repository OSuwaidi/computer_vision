import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, latent_dim)

        # Basis and support point
        self.basis = nn.Parameter(torch.empty(input_dim, latent_dim))
        nn.init.xavier_uniform_(self.basis)

        self.support_point = nn.Parameter(torch.zeros(input_dim))

        # Decoder
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encoder(self, x):
        h = self.fc1(x).relu()
        coord = self.fc_c(h).tanh()
        return coord  # (BS, latent_dim)

    def linear_comb(self, coord):  # "torch.einsum('bd,bdo->bo', coord, basis_vectors)" <==> "torch.stack([c @ b for c, b in zip(coord, basis_vectors)])"
        return F.linear(coord, self.basis, self.support_point)  # ==> X @ W.T + b (BS, input_dim)

    def decoder(self, coord):
        z = self.linear_comb(coord)
        h = self.fc2(z).relu()
        return self.fc3(h).tanh()  # Apply sigmoid for [0, 1] normalized data and tanh for [-1, 1]

    def forward(self, x):
        coord = self.encoder(x)
        recon_x = self.decoder(coord)
        return recon_x


def vae_loss(recon_x, x):
    # Reconstruction loss (BCE or MSE, depending on data type)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)  # Averaged over batch
    return recon_loss


def gauss_activ(x: torch.Tensor):
    return torch.exp(-x.pow(2)/2)*x*1.07
