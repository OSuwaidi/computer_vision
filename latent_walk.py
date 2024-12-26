import torch
import matplotlib.pyplot as plt
from vae_lm import gauss_activ
import torch.nn as nn

device = torch.device('cuda')
torch.cuda.empty_cache()

vae_v = torch.load("vae_v.pth")
vae_v.eval()

vae_t = torch.load("vae_t.pth")
vae_t.eval()

vae_g = torch.load("vae_g.pth")
vae_g.eval()

vaes = (vae_v, vae_t, vae_g)

def linear_interpolation(z1, z2, steps=10):
    return [(1 - t) * z1 + t * z2 for t in torch.linspace(0, 1, steps)]


# Random sampling from the latent space
z_dim = 2  # Adjust this to your latent space dimension
z1 = torch.randn(1, z_dim).to(device)  # Latent vector 1
z2 = torch.randn(1, z_dim).to(device)  # Latent vector 2

latent_vectors = linear_interpolation(z1, z2, steps=50)

activations = (nn.Identity(), torch.tanh, gauss_activ)

for c, vae in enumerate(vaes):
    f = activations[c]
    generated_images = [vae.decoder(f(z)).view(28, 28).detach().cpu().numpy() for z in latent_vectors]

    # Reshape and visualize the images
    plt.figure(figsize=(10, 5))
    for i, img in enumerate(generated_images):
        plt.subplot(5, 10, i + 1)
        plt.imshow(img.squeeze(), cmap='gray')  # Adjust for RGB if needed
        plt.axis('off')
    plt.savefig(f"vis_{c}.png")
