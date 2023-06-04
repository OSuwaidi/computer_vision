# بسم الله الرحمن الرحيم و به نستعين

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from matplotlib import pyplot as plt


# Hyper-parameters:
device = "cuda"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
NUM_CLASSES = 10
IMG_CHANNELS = 1
Z_DIM = 100
EPOCHS = 50
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)]),
    ]
)

dataset = datasets.MNIST(root="datasets/MNIST", transform=transforms, download=True)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


class Discriminator(nn.Module):
    def __init__(self, img_channels, out_dims, num_classes, img_size):
        super().__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            # input: (BS, 1, 64, 64)
            nn.Conv2d(img_channels, out_dims, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(out_dims, out_dims * 2, 4, 2, 1),
            self._block(out_dims * 2, out_dims * 4, 4, 2, 1),
            self._block(out_dims * 4, out_dims * 8, 4, 2, 1),  # (4, 4)
            nn.Conv2d(out_dims * 8, 11, kernel_size=4, stride=1, padding=0),  # (11, 1, 1)
        )
        self.embed = nn.Embedding(num_classes, img_size**2)

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, y):
        embedding = self.embed(y).view(y.size(0), 1, self.img_size, self.img_size)
        return self.disc(x + embedding).view(-1, 11)


class Generator(nn.Module):
    def __init__(self, noise_channels, img_channels, out_dims, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            # input: (BS, 100, 1, 1)
            self._block(noise_channels, out_dims * 16, 4, 1, 0),  # (100, 4, 4)
            self._block(out_dims * 16, out_dims * 8, 4, 2, 1),  # (8, 8)
            self._block(out_dims * 8, out_dims * 4, 4, 2, 1),  # (16, 16)
            self._block(out_dims * 4, out_dims * 2, 4, 2, 1),  # (32, 32)
            nn.ConvTranspose2d(out_dims * 2, img_channels, kernel_size=4, stride=2, padding=1),  # (1, 64, 64)
            nn.Tanh(),
        )
        self.embed = nn.Embedding(num_classes, noise_channels)

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, y):
        embedding = self.embed(y).unsqueeze(2).unsqueeze(3)
        return self.net(x + embedding)


def initialize_weights(model):
    # Initialize weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


################################################################################################################################################################################


gen = Generator(Z_DIM, IMG_CHANNELS, FEATURES_GEN, NUM_CLASSES).to(device)
critic = Discriminator(IMG_CHANNELS, FEATURES_CRITIC, NUM_CLASSES, IMAGE_SIZE).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# For tensorboard plotting
writer_real = SummaryWriter(f"logs/CGAN_MNIST/real")
writer_fake = SummaryWriter(f"logs/CGAN_MNIST/fake")
step = 0

gen.train()
critic.train()
y_real = torch.ones(BATCH_SIZE).to(device)
y_fake = torch.zeros(BATCH_SIZE).to(device)
for epoch in trange(EPOCHS):
    for batch_idx, (real, y) in enumerate(loader):
        real, y = real.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
            fake = gen(noise, y)
            preds_real = critic(real, y)
            preds_fake = critic(fake, y)
            loss_critic_real = (F.cross_entropy(preds_real[:, :-1], y) + F.binary_cross_entropy(F.sigmoid(preds_real[:, -1]), y_real))
            loss_critic_fake = (F.cross_entropy(preds_fake[:, :-1], y) + F.binary_cross_entropy(F.sigmoid(preds_fake[:, -1]), y_fake))
            loss_critic = (loss_critic_real + loss_critic_fake)/2
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        preds_fake = critic(fake, y)
        loss_gen = F.binary_cross_entropy(F.sigmoid(preds_fake[:, -1]), y_real)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            tqdm.write(
                f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4}, loss G: {loss_gen:.4}"
            )

            with torch.no_grad():
                fake = gen(noise, y)
                # Extract 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
            step += 1

gen.eval()
with torch.no_grad():
    y = torch.tensor([1, 0, 3, 5, 2], dtype=torch.int64).to(device)
    test_noise = torch.rand(5, Z_DIM, 1, 1).to(device)
    fakes = gen(test_noise, y)  # (5, 1, 64, 64)
    fakes = torch.squeeze(fakes).cpu().detach()

for i, img in enumerate(fakes):
    plt.imshow(img)
    plt.show()
    plt.imsave(f'{i}.png', img)
