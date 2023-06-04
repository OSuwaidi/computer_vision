# بسم الله الرحمن الرحيم

import torch
from matplotlib import pyplot as plt
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from string import ascii_lowercase as letters
import random
from tqdm import trange

'''
    Fourier Transform:
    img = torch.moveaxis(i[0], 0, 2).cpu().numpy()
    dft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft = np.fft.fftshift(dft)  # Shift the center of the fourier transformed image
    mag = 20*np.log((cv2.magnitude(dft[:, :, 0], dft[:, :, 1])))
    plt.imshow(mag, cmap='gray')
'''
device = torch.device('cuda')
T = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])  # Makes image range from 0 to 1: "(xi - μ)/σ", where μ=0, σ=1
LR = 0.05
BS = 200
EPOCHS = 10

train_data = datasets.MNIST(root='data', transform=T, download=True)
train_loader = DataLoader(dataset=train_data, pin_memory=True, batch_size=BS, shuffle=True)

test_data = datasets.MNIST(root='data', transform=T, train=False)
test_subset = Subset(test_data, random.sample(range(10000), k=30))  # Only take 30 random images from the 10,000 images contained in "test_data"
test_loader = DataLoader(dataset=test_subset, pin_memory=False, batch_size=30)  # "pin_memory=False" since we're not loading/using a lot of data for testing


class Autoencoder(nn.Module):
    def __init__(self, out_channel):  # Input (1, 28, 28)
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_channel, 5, padding=2)  # (28, 28)
        self.conv2 = nn.Conv2d(out_channel, out_channel * 2, 5)  # (24, 24) --> (12, 12)
        self.conv3 = nn.Conv2d(out_channel * 2, out_channel * 4, 3, padding=1)  # (12, 12)
        self.conv4 = nn.Conv2d(out_channel * 4, 1, 5)  # (8, 8) --> (1, 4, 4)
        self.act = F.mish
        self.deconv1 = nn.ConvTranspose2d(1, out_channel * 8, 5)  # (8, 8)
        self.deconv2 = nn.ConvTranspose2d(out_channel * 8, out_channel * 4, 5)  # (12, 12)
        self.deconv3 = nn.ConvTranspose2d(out_channel * 4, out_channel * 4, 3, padding=1)  # (12, 12)
        self.deconv4 = nn.ConvTranspose2d(out_channel * 4, out_channel * 2, 4, stride=2, padding=1)  # (24, 24)
        self.deconv5 = nn.ConvTranspose2d(out_channel * 2, out_channel, 5)  # (28, 28)
        self.deconv6 = nn.ConvTranspose2d(out_channel, 1, 3, padding=1)  # (28, 28)

    def encode(self, x):
        x = self.act(self.conv1(x))
        x = self.act(F.max_pool2d(self.conv2(x), 2))
        x = self.act(self.conv3(x))
        return F.max_pool2d(self.conv4(x), 2)

    def decode(self, x):
        x = (self.act(self.deconv1(x)))
        x = (self.act(self.deconv2(x)))
        x = (self.act(self.deconv3(x)))
        x = (self.act(self.deconv4(x)))
        x = (self.act(self.deconv5(x)))
        return torch.tanh(self.deconv6(x))

    def forward(self, x):
        return self.decode(self.encode(x))


AE = Autoencoder(10).to(device)

# base_params = [p[1] for p in AE.named_parameters() if p[0] not in exclude]
# optim = torch.optim.Adam([{'params': base_params}, {'params': AE.some_param, 'lr': 0.1}], lr=LR)  # "self.parameters()" will optimize whatever parameters exist under the "__init__" method!
optim = torch.optim.Adam(AE.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, LR, epochs=EPOCHS, steps_per_epoch=len(train_loader))

for _ in trange(EPOCHS):
    for img, _ in train_loader:
        img = img.to(device, non_blocking=True)
        noised = img + torch.randn_like(img)
        loss = torch.log(torch.sum((AE.forward(noised) - img) ** 2) + 1)
        loss.backward()
        optim.step()
        scheduler.step()
        optim.zero_grad()

AE.eval()
with torch.no_grad():
    for img, lbl in test_loader:
        img = img.to(device, non_blocking=True)
        noised = img + torch.randn_like(img)
        output = AE.forward(noised)
        for letter, image, num in zip(letters, output, lbl):
            image = torch.squeeze(image).cpu().numpy()
            plt.imsave(f'generated/{letter}_{num}.png', image, cmap='gray')

'''
torch.save(model.state_dict(), 'denoiser.pt')
To use the trained + saved model's parameters again without training, you first create an instance of the same model/network: "model = Autoencoder(lr, out_channel)
Then you load the saved parameters into that instantiated model: "model.load_state_dict(torch.load('denoiser.pt', map_location=device))" 
'''
