# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ و به نستعين

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import trange
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt

# plt.style.use('seaborn')
plt.rcParams['figure.autolayout'] = True

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
T = transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)])
train_data = datasets.MNIST(root=f'datasets/MNIST', download=True, transform=T)  # (BS, 784)
test_data = datasets.MNIST(root=f'datasets/MNIST', train=False, transform=T)

BS = 100
LR = 10
EPOCHS = 5

# SGD:
torch.manual_seed(0)
np.random.seed(0)

train_loader = DataLoader(train_data, batch_size=BS, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=BS, pin_memory=True)

w = torch.randn(784, 100, requires_grad=True)
b = torch.randn(100, requires_grad=True)
wh = torch.randn(100, 10, requires_grad=True)
bh = torch.randn(10, requires_grad=True)


losses_sgd = []
for _ in trange(EPOCHS):
    loss_batch = []
    for x, y in train_loader:
        out = torch.softmax(torch.tanh(x @ w + b) @ wh + bh, dim=1)
        loss = F.cross_entropy(out, y)
        loss_batch.append(loss.item())
        grad_w = torch.autograd.grad(loss, w, retain_graph=True)[0]  # (784, 100)
        grad_b = torch.autograd.grad(loss, b, retain_graph=True)[0]  # (100)
        grad_wh = torch.autograd.grad(loss, wh, retain_graph=True)[0]  # (100, 10)
        grad_bh = torch.autograd.grad(loss, bh)[0]  # (10)
        with torch.no_grad():
            w -= grad_w * LR
            b -= grad_b * LR
            wh -= grad_wh * LR
            bh -= grad_bh * LR
    losses_sgd.append(np.mean(loss_batch))
plt.semilogy(losses_sgd, label='SGD')

acc = []
with torch.no_grad():
    for x, y in test_loader:
        out = torch.tanh(x @ w + b) @ wh + bh
        _, pred = out.max(1)
        acc.append((pred == y).float().mean().item())
print(f'Accuracy: {np.mean(acc)*100:.4}%')

############################################################################################################################################################

# BGD:
torch.manual_seed(0)
np.random.seed(0)

train_loader = DataLoader(train_data, batch_size=BS, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=BS, pin_memory=True)

w = torch.randn(784, 100, requires_grad=True)
b = torch.randn(100, requires_grad=True)
wh = torch.randn(100, 10, requires_grad=True)
bh = torch.randn(10, requires_grad=True)


def dist(g1, g2):
    e = 1e-08
    d1 = 1 / (g1.norm().item() + e)
    d2 = 1 / (g2.norm().item() + e)
    dists = torch.Tensor([d1, d2])
    dists /= dists.sum()
    return dists


TH = 0.9  # TH value is inversely proportional to "ini"
shrink = 1.1
LRw = torch.ones(100)*LR
LRwh = torch.ones(10)*LR

losses_bgd = []
for _ in trange(EPOCHS):
    loss_batch = []
    for x, y in train_loader:
        out = torch.softmax(torch.tanh(x @ w + b) @ wh + bh, dim=1)
        loss = F.cross_entropy(out, y)
        loss_batch.append(loss.item())
        grad_w = torch.autograd.grad(loss, w, retain_graph=True)[0]  # (784, 100)
        grad_b = torch.autograd.grad(loss, b, retain_graph=True)[0]  # (100)
        grad_wh = torch.autograd.grad(loss, wh, retain_graph=True)[0]  # (100, 10)
        grad_bh = torch.autograd.grad(loss, bh)[0]  # (10)

        # Find oracle weights:
        oracle_w = w - grad_w * LRw
        oracle_wh = wh - grad_wh * LRwh
        
        # Update bias vectors:
        with torch.no_grad():
            b -= grad_b * LR
            bh -= grad_bh * LR

        out = torch.softmax(torch.tanh(x @ oracle_w + b) @ oracle_wh + bh, dim=1)
        loss = F.cross_entropy(out, y)
        grad_orc_w = torch.autograd.grad(loss, oracle_w, retain_graph=True)[0]  # (784, 100)
        grad_orc_wh = torch.autograd.grad(loss, oracle_wh, retain_graph=True)[0]  # (100, 10)

        with torch.no_grad():
            # Update w:
            for i, (g, g_orc) in enumerate(zip(grad_w.T, grad_orc_w.T)):  # (100, 784)
                if g @ g_orc < 0:  # (784)
                    # print('bounce')
                    d1, d2 = dist(g, g_orc)
                    if d1 > TH:
                        LRw[i] /= shrink
                    w[:, i] = w[:, i]*d1 + oracle_w[:, i]*d2
                else:
                    w[:, i] = oracle_w[:, i] - g_orc * LRw[i]

            # Update wh:
            for i, (g, g_orc) in enumerate(zip(grad_wh.T, grad_orc_wh.T)):  # (10, 100)
                if g @ g_orc < 0:  # (100)
                    # print('bounce')
                    d1, d2 = dist(g, g_orc)
                    if d1 > TH:
                        LRwh[i] /= shrink
                    wh[:, i] = wh[:, i]*d1 + oracle_wh[:, i]*d2
                else:
                    wh[:, i] = oracle_wh[:, i] - g_orc * LRwh[i]

    losses_bgd.append(np.mean(loss_batch))
plt.semilogy(losses_bgd, label='BGD')

acc = []
with torch.no_grad():
    for x, y in test_loader:
        out = torch.tanh(x @ w + b) @ wh + bh
        _, pred = out.max(1)
        acc.append((pred == y).float().mean().item())
print(f'Accuracy: {np.mean(acc)*100:.4}%')

plt.gca().text(.5, .8, f'$LR={LR}$', c='orange', size=15, transform=plt.gca().transAxes)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
