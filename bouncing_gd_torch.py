# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ و به نستعين

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import trange
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
import random

# plt.style.use('seaborn')
# plt.rcParams['figure.autolayout'] = True

torch.cuda.empty_cache()
device = torch.device('cpu')

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
T = transforms.Compose([transforms.ToTensor(), norm, transforms.Lambda(lambda x: torch.cat((x.flatten(), torch.Tensor([1]))))])
train_data = datasets.MNIST(root=f'datasets/MNIST', download=True, transform=T)  # (BS, 785)
test_data = datasets.MNIST(root=f'datasets/MNIST', train=False, transform=T)

BS: int = 100
LR: float = 10.
EPOCHS: int = 5
SEED: int = 0

# SGD:
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

train_loader = DataLoader(train_data, batch_size=BS, num_workers=0, shuffle=True, pin_memory=True, generator=torch.Generator().manual_seed(SEED))
test_loader = DataLoader(test_data, batch_size=BS, num_workers=0, pin_memory=True)

w = torch.randn(784, 100, requires_grad=True, device=device)
b = torch.randn(100, requires_grad=True, device=device)
wh = torch.randn(100, 10, requires_grad=True, device=device)
bh = torch.randn(10, requires_grad=True, device=device)

losses_sgd = []
for _ in trange(EPOCHS):
    loss_batch = []
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = torch.tanh(x @ w + b) @ wh + bh  # raw logits
        loss = F.cross_entropy(out, y)
        loss_batch.append(loss.item())
        grad_w, grad_b, grad_wh, grad_bh = torch.autograd.grad(loss, (w, b, wh, bh))

        with torch.no_grad():  # computations below are untracked as tensors are treated as "detached" tensors
            w -= grad_w * LR  # (784, 100)
            b -= grad_b * LR  # (100)
            wh -= grad_wh * LR  # (100, 10)
            bh -= grad_bh * LR  # (10)

    losses_sgd.append(np.mean(loss_batch))
plt.semilogy(losses_sgd, label='SGD')

acc = []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = torch.tanh(x @ w + b) @ wh + bh
        _, pred = out.max(1)
        acc.append((pred == y).float().mean().item())
print(f'Accuracy: {np.mean(acc) * 100:.4}%')

############################################################################################################################################################

# BGD:
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

train_loader = DataLoader(train_data, batch_size=BS, num_workers=0, shuffle=True, pin_memory=True, generator=torch.Generator().manual_seed(SEED))
test_loader = DataLoader(test_data, batch_size=BS, num_workers=0, pin_memory=True)

w = torch.randn(785, 100, requires_grad=True, device=device)  # each column corresponds to an output node (neuron) in the network
wh = torch.randn(101, 10, requires_grad=True, device=device)

TH = 0.9  # ThreshHold value is inversely proportional to the initial learning rate
EPS = torch.Tensor([1e-11])
BIAS = torch.ones(100, device=device).unsqueeze(1)
SHRINK = torch.tensor(1.1, device=device)
LRw = torch.ones(100, device=device) * LR  # per output node (neuron) adaptive learning rate (not per parameter/weight)
LRwh = torch.ones(10, device=device) * LR


def dist(g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
    flatness_1, flatness_2 = g1.norm().item(), g2.norm().item()
    dists = torch.Tensor([flatness_2, flatness_1])
    return dists / (dists.sum() + EPS)


def bounce_update(weight, oracle, weight_gradient, oracle_gradient, weight_LR) -> None:
    dot_prods: torch.Tensor = torch.einsum("kj,kj->j", weight_gradient, oracle_gradient)
    for i, dot_prod in enumerate(dot_prods):
        if dot_prod.item() < 0:
            # print('bounce')
            d1, d2 = dist(weight_gradient[:, i], oracle_gradient[:, i])
            if d1.item() > TH:
                weight_LR[i] /= SHRINK

            weight[:, i] = weight[:, i] * d1 + oracle[:, i] * d2

        else:
            weight[:, i] = oracle[:, i] - oracle_gradient[:, i] * weight_LR[i]


losses_bgd = []
for _ in trange(EPOCHS):
    loss_batch = []
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = torch.cat(((x @ w).tanh(), BIAS), dim=1) @ wh
        loss = F.cross_entropy(out, y)
        loss_batch.append(loss.item())
        loss.backward()
        grad_w, grad_wh = w.grad, wh.grad

        with torch.no_grad():  # computations below are untracked as tensors are treated as "detached" tensors
            # Find oracle weights:
            w_oracle = (w - grad_w * LRw).requires_grad_()  # (784, 100)
            wh_oracle = (wh - grad_wh * LRwh).requires_grad_()  # (100, 10)

        out = torch.cat(((x @ w_oracle).tanh(), BIAS), dim=1) @ wh_oracle
        loss = F.cross_entropy(out, y)
        loss.backward()
        grad_w_orc, grad_wh_orc = w_oracle.grad, wh_oracle.grad

        with torch.no_grad():
            bounce_update(w, w_oracle, grad_w, grad_w_orc, LRw)
            bounce_update(wh, wh_oracle, grad_wh, grad_wh_orc, LRwh)

        # Zero each parameters' gradients to avoid gradient accumulation across iterations:
        w.grad.zero_()
        wh.grad.zero_()

    losses_bgd.append(np.mean(loss_batch))
plt.semilogy(losses_bgd, label='BGD')

acc = []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = torch.cat(((x @ w).tanh(), BIAS), dim=1) @ wh
        _, pred = out.max(1)
        acc.append((pred == y).float().mean().item())
print(f'Accuracy: {np.mean(acc) * 100:.4}%')

plt.gca().text(.5, .8, f'$LR={LR}$', c='orange', size=15, transform=plt.gca().transAxes)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
