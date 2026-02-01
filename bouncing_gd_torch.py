# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ و به نستعين

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from tqdm import trange
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
import random

# plt.style.use('seaborn')
# plt.rcParams['figure.autolayout'] = True

# torch.mps.empty_cache()
device = torch.device('mps')


norm = v2.Normalize(mean=[0.45], std=[0.23])
T = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    # norm,
    v2.Lambda(lambda x: torch.cat((x.flatten(), torch.Tensor([1]))))
])
train_data = datasets.MNIST(root=f'datasets/MNIST', download=True, transform=T)  # (BS, 785)
test_data = datasets.MNIST(root=f'datasets/MNIST', train=False, transform=T)

BS: int = 100
LR: float = 10.
EPOCHS: int = 5
SEED: int = 0
PIN_MEM: bool = False


def run_sgd():
    # SGD:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_loader = DataLoader(train_data, batch_size=BS, num_workers=0, shuffle=True, pin_memory=PIN_MEM, generator=torch.Generator().manual_seed(SEED))
    test_loader = DataLoader(test_data, batch_size=BS, num_workers=0, pin_memory=PIN_MEM)

    w = torch.randn(785, 100, requires_grad=True, device=device)
    wh = torch.randn(100, 10, requires_grad=True, device=device)
    bh = torch.randn(10, requires_grad=True, device=device)

    losses_sgd = []
    for _ in trange(EPOCHS):
        loss_batch = []
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=PIN_MEM), y.to(device, non_blocking=PIN_MEM)
            out = torch.tanh(x @ w) @ wh + bh  # raw logits
            loss = F.cross_entropy(out, y)
            loss_batch.append(loss.item())
            loss.backward()
            # grad_w, grad_wh, grad_bh = torch.autograd.grad(loss, (w, wh, bh))

            with torch.no_grad():  # computations below are untracked as tensors are treated as "detached" tensors
                w -= w.grad * LR  # (784, 100)
                wh -= wh.grad * LR  # (100, 10)
                bh -= bh.grad * LR  # (10)

            w.grad.zero_()
            wh.grad.zero_()
            bh.grad.zero_()

        losses_sgd.append(np.mean(loss_batch))
    plt.semilogy(losses_sgd, label='SGD')

    acc = []
    with torch.inference_mode():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=PIN_MEM), y.to(device, non_blocking=PIN_MEM)
            out = torch.tanh(x @ w) @ wh + bh
            _, pred = out.max(1)
            acc.append((pred == y).float().mean().item())
    print(f'Accuracy: {np.mean(acc) * 100:.4}%')

############################################################################################################################################################


def run_bgd():
    # BGD:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_loader = DataLoader(train_data, batch_size=BS, num_workers=0, shuffle=True, pin_memory=PIN_MEM, generator=torch.Generator().manual_seed(SEED))
    test_loader = DataLoader(test_data, batch_size=BS, num_workers=0, pin_memory=PIN_MEM)

    w = torch.randn(785, 100, requires_grad=True, device=device)  # each column corresponds to an output node (neuron) in the network
    wh = torch.randn(101, 10, requires_grad=True, device=device)

    TH = 0.9  # ThreshHold value is inversely proportional to the initial learning rate
    EPS = 1e-11
    BIAS = torch.ones(BS, device=device).unsqueeze(1)
    SHRINK = torch.tensor(1.1, device=device)
    LRw = torch.ones(1, device=device) * LR  # per output node (neuron) adaptive learning rate (not per parameter/weight)
    LRwh = torch.ones(1, device=device) * LR

    def bounce_update(weight, oracle, weight_gradient, oracle_gradient, weight_LR) -> None:
        if (weight_gradient.view(-1) @ oracle_gradient.view(-1)).item() < 0:
            # print('bounce')
            flatness_1, flatness_2 = torch.linalg.vector_norm(weight_gradient), torch.linalg.vector_norm(oracle_gradient)
            s = flatness_1 + flatness_2 + EPS
            d1, d2 = flatness_2/s, flatness_1/s
            if d1.item() > TH:
                weight_LR /= SHRINK

            weight.data = weight * d1 + oracle * d2

        else:
            weight.data = oracle - oracle_gradient * weight_LR

    losses_bgd = []
    for _ in trange(EPOCHS):
        loss_batch = []
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=PIN_MEM), y.to(device, non_blocking=PIN_MEM)
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
    print(LRw, LRwh)
    plt.semilogy(losses_bgd, label='BGD')

    acc = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=PIN_MEM), y.to(device, non_blocking=PIN_MEM)
            out = torch.cat(((x @ w).tanh(), BIAS), dim=1) @ wh
            _, pred = out.max(1)
            acc.append((pred == y).float().mean().item())
    print(f'Accuracy: {np.mean(acc) * 100:.4}%')


run_bgd()

# plt.gca().text(.5, .8, f'$LR={LR}$', c='orange', size=15, transform=plt.gca().transAxes)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.tight_layout()
# plt.show()


from bgd import BGD


class basic_mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(785, 100, bias=False)
        self.act = F.tanh
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)


m = basic_mlp().to(device)

opt = BGD(m, F.cross_entropy, lr=LR, _eps=1e-11, bounce_th=0.9)


def run_BGD():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_loader = DataLoader(train_data, batch_size=BS, num_workers=0, shuffle=True, pin_memory=PIN_MEM, generator=torch.Generator().manual_seed(SEED))
    test_loader = DataLoader(test_data, batch_size=BS, num_workers=0, pin_memory=PIN_MEM)

    for _ in trange(EPOCHS):
        opt.train(train_loader)

    acc = opt.test(test_loader)
    print(f'Accuracy: {acc:.4}%')


# run_BGD()
