# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ و به نستعين

from typing import Callable
import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision.models import squeezenet1_0, resnet18, mobilenet_v3_small
import torch.nn as nn
from tqdm import trange
import numpy as np
import torch.nn.functional as F
import random

T_mnist = v2.Compose([
    v2.PILToTensor(),
    v2.RandomCrop(32, padding=6, padding_mode="edge"),
    v2.RandomInvert(p=0.5),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.1)),
    v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(1., 3.))], p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5], [0.5])
])
mnist_train = datasets.MNIST('../datasets/MNIST', train=True, download=True,)
mnist_test = datasets.MNIST('../datasets/MNIST', train=False, download=True,)
mnist_data = ConcatDataset([mnist_train, mnist_test])


class TransformData(Dataset):
    def __init__(self, data, transforms):
        self.data = data
        self.T = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return self.T(x), y


mnist_data = TransformData(mnist_data, T_mnist)


T_svhn = v2.Compose([v2.PILToTensor(),
                v2.Grayscale(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.5], [0.5])
                ])

svhn = datasets.SVHN('../datasets/SVHN', split="test", download=True, transform=T_svhn)
