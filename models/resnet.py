# بسم الله الرحمن الرحيم و به نستعين

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Callable


def _weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        """
        gammas = np.random.gamma(1, 0.0003, m.weight.shape)
        with torch.no_grad():
            m.weight.data = torch.from_numpy(gammas)
        """
        if m.bias is not None:
            init.constant_(m.bias, 0.01)


class Residual(nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = fn  # usually is nn.Sequential() or a Block

    def forward(self, x: torch.Tensor):
        return self.fn(x) + x


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    def __init__(self, in_dims, out_dims, act, stride=1, kernel_size=3, option='B'):
        super().__init__()
        self.act = act
        self.conv1 = nn.Conv2d(in_dims, out_dims, kernel_size, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_dims)
        self.conv2 = nn.Conv2d(out_dims, out_dims, kernel_size, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dims)

        self.shortcut = nn.Identity()

        # If shape was not preserved (not the same shape as input feature map) reduce the identity's (x) shape to match that of the processed units:
        if stride != 1 or in_dims != out_dims:
            if option == 'A':
                """
                The CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_dims // 4, out_dims // 4),
                                                  "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_dims, out_dims, 1, stride, bias=False),
                    nn.BatchNorm2d(out_dims))

    def forward(self, x):
        out = self.bn1(self.act(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.act(out)


class ResNet(nn.Module):
    def __init__(self, in_dims, num_blocks, num_classes, act, ):
        super().__init__()
        self.act = act
        # Initialize relevant parameters and convolutional residual blocks:
        self.initial_dims = 16
        self.conv1 = nn.Conv2d(in_dims, self.initial_dims, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.initial_dims)
        self.layer1 = self._make_layer(self.initial_dims, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)  # applies function on "model.modules()" (recursively)

    def _make_layer(self, out_dims, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.initial_dims, out_dims, self.act, stride, ))
            self.initial_dims = out_dims
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.act(self.conv1(x)))  # (BS, 16, H, W)
        out = self.layer1(out)  # (BS, 16, H, W)
        out = self.layer2(out)  # (BS, 32, H//2, W//2)
        out = self.layer3(out)  # (BS, 64, H//4, W//4)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)  # (BS, 64)
        return self.linear(out)


def resnet20(in_dims=3, num_classes=10, act=F.relu, ):
    return ResNet(in_dims, [3, 3, 3], num_classes, act)  # 3 stages, with 3 res-blocks per stage


def resnet32(in_dims=3, num_classes=10, act=F.relu, ):
    return ResNet(in_dims, [5, 5, 5], num_classes, act, )  # 3 stages, with 5 res-blocks per stage


def resnet44(in_dims=3, num_classes=10, act=F.relu, ):
    return ResNet(in_dims, [7, 7, 7], num_classes, act, )


def resnet56(in_dims=3, num_classes=10, act=F.relu, ):
    return ResNet(in_dims, [9, 9, 9], num_classes, act, )


def resnet110(in_dims=3, num_classes=10, act=F.relu, ):
    return ResNet(in_dims, [18, 18, 18], num_classes, act, )


def resnet1202(in_dims=3, num_classes=10, act=F.relu, ):
    return ResNet(in_dims, [200, 200, 200], num_classes, act, )
