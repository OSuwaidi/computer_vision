# بسم الله الرحمن الرحيم و به نستعين

from typing import Union
from torchvision.ops import deform_conv2d
import torch
import torch.nn as nn
import torch.nn.init as init


class FlexPool(nn.Module):
    def __init__(self, layer_shape):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(layer_shape, layer_shape))

    def forward(self, x):
        fp = self.weights.view(-1).softmax(0).view(self.weights.shape)  # FlexPool weights
        return (x * fp).sum((2, 3))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.channel_conv = nn.Conv2d(in_channels, out_channels, 1, stride, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x)) + self.channel_conv(x)


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size**2, kernel_size, stride, padding=padding)
        init.constant_(self.offset_conv.weight, 0.)
        init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels, kernel_size**2, kernel_size, stride, padding=padding)  # confidence mask for each movement/shift
        init.constant_(self.modulator_conv.weight, 0.)
        init.constant_(self.modulator_conv.bias, 0.)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

    def forward(self, x):
        offsets = self.offset_conv(x)
        modulator = torch.sigmoid(self.modulator_conv(x))
        return deform_conv2d(x, offset=offsets, weight=self.conv.weight, bias=self.conv.bias, padding=self.padding, mask=modulator, stride=self.stride,)


class FreezeConnect(nn.Module):
    def __init__(self, p=0.5, is_bias=True):
        super().__init__()
        self.p = p
        self.is_bias = is_bias
        self.eps = 1e-11

    def grad_freeze_hook(self, grad: torch.Tensor):
        # mask = (torch.rand_like(grad) > self.p).float() / (1 - self.p)

        # Random:
        # mask = torch.bernoulli(torch.full_like(grad, 1 - self.p)) / (1 - self.p)

        # Deterministic:
        mask = torch.bernoulli(torch.full_like(grad, 1 - self.p))
        mask = (mask * mask.numel()) / (mask.count_nonzero() + self.eps)

        # OR:
        # mask = torch.rand_like(grad) > self.p  # Slow down the learning by "slow_rate" fraction
        # grad[~mask] *= slow_rate
        return mask * grad

    def forward(self, module: Union[nn.Conv2d, nn.Linear]):
        module.weight.register_hook(self.grad_freeze_hook)
        if self.is_bias:
            module.bias.register_hook(self.grad_freeze_hook)
        return module
