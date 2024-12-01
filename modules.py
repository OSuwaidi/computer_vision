# بسم الله الرحمن الرحيم و به نستعين

from torchvision.ops import deform_conv2d
import torch
import torch.nn as nn
import torch.nn.init as init


class FlexPool(nn.Module):
    def __init__(self, feature_map_shape):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(feature_map_shape, feature_map_shape))  # to apply in ResNet: feature_map_shape: int = ceil(image_shape / 4)

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
        """
        p: Probability of selecting gradients to be scaled down.
        is_bias: Whether to apply the scaling to the bias gradients as well.

        Example usage:
        self.freeze_connect = FreezeConnect(p=0.5)
        self.freeze_connect(self.fc1)
        self.freeze_connect(self.fc2)
        """
        super().__init__()
        self.p = p
        self.is_bias = is_bias

    def grad_freeze_hook(self, grad: torch.Tensor) -> torch.Tensor:
        # TODO: make the mask be per neuron (acts per row of the gradient matrix)

        # Create a binary mask with probability "p" of zeroing out the gradient:
        mask = (torch.rand_like(grad) > self.p).float()  # or: mask = torch.bernoulli(grad, 1 - self.p)

        # Scale remaining gradients to keep the overall gradient magnitude consistent:
        return grad * mask / (1 - self.p)  # or: "return grad * mask * mask.numel() / (mask.count_nonzero() + 1e-8)"

        # Soft FreezeConnect:
        # return grad * (mask + (1 - mask) * self.slow_factor)

    def forward(self, module: nn.Conv2d | nn.Linear):
        # Register hook for weights:
        module.weight.register_hook(self.grad_freeze_hook)

        # Register hook for bias if applicable and "is_bias" is True:
        if self.is_bias and module.bias is not None:
            module.bias.register_hook(self.grad_freeze_hook)

        return module
