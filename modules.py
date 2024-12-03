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
    def __init__(self, p=0.5, bias=True):
        """
        p: Probability of selecting gradients to be scaled down.
        bias: Whether to apply the freezing and scaling to the bias gradients as well.

        Example usage:
        self.freeze_connect = FreezeConnect(p=0.5)
        self.freeze_connect(self.fc)
        self.freeze_connect(self.conv)
        """
        super().__init__()
        if not (0 < p < 1):
            raise ValueError("p must be a value between 0 and 1 (exclusive)")

        self.p = p
        self.bias = bias

    def grad_freeze_hook_w(self, grad: torch.Tensor) -> torch.Tensor:
        # Create a binary mask with probability "p" of zeroing out the gradient:
        mask = (torch.rand_like(grad) > self.p).float().flatten(1)  # or: mask = torch.bernoulli(grad, 1 - self.p)

        # Scale remaining gradients to keep the overall gradient magnitude consistent:
        scale = mask.size(1) / (mask.sum(1) + 1e-8)
        mask = (mask * scale.unsqueeze(1)).view(grad.shape)

        return grad * mask  # Soft FreezeConnect: return grad * (mask + (1 - mask) * self.slow_factor)

    def grad_freeze_hook_b(self, grad: torch.Tensor) -> torch.Tensor:
        mask = (torch.rand_like(grad) > self.p).float()
        scale = len(mask) / (mask.sum() + 1e-8)

        return grad * mask * scale

    def forward(self, module: nn.Conv2d | nn.Linear):
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            raise TypeError(
                f"Expected module to be an instance of nn.Conv2d or nn.Linear, but got {type(module).__name__}."
            )

        # Register (backward) hook for weights:
        module.weight.register_hook(self.grad_freeze_hook_w)

        # Register hook for bias if applicable and "bias" is True:
        if self.bias and module.bias is not None:
            module.bias.register_hook(self.grad_freeze_hook_b)

        return module
