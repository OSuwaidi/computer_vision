import torch
import torch.nn as nn
import torch.nn.functional as F


class FlexPool(nn.Module):
    def __init__(self, layer_shape: int):
        super().__init__()
        self.fp_weights = nn.Parameter(torch.zeros(layer_shape, layer_shape))  # FlexPool weights

    def forward(self, x):
        fp = self.fp_weights.flatten().softmax(0).view(self.fp_weights.shape)
        return (x * fp).sum((2, 3))


class DeepFlexPool(nn.Module):
    def __init__(self, num_channels: int, layer_shape: int):
        super().__init__()
        self.fp_weights = nn.Parameter(torch.zeros(num_channels, layer_shape, layer_shape))  # FlexPool weights

    def forward(self, x):
        fp =  self.fp_weights.view(self.fp_weights.size(0), -1).softmax(1).view(self.fp_weights.shape)
        return (x * fp).sum((2, 3))


# EfficientNet Block
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(MBConv, self).__init__()
        self.stride = stride
        self.expanded_channels = in_channels * expansion_factor

        # Pointwise expansion
        self.conv1 = nn.Conv2d(in_channels, self.expanded_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.expanded_channels)

        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(self.expanded_channels, self.expanded_channels, kernel_size=3, stride=stride, padding=1, groups=self.expanded_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expanded_channels)

        # Pointwise projection
        self.pointwise_conv = nn.Conv2d(self.expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip_connection(x)

        # Forward pass through the block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.depthwise_conv(x))
        x = self.bn3(self.pointwise_conv(x))

        # Add the skip connection
        x += identity
        return F.silu(x)  # Swish activation (SiLU)


# EfficientNet Model
class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(32))
        self.mbconv1 = MBConv(32, 16, expansion_factor=1, stride=1)
        self.mbconv2 = MBConv(16, 24, expansion_factor=6, stride=2)
        self.mbconv3 = MBConv(24, 40, expansion_factor=6, stride=2)
        self.mbconv4 = MBConv(40, 80, expansion_factor=6, stride=2)
        self.mbconv5 = MBConv(80, 112, expansion_factor=6, stride=1)
        self.mbconv6 = MBConv(112, 192, expansion_factor=6, stride=2)
        self.mbconv7 = MBConv(192, 320, expansion_factor=6, stride=1)
        self.conv2 = nn.Sequential(nn.Conv2d(320, 1280, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(1280))
        # self.fp = FlexPool(7)
        # self.dfp = DeepFlexPool(1280, 7)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        x = F.relu(self.conv2(x))  # (7, 7) for (224, 224) input

        # Global average pooling:
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # FlexPooling:
        # x = self.fp(x)

        # DeepFlexPooling:
        # x = self.dfp(x)

        x = self.dropout(x)  # Apply dropout before the fully connected layer
        x = self.fc(x)  # Final fully connected layer for classification
        return x


# Example usage
torch.manual_seed(0)
if __name__ == "__main__":
    model = EfficientNet(num_classes=6)
    print(model)

    # Test the model with a random input
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape, output.norm())  # Should output torch.Size([1, num_classes])
