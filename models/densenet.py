# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ و بهِ نَستَعين

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
                nn.Dropout2d(0.25)
            ))
            in_channels += growth_rate
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(2, 2)
        )

    def forward(self, x):
        return self.layer(x)


class DenseNet(nn.Module):
    def __init__(self, growth_rate, block_config, num_classes=10):
        super(DenseNet, self).__init__()
        in_channels = 64
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=3, padding=1)
        self.dense_blocks = nn.ModuleList([])
        for i, num_layers in enumerate(block_config):
            out_channels = in_channels + growth_rate * num_layers
            self.dense_blocks.append(DenseBlock(in_channels, growth_rate, num_layers))
            if i != len(block_config) - 1:
                self.dense_blocks.append(Transition(out_channels, out_channels // 2))
                in_channels = out_channels // 2
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.dense_blocks:
            x = block(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        x = self.fc(x)
        return x
