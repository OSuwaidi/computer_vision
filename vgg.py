# بسم الله الرحمن الرحيم وبه نستعين

import torch.nn as nn

# Configuration:
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, in_dims, num_classes, vgg_type):
        super().__init__()
        self.in_dims = in_dims
        self.features = self._make_layers(cfg[vgg_type])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)  # Flatten
        return self.classifier(out)

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_dims
        for depth in cfg:
            if depth == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([nn.Conv2d(in_channels, depth, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.BatchNorm2d(depth)])
                in_channels = depth
        return nn.Sequential(*layers)


def vgg11(in_dims=3, num_classes=10):
    return VGG(in_dims, num_classes, 'VGG11')


def vgg13(in_dims=3, num_classes=10):
    return VGG(in_dims, num_classes, 'VGG13')


def vgg16(in_dims=3, num_classes=10):
    return VGG(in_dims, num_classes, 'VGG16')


def vgg19(in_dims=3, num_classes=10):
    return VGG(in_dims, num_classes, 'VGG19')
