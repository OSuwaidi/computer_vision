from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import math
import torch


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        activation: nn.Module = nn.ReLU(inplace=True),
        gate: nn.Module = nn.Sigmoid()
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if reduction < 1:
            raise ValueError("reduction must be >= 1")

        hidden = max(1, in_channels // reduction)

        self.pool = GeM()
        self.excite = nn.Parameter(torch.tensor(2.))
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True),
            activation,
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=True),
            gate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze: global context per channel
        s = self.pool(x)
        # Excitation: learn per-channel gates
        w = self.fc(s) * self.excite
        # Scale: channel-wise reweighting (broadcast over H, W)
        return x * w


class ConvBnAct(nn.Module):
    def __init__(self, in_dims, out_dims, k, s, p, act=True, reflect=False):
        super().__init__()
        p_mode = "zeros" if not reflect else "reflect"
        self.conv = nn.Conv2d(in_dims, out_dims, kernel_size=k, stride=s, padding=p, bias=False, padding_mode=p_mode)
        self.bn = nn.BatchNorm2d(out_dims)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GeM(nn.Module):
    """Generalized Mean Pooling"""
    def __init__(self, p=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.Tensor([p]))
        self.eps = 1e-5

    def forward(self, x):
        # clamp to avoid zeros then raise to power p, mean, then root
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)


class BasicBlock(nn.Module):
    def __init__(self, in_dims, out_dims, kernel=3, stride=1, use_SE=False):
        super().__init__()
        self.conv1 = ConvBnAct(in_dims, out_dims, k=kernel, s=stride, p=kernel//2)
        self.conv2 = ConvBnAct(out_dims, out_dims, k=kernel, s=1, p=kernel//2, act=False)
        self.SE = SqueezeExcite(out_dims) if use_SE else nn.Identity()

        if stride != 1 or in_dims != out_dims:
            self.shortcut = ConvBnAct(in_dims, out_dims, k=1, s=stride, p=0, act=False)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.SE(out)
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class CosineClassifier(nn.Module):
    def __init__(self, embedding_dims, num_classes, s=64.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dims))
        nn.init.xavier_uniform_(self.weight)
        self.s = s

    def forward(self, x, _):  # x should be L2-normalized
        W = F.normalize(self.weight, dim=1)
        logits = F.linear(x, W)      # cosine similarities
        return logits * self.s       # scale up for CE


class ArcMarginProduct(nn.Module):
    """
    Implements ArcFace: additive angular margin.
    Given L2-normalized features and weights, modifies cosine logits.
    """
    def __init__(self, embedding_size, num_classes, s=64.0, m=0.50, easy_margin=False):
        super().__init__()
        self.s = s
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # threshold
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        # Normalize weights
        W = F.normalize(self.weight, dim=1)

        # cos(theta): cosine similarity
        cosine = F.linear(embeddings, W)  # [B, C]
        # sin(theta) = sqrt(1 - cos^2)
        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, min=0.0))

        # cos(theta + m) = cos*cos_m - sin*sin_m
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # target logit gets phi; others remain cosine
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.s


class FaceNet(nn.Module):
    def __init__(self,
                 in_dims=3,
                 out_dims=64,
                 use_SE=False,
                 blocks_per_stage=(2, 2, 2),
                 num_classes=1000,
                 embedding_size=128,
                 classifier_head="cosine"):
        super().__init__()
        # Pre-processing:
        # self.conv0 = nn.Sequential(ConvBnAct(in_dims, in_dims, k=3, s=1, p=1, reflect=True), nn.MaxPool2d(3))

        # Stem layer:
        self.stem = nn.Sequential(
            ConvBnAct(in_dims, out_dims, k=3, s=2, p=1, reflect=True),
            ConvBnAct(out_dims, out_dims, k=3, s=1, p=1),
            nn.MaxPool2d(kernel_size=3, stride=2)  # maybe use GeM?
        )

        c = out_dims
        BB = partial(BasicBlock, use_SE=use_SE)
        backbone = [BB(c, c) for _ in range(blocks_per_stage[0])]
        for num_blocks in blocks_per_stage[1:]:
            c_out = c*2
            layers = [BB(c, c_out, stride=2)] + [BB(c_out, c_out) for _ in range(num_blocks - 1)]
            backbone.extend(layers)
            c = c_out
        self.backbone = nn.Sequential(*backbone)

        # Global pooling and embedding
        self.global_pool = GeM()
        self.fc = nn.Linear(c, embedding_size)

        # Classification head
        if classifier_head == "cosine":
            self.classifier = CosineClassifier(embedding_size, num_classes)
        elif classifier_head == "arc":
            self.classifier = ArcMarginProduct(embedding_size, num_classes)
        else:
            print("Choose between ")

        self._init_weights()

    def _init_weights(self):
        # He init for convs, BN gamma=1, beta=0. Linear with xavier
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # def pre_process(self, x):
    #     return self.conv0(x) + F.avg_pool2d(x, 3)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Backbone
        x = self.backbone(x)

        # Embedding generation
        x = self.global_pool(x).flatten(1)
        return F.normalize(self.fc(x), p=2, dim=1)

    def classify(self, embedding, y):
        # Classification
        logits = self.classifier(embedding, y)
        return logits  # (BS, num_classes)
