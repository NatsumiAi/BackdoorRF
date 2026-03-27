import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SEAttention1d(nn.Module):
    def __init__(self, channel, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        batch_size, channels, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1)
        return x * y.expand_as(x)


class MACNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=None, reduction=16):
        super().__init__()
        if kernel_size is None:
            kernel_size = [3, 6, 12]

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size[0], stride=1, padding="same")
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size[1], stride=1, padding="same")
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size[2], stride=1, padding="same")
        self.bn = nn.BatchNorm1d(out_channels * 3)
        self.relu = nn.ReLU()
        self.se = SEAttention1d(out_channels * 3, reduction=reduction)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return self.se(out)


class MACNN(nn.Module):
    def __init__(self, in_channels=2, channels=64, num_classes=7, block_num=None):
        super().__init__()
        if block_num is None:
            block_num = [2, 2, 2]

        self.in_channel = in_channels
        self.channel = channels

        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.max_pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.channel * 4 * 3, num_classes)

        self.layer1 = self._make_layer(block_num[0], self.channel)
        self.layer2 = self._make_layer(block_num[1], self.channel * 2)
        self.layer3 = self._make_layer(block_num[2], self.channel * 4)

    def _make_layer(self, block_num, channel, reduction=16):
        layers = []
        for _ in range(block_num):
            layers.append(MACNNBlock(self.in_channel, channel, reduction=reduction))
            self.in_channel = 3 * channel
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.max_pool1(x)
        x = self.layer2(x)
        x = self.max_pool2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        embedding_output = F.normalize(x, dim=1)
        cls_output = self.fc(embedding_output)
        return embedding_output, cls_output
