import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
import random


class MixStyle1d(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):
        super(MixStyle1d, self).__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.mix = mix
        self.beta = torch.distributions.Beta(alpha, alpha)
        self._activated = True

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated or self.p <= 0.0:
            return x

        if random.random() > self.p:
            return x

        batch_size = x.size(0)
        mu = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        mu = mu.detach()
        sig = sig.detach()
        x_normed = (x - mu) / sig

        lam = self.beta.sample((batch_size, 1, 1)).to(x.device)

        if self.mix == "random":
            perm = torch.randperm(batch_size, device=x.device)
        elif self.mix == "crossdomain":
            perm = torch.arange(batch_size - 1, -1, -1, device=x.device)
        else:
            raise ValueError(f"Unsupported mix method: {self.mix}")

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lam + mu2 * (1 - lam)
        sig_mix = sig * lam + sig2 * (1 - lam)
        return x_normed * sig_mix + mu_mix

class SEAttention1d(nn.Module):
    '''
    Modified from https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/SEAttention.py
    '''
    def __init__(self, channel, reduction):
        super(SEAttention1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b,c,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1)

        return x*y.expand_as(x)


class macnn_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=None, stride=1, reduction=16):
        super(macnn_block, self).__init__()

        if kernel_size is None:
            kernel_size = [3, 6, 12]

        self.reduction = reduction

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size[0], stride=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size[1], stride=1, padding='same')
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size[2], stride=1, padding='same')

        self.bn = nn.BatchNorm1d(out_channels*3)
        self.relu = nn.ReLU()

        self.se = SEAttention1d(out_channels*3,reduction=reduction)

    def forward(self,x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x_con = torch.cat([x1,x2,x3], dim=1)

        out = self.bn(x_con)
        out = self.relu(out)

        out_se = self.se(out)

        return out_se

class MACNN(nn.Module):

    def __init__(self, in_channels=3, channels=64, num_classes=7, block_num=None,
                 use_mixstyle=False, mixstyle_p=0.5, mixstyle_alpha=0.1,
                 mixstyle_layers=None, mixstyle_mode="random"):#proj_dim = 128, 
        super(MACNN, self).__init__()

        if block_num is None:
            block_num = [2, 2, 2]
        if mixstyle_layers is None:
            mixstyle_layers = [1, 2]

        self.in_channel = in_channels
        self.num_classes = num_classes
        self.channel = channels
        self.use_mixstyle = use_mixstyle
        self.mixstyle_layers = set(mixstyle_layers)
        # self.proj_dim = proj_dim

        self.max_pool1 = nn.MaxPool1d(kernel_size=3,stride=2,padding=1)
        self.max_pool2 = nn.MaxPool1d(kernel_size=3, stride=2,padding=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.channel*4*3, num_classes)
        self.mixstyle = MixStyle1d(p=mixstyle_p, alpha=mixstyle_alpha, mix=mixstyle_mode)

        self.layer1 = self._make_layer(macnn_block, block_num[0], self.channel)
        self.layer2 = self._make_layer(macnn_block, block_num[1], self.channel*2)
        self.layer3 = self._make_layer(macnn_block, block_num[2], self.channel*4)

    def _make_layer(self, block, block_num, channel, reduction=16):

        layers = []
        for i in range(block_num):
            layers.append(block(self.in_channel, channel, kernel_size=None,
                                stride=1, reduction=reduction))
            self.in_channel = 3*channel

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer1(x)
        if self.use_mixstyle and 1 in self.mixstyle_layers:
            x = self.mixstyle(x)
        x = self.max_pool1(x)

        x = self.layer2(x)
        if self.use_mixstyle and 2 in self.mixstyle_layers:
            x = self.mixstyle(x)
        x = self.max_pool2(x)

        x = self.layer3(x)
        if self.use_mixstyle and 3 in self.mixstyle_layers:
            x = self.mixstyle(x)
        
        x = self.avg_pool(x)

        x = torch.flatten(x, 1)

        embedding_output = F.normalize(x)

        cls_output = self.fc(embedding_output)

        return embedding_output, cls_output

    def classify_embedding(self, embedding):
        return self.fc(F.normalize(embedding, dim=1))

