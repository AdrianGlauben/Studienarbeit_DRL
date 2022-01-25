# Code based on : https://github.com/FrancescoSaverioZuppichini/ResNet
# and           : https://github.com/QueensGambit/CrazyAra/blob/master/DeepCrazyhouse/src/domain/neural_net/architectures/a0_resnet.py

import numpy as np
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial

class _Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

conv3x3 = partial(_Conv2dAuto, kernel_size=3, bias=False)
conv1x1 = partial(_Conv2dAuto, kernel_size=1, bias=False)



class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()


    def forward(self, x):
        residual = x
        if self.apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x


    @property
    def apply_shortcut(self):
        return self.in_channels != self.out_channels



class _ResNetResidualBlock(_ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion = expansion
        self.downsampling = downsampling
        self.conv = conv
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                stride=self.downsampling, bias=False),
            'bn': nn.BatchNorm2d(self.expanded_channels, momentum=0.9)
        })) if self.apply_shortcut else None


    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion


    @property
    def apply_shortcut(self):
        return self.in_channels != self.expanded_channels



def _conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({
        'conv': conv(in_channels, out_channels, *args, **kwargs),
        'bn': nn.BatchNorm2d(out_channels, momentum=0.9)}))


class _ResNetBasicBlock(_ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            _conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            _conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )



class _ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=_ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        downsampling = 1 # 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )


    def forward(self, x):
        x = self.blocks(x)
        return x



class _ValueHead(nn.Module):
    def __init__(self, in_features, channels, fc_size, activation=nn.ReLU):
        super().__init__()
        self.conv_block = nn.Sequential(
            _conv_bn(in_features, channels, conv=conv1x1, bias=False),
            activation(),
        )
        self.fc = nn.Sequential(
            nn.Linear(channels * 8 * 8, fc_size),
            activation(),
            nn.Linear(fc_size, 1),
            nn.Tanh(),
        )


    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class _PolicyHead(nn.Module):
    def __init__(self, in_features, channels, n_labels, activation=nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(
            _conv_bn(in_features, channels, conv=conv1x1, bias=False),
            activation(),
        )
        self.fc = nn.Sequential(
            nn.Linear(channels * 8 * 8, n_labels),
            nn.Softmax(1),
        )


    def forward(self, x):
        x = self.block(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class AlphaZeroResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        block_channels=128,
        n_labels=4096,
        channels_value_head = 2,
        fc_size_value_head = 128,
        channels_policy_head = 4,
        num_res_blocks = 8,
        bn_mom = 0.9,
        *args,
        **kwargs
    ):
        super().__init__()
        self.body = _ResNetLayer(in_channels, block_channels, n=num_res_blocks)
        self.value_head = _ValueHead(block_channels, channels_value_head, fc_size_value_head)
        self.policy_head = _PolicyHead(block_channels, channels_policy_head, n_labels)


    def forward(self, x):
        out = self.body(x)
        value = self.value_head(out)
        policy = self.policy_head(out)

        return [value, policy]
