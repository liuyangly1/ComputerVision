#!D:\Miniconda3\envs\cv\python
# -*- encoding: utf-8 -*-
"""
################################################################################
@File              :   VGG.py
@Time              :   2021/02/20 09:16:28
@Author            :   l-yangly
@Email             :   522927317@qq.com
@Version           :   1.0
@Desc              :   None
################################################################################
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


r"""VGGNet

E - 19 weight layers

Input - 3x224x224     (Input)

C1    - 64@224x224      (Conv, kernel=3x3, stride=1, padding=1, ReLU)
C2    - 64@224x224      (Conv, kernel=3x3, stride=1, padding=1, ReLU)
P     - 64@112x112      (MaxPool, kernel=2x2, stride=2)
C3    - 128@112x112     (Conv, kernel=3x3, stride=1, padding=1, ReLU)
C4    - 128@112x112     (Conv, kernel=3x3, stride=1, padding=1, ReLU)
P     - 128@56x56       (MaxPool, kernel=2x2, stride=2)
C5    - 256@56x56       (Conv, kernel=3x3, stride=1, padding=1, ReLU)
C6    - 256@56x56       (Conv, kernel=3x3, stride=1, padding=1, ReLU)
C7    - 256@56x56       (Conv, kernel=3x3, stride=1, padding=1, ReLU)
C8    - 256@56x56       (Conv, kernel=3x3, stride=1, padding=1, ReLU)
P     - 256@27x27       (MaxPool, kernel=2x2, stride=2)
C9    - 512@27x27       (Conv, kernel=3x3, stride=1, padding=1, ReLU)
C10   - 512@27x27       (Conv, kernel=3x3, stride=1, padding=1, ReLU)
C11   - 512@27x27       (Conv, kernel=3x3, stride=1, padding=1, ReLU)
C12   - 512@27x27       (Conv, kernel=3x3, stride=1, padding=1, ReLU)
P     - 512@14x14       (MaxPool, kernel=2x2, stride=2)
C13   - 512@14x14       (Conv, kernel=3x3, stride=1, padding=1, ReLU)
C14   - 512@14x14       (Conv, kernel=3x3, stride=1, padding=1, ReLU)
C15   - 512@14x14       (Conv, kernel=3x3, stride=1, padding=1, ReLU)
C16   - 512@14x14       (Conv, kernel=3x3, stride=1, padding=1, ReLU)
P     - 512@14x14       (MaxPool, kernel=2x2, stride=2)
F17   - 4096          (Linear, ReLU, Dropout)
F18   - 4096          (Linear, ReLU, Dropout)
F19   - 1000          (Linear, ReLU, Softmax)
"""

# 64表示conv2d(c=64,k=3x3,s=1,p=1)  ’M’ 表示maxpooling()
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    # 'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256-1x1, 'M', 512, 512, 512-1x1, 'M', 512, 512, 512-1x1, 'M'],

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],

    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class Conv_ReLu(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class VGGNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.feature = self._make_layers(cfg)
        self.classify = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):

        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)

        return x

    def _make_layers(self, configs):

        layers = []
        in_channel = 3
        for config in configs:

            if config == 'M':
                layers.append(nn.MaxPool2d((2, 2), (2, 2)))

            else:
                layers.append(Conv_ReLu(in_channel, config))
                in_channel = config

        return nn.Sequential(*layers)


def vgg11_bn():
    return VGGNet(cfgs['A'])


def vgg13_bn():
    return VGGNet(cfgs['B'])


def vgg16_bn():
    return VGGNet(cfgs['D'])


def vgg19_bn():
    return VGGNet(cfgs['E'])


if __name__ == "__main__":

    net = vgg19_bn()
    summary(net, input_size=(3, 224, 224), batch_size=-1, device='cpu')

    x = torch.zeros((2, 3, 224, 224))

    print("input:", x.shape)

    x = net(x)

    print("output:", x.shape)

    import torchvision.models as models
    model = models.vgg19()
    summary(model, input_size=(3, 224, 224), batch_size=-1, device='cpu')
