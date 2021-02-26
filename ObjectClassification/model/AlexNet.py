#!D:\Miniconda3\envs\cv\python
# -*- encoding: utf-8 -*-
"""
################################################################################
@File              :   AlexNet.py
@Time              :   2021/02/19 13:13:26
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

# class LRN(nn.Module):
#     """LRN
#     Local response normalization

#     https://github.com/pytorch/pytorch/issues/653
#     """
#     def __init__(self, n, alpha=1e-4, beta=0.75, k=1):

#         self.average = nn.AvgPool2d(kernel_size=n, stride=1, padding=int((n-1.0)/2))
#         nn.LocalResponseNorm
#         self.alpha = alpha
#         self.beta = beta
#         self.k = k

#     def forward(self, x):

#         div = x.pow(2)
#         div = self.average(div)
#         div = div.mul(self.alpha).add(1.0).pow(self.beta)

#         x = x.div(div)

#         return x


class AlexNet(nn.Module):
    r"""AlexNet
    最大池化代替重叠池化
    LRN 局部响应归一化

    Input - 3x227x227     (Input)

    C1    - 96@55x55      (Conv, kernel=11x11, stride=4, padding=2, ReLU)
    P1    - 96@27x27      (MaxPool, kernel=3x3, stride=2)
    LRN

    Split - 2x48@27x27  2xGPU

    C2    - 2x128@27x27   (Conv, kernel=5x5, stride=1, padding=2, ReLU)
    P2    - 2x128@13x13   (MaxPool, kernel=3x3, stride=2)
    LRN

    C3    - 2x192@13x13   (Conv, kernel=3x3, stride=1, padding=0, Relu)
    C4    - 2x192@13x13   (Conv, kernel=3x3, stride=1, padding=0, Relu)
    C5    - 2x128@13x13   (Conv, kernel=3x3, stride=1, padding=0, Relu)
    P5    - 2x128@6x6     (MaxPool, kernel=3x3, stride=2)
    F6    - 4096          (Linear, ReLU, Dropout)
    F7    - 4096          (Linear, ReLU, Dropout)
    F8    - 1000          (Linear)
    """
    def __init__(self, num_classes=1000):

        super(AlexNet, self).__init__()

        self.feature = nn.Sequential(

            # layer 1
            nn.Conv2d(3, 96, (11, 11), 4, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), (2, 2)),
            nn.LocalResponseNorm(size=2, alpha=1e-4, beta=0.75, k=1.),

            # layer 2
            nn.Conv2d(96, 256, (5, 5), 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), (2, 2)),
            nn.LocalResponseNorm(size=2, alpha=1e-4, beta=0.75, k=1.),

            # layer 3
            nn.Conv2d(256, 384, (3, 3), 1, 1),
            nn.ReLU(inplace=True),

            # layer 4
            nn.Conv2d(384, 384, (3, 3), 1, 1),
            nn.ReLU(inplace=True),

            # layer 5
            nn.Conv2d(384, 256, (3, 3), 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), (2, 2))
        )

        self.classify = nn.Sequential(
            # layer 6
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # layer 7
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # layer 8
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):

        x = self.feature(x)

        x = x.view(x.size(0), -1)

        x = self.classify(x)

        return x


if __name__ == "__main__":
    net = AlexNet(num_classes=1000)
    summary(net, input_size=(3, 224, 224), batch_size=-1, device='cpu')
    x = torch.zeros((2, 3, 224, 224))
    # print(net)
    print("input: ", x.shape)
    out = net(x)
    print("output: ", out.shape)
