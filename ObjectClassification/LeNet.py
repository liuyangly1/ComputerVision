#!D:\Miniconda3\envs\cv\python
# -*- encoding: utf-8 -*-
"""
################################################################################
@File              :   LeNet.py
@Time              :   2021/02/19 09:30:28
@Author            :   l-yangly
@Email             :   522927317@qq.com
@Version           :   1.0
@Desc              :   None
################################################################################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """LeNet
    不完全连接卷积未实现，用完全卷积代替

    Input - 1x32x32  (Input)
    C1    - 6@28x28  (Conv, kernel=5x5, relu)
    S2    - 6@14x14  (MaxPool,kernel=2x2, stride=2)
    C3    - 16@10x10 (Conv, kernel=5x5, relu)
    S4    - 16@5x5   (MaxPool, kernel=2x2, stride=2)
    C5    - 120@1x1  (Conv, kernel=5x5, relu)
    F6    - 84       (Linear, relu)
    F7    - 10       (Linear, LogSoftmax)
    """
    def __init__(self):
        super(LeNet, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(1, 6, (5, 5), 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(6, 16, (5, 5), 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(16, 120, (5, 5), 1, 0),
            nn.ReLU(inplace=True)
        )
        self.classify = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):

        x = self.feature(x)

        x = x.view(x.size(0), -1)

        x = self.classify(x)

        return x


if __name__ == "__main__":

    net = LeNet()
    x = torch.zeros((2, 1, 32, 32))
    print("input: ", x.shape)
    out = net(x)
    print("output: ", out.shape)
