#!D:\Miniconda3\envs\cv\python
# -*- encoding: utf-8 -*-
"""
################################################################################
@File              :   InceptionV2
@Time              :   2021/02/23 09:37:50
@Author            :   l-yangly
@Email             :   522927317@qq.com
@Version           :   1.0
@Desc              :   None
################################################################################
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchsummary import summary


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n3x3_dbl_reduce, n3x3_dbl, pool_proj):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, n1x1, kernel_size=1, stride=1, padding=0)

        self.branch3x3_1 = BasicConv2d(in_channels, n3x3_reduce, kernel_size=1, stride=1, padding=0)
        self.branch3x3_2 = BasicConv2d(n3x3_reduce, n3x3, kernel_size=3, stride=1, padding=1)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, n3x3_dbl_reduce, kernel_size=1, stride=1, padding=0)
        self.branch3x3dbl_2 = BasicConv2d(n3x3_dbl_reduce, n3x3_dbl, kernel_size=3, stride=1, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(n3x3_dbl, n3x3_dbl, kernel_size=3, stride=1, padding=1)

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch3x3_1(x)
        branch5x5 = self.branch3x3_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, dim=1)


class InceptionB(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n3x3_dbl_reduce, n3x3_dbl, pool_proj):
        super(InceptionB, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, n1x1, kernel_size=1, stride=1, padding=0)

        self.branch3x3_1 = BasicConv2d(in_channels, n3x3_reduce, kernel_size=1, stride=1, padding=0)
        self.branch3x3_2 = BasicConv2d(n3x3_reduce, n3x3, kernel_size=3, stride=1, padding=1)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, n3x3_dbl_reduce, kernel_size=1, stride=1, padding=0)
        self.branch3x3dbl_2 = BasicConv2d(n3x3_dbl_reduce, n3x3_dbl, kernel_size=3, stride=1, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(n3x3_dbl, n3x3_dbl, kernel_size=3, stride=1, padding=1)

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch3x3_1(x)
        branch5x5 = self.branch3x3_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, dim=1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n3x3_dbl_reduce, n3x3_dbl):
        super(InceptionC, self).__init__()

        self.branch3x3_1 = BasicConv2d(in_channels, n3x3_reduce, kernel_size=1, stride=1, padding=0)
        self.branch3x3_2 = BasicConv2d(n3x3_reduce, n3x3, kernel_size=3, stride=2, padding=1)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, n3x3_dbl_reduce, kernel_size=1, stride=1, padding=0)
        self.branch3x3dbl_2 = BasicConv2d(n3x3_dbl_reduce, n3x3_dbl, kernel_size=3, stride=1, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(n3x3_dbl, n3x3_dbl, kernel_size=3, stride=2, padding=1)

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):

        branch5x5 = self.branch3x3_1(x)
        branch5x5 = self.branch3x3_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool(x)

        outputs = [branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, dim=1)


class InceptionV2(nn.Module):
    r"""Overview
    Input - 3x224x224     (Input)

    C1             - 64@112x112  (Conv, kernel=7x7, stride=2, padding=3, ReLU)
    P              - 64@56x56    (MaxPool, kernel=3x3, stride=2)
    C2_3x3reduce   - 64@56x56    (Conv, kernel=1x1, stride=1, padding=0, ReLU)
   *C2             - 192@56x56   (Conv, kernel=3x3, stride=1, padding=1, ReLU)
    P              - 192@28x28   (MaxPool, kernel=3x3, stride=2)
    inceptionA(3a) - 256@28x28   (#1x1@64  #3x3-reduce@64  #3x3@64  #double3x3-reduce@64  #double3x3@96  avg@32)
    inceptionA(3b) - 320@28x28   (#1x1@64  #3x3-reduce@64  #3x3@64  #double3x3-reduce@64  #double3x3@96  avg@64)
    # 每分支最后一层stride=2 (包含最大池化层)
    inceptionC(3c) - 576@28x28/2 (#1x1@0   #3x3-reduce@128 #3x3@160 #double3x3-reduce@64  #double3x3@96  max+passThrough)
    inceptionA(4a) - 576@14x14   (#1x1@224 #3x3-reduce@64  #3x3@96  #double3x3-reduce@96  #double3x3@128 avg@128)
    inceptionA(4b) - 576@14x14   (#1x1@192 #3x3-reduce@96  #3x3@128 #double3x3-reduce@96  #double3x3@128 avg@128)
    # inceptionA(4c) avg@128 有问题 改为96
    inceptionA(4c) - 576@14x14   (#1x1@160 #3x3-reduce@128 #3x3@160 #double3x3-reduce@128 #double3x3@160 avg@96)
    # inceptionA(4c) avg@128 有问题 改为96
    inceptionA(4d) - 576@14x14   (#1x1@96  #3x3-reduce@128 #3x3@192 #double3x3-reduce@160 #double3x3@192 avg@96)
    # 每分支最后一层stride=2 (包含最大池化层)
    inceptionC(4e) - 1024@14x14/2(#1x1@0   #3x3-reduce@128 #3x3@192 #double3x3-reduce@192 #double3x3@256 max+passThrough)
    inceptionA(5a) - 1024@7x7    (#1x1@352 #3x3-reduce@192 #3x3@320 #double3x3-reduce@160 #double3x3@224 avg@128)
    inceptionB(5b) - 1024@7x7    (#1x1@352 #3x3-reduce@192 #3x3@320 #double3x3-reduce@192 #double3x3@224 max@128)
    P              - 1024@1x1    (AvgPool, kernel=7x7, stride=1)
    F              - 1000        (Linear, ReLU, Softmax)
    """
    def __init__(self, num_class=1000):
        super().__init__()

        self.block1 = nn.Sequential(
            BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block2 = nn.Sequential(
            BasicConv2d(64, 64, kernel_size=1),
            BasicConv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self. block3 = nn.Sequential(
            InceptionA(192, 64, 64, 64, 64, 96, 32),
            InceptionA(256, 64, 64, 96, 64, 96, 64),
            InceptionC(320, 0, 128, 160, 64, 96)
        )

        self.block4 = nn.Sequential(
            InceptionA(576, 224, 64, 96, 96, 128, 128),
            InceptionA(576, 192, 96, 128, 96, 128, 128),
            InceptionA(576, 160, 128, 160, 128, 160, 96),
            InceptionA(576, 96, 128, 192, 160, 192, 96),
            InceptionC(576, 0, 128, 192, 192, 256)
        )

        self.block5 = nn.Sequential(
            InceptionA(1024, 352, 192, 320, 160, 224, 128),
            InceptionB(1024, 352, 192, 320, 192, 224, 128)
        )

        self.max_pool6 = nn.MaxPool2d(kernel_size=7, stride=1)

        self.fc7 = nn.Linear(1024, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.max_pool6(x)
        x = x.view(x.size(0), -1)
        x = self.fc7(x)
        return x


if __name__ == "__main__":

    net = InceptionV2(num_class=1000)
    summary(net, input_size=(3, 224, 224), batch_size=-1, device='cpu')

    x = torch.zeros((2, 3, 224, 224))

    x = net(x)

    print("output:", x.shape)

    # model = torchvision.models.inception

    # summary(model, input_size=(3, 224, 224), batch_size=-1, device='cpu')
