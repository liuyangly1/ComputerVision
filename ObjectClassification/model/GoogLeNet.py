#!D:\Miniconda3\envs\cv\python
# -*- encoding: utf-8 -*-
"""
################################################################################
@File              :   GoogLeNet.py
@Time              :   2021/02/20 17:14:35
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
import torchvision


class inception(nn.Module):

    r"""Overview
    inception = [conv1x1,conv3x3(conv1x1), conv5x5(conv1x1), conv1x1(maxpool3x3)]
    """
    def __init__(self, in_channel, conv1x1, conv3x3_reduce, conv3x3, conv5x5_reduce, conv5x5, pool_proj):
        super().__init__()

        self.branch1 = nn.Conv2d(in_channel, conv1x1, (1, 1), 1, 0)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, conv3x3_reduce, (1, 1), 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv3x3_reduce, conv3x3, (3, 3), 1, 1),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, conv5x5_reduce, (1, 1), 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv5x5_reduce, conv5x5, (5, 5), 1, 2),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(

            nn.MaxPool2d((3, 3), 1, 1),  # TODO: 不跟ReLU
            nn.Conv2d(in_channel, pool_proj, (1, 1), 1, 0),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)

        return x


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 2048
        x = F.dropout(x, 0.7, training=self.training)
        # N x 2048
        x = self.fc2(x)
        # N x 1024

        return x


class GoogLeNet(nn.Module):
    r"""Overview
    Input - 3x224x224     (Input)

    C1            - 64@112x112  (Conv, kernel=7x7, stride=2, padding=3, ReLU)
    P             - 64@56x56    (MaxPool, kernel=3x3, stride=2)
    C2_reduce     - 64@56x56    (Conv, kernel=1x1, stride=1, padding=0, ReLU)
   *C2            - 192@56x56   (Conv, kernel=3x3, stride=1, padding=1, ReLU)
    P             - 192@28x28   (MaxPool, kernel=3x3, stride=2)
    inception(3a) - 256@28x28   (#1x1@64  #3x3-reduce@96  #3x3@128 #5x5-reduce@16 #5x5@32  poolProj@32)
    inception(3b) - 480@28x28   (#1x1@128 #3x3-reduce@128 #3x3@192 #5x5-reduce@32 #5x5@96  poolProj@64)
    P             - 480@14x14   (MaxPool, kernel=3x3, stride=2)
    inception(4a) - 512@14x14   (#1x1@192 #3x3-reduce@96  #3x3@208 #5x5-reduce@16 #5x5@48  poolProj@64)
    inception(4b) - 512@14x14   (#1x1@160 #3x3-reduce@112 #3x3@224 #5x5-reduce@24 #5x5@64  poolProj@64)
    inception(4c) - 512@14x14   (#1x1@128 #3x3-reduce@128 #3x3@256 #5x5-reduce@24 #5x5@64  poolProj@64)
    inception(4d) - 528@14x14   (#1x1@112 #3x3-reduce@144 #3x3@288 #5x5-reduce@32 #5x5@64  poolProj@64)
    inception(4E) - 832@14x14   (#1x1@256 #3x3-reduce@160 #3x3@320 #5x5-reduce@32 #5x5@128 poolProj@128)
    P             - 832@7x7     (MaxPool, kernel=3x3, stride=2)
    inception(5a) - 832@7x7     (#1x1@256 #3x3-reduce@160 #3x3@320 #5x5-reduce@32 #5x5@128 poolProj@128)
    inception(5b) - 1024@7x7    (#1x1@384 #3x3-reduce@192 #3x3@384 #5x5-reduce@48 #5x5@128 poolProj@128)
    P             - 1024@1x1    (AdaptiveAvgPool2d, output_size=1x1)TODO:(AvgPool, kernel=7x7, stride=1)
                                (Dropout(0.4))
    F             - 1000        (Linear, ReLU, Softmax)
    """
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=True):
        super().__init__()
        self.aux_logits = aux_logits

        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, 64, (7, 7), 2, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), 2, 1),
            nn.Conv2d(64, 64, (1, 1), 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, (3, 3), 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), 2, 1)
        )
        self.feature = nn.Sequential(
            # ('inception(3a)', inception(64, 64, 96, 128, 16, 32, 32)),
            # ('inception(3b)', inception(256, 128, 128, 192, 32, 96, 64)),
            # ('max pool', nn.MaxPool2d((3, 3), (2, 2)))
            inception(192, 64, 96, 128, 16, 32, 32),
            inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d((3, 3), 2, 1),
            inception(480, 192, 96, 208, 16, 48, 64),
            inception(512, 160, 112, 224, 24, 64, 64),
            inception(512, 128, 128, 256, 24, 64, 64),
            inception(512, 112, 144, 288, 32, 64, 64),
            inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d((3, 3), 2, 1),
            inception(832, 256, 160, 320, 32, 128, 128),
            inception(832, 384, 192, 384, 48, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(p=0.4)
        )

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.classify = nn.Sequential(
            nn.Linear(1024*1*1, num_classes),
            # nn.Sigmoid()
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x


if __name__ == "__main__":

    net = GoogLeNet()

    summary(net, input_size=(3, 224, 224), batch_size=-1, device='cpu')

    x = torch.zeros((2, 3, 224, 224))
    print("input:", x.shape)
    x = net(x)
    print("output:", x.shape)

    # 需注释掉BN层
    # model = torchvision.models.googlenet(aux_logits=False)

    # summary(model, input_size=(3, 224, 224), batch_size=-1, device='cpu')
    # Linear-139                 [-1, 1000]       1,025,000
    # Input size (MB): 0.57
    # Forward/backward pass size (MB): 69.49
    # Params size (MB): 25.22
    # Estimated Total Size (MB): 95.28