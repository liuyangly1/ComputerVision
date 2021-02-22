#!D:\Miniconda3\envs\cv\python
# -*- encoding: utf-8 -*-
"""
################################################################################
@File              :   train.py
@Time              :   2021/02/22 13:20:23
@Author            :   l-yangly
@Email             :   522927317@qq.com
@Version           :   1.0
@Desc              :   None
################################################################################
"""

import argparse
import os
import random
import time
from datetime import datetime

import warnings

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from utils import adjust_learning_rate
from utils import save_checkpoint
from utils import train
from utils import validate
from utils import get_network
from utils import get_cifar100_training_dataloader, get_cifar100_test_dataloader

# # mixed_precision = True
# # try:
# #     from apex import amp
# # except Error:
# #     mixed_precision = False
# #     warnings.warn("Warning: Apex tool not install.")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR', default='data',
#                     help='path to dataset')
parser.add_argument('-m', '--comment', metavar='COMMENT', default='',
                    help='reason for training')
parser.add_argument('--seed', metavar='N', default=0, type=int,
                    help='seed for training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='AlexNet',
                    help='model architecture (default: alexnet)')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                    help='shuffle data for train.')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('--opt_level', default="O1", type=str,
#                     help="Choose which accuracy to train. (default: 'O1')")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--num_classes', type=int, default=100,
                    help="number of dataset category.")
# parser.add_argument('--world-size', default=-1, type=int,
#                     help='number of nodes for distributed training')
# parser.add_argument('--rank', default=-1, type=int,
#                     help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str,
#                     help='distributed backend')
# parser.add_argument('--seed', default=None, type=int,
#                     help='seed for initializing training. ')
# parser.add_argument('--multiprocessing-distributed', action='store_true',
#                     help='Use multi-processing distributed training to launch '
#                          'N processes per node, which has N GPUs. This is the '
#                          'fastest way to use PyTorch for either single node or '
#                          'multi node data parallel training')
best_acc1 = 0


def main():
    # 参数
    args = parser.parse_args()

    # 设置工作时间
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.job_name = f"{start_time}_{args.comment}" if args.comment is not None else start_time

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # gpu
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        print(f"Use GPU: {args.gpu} for training!")

    global best_acc1

    model = get_network(args)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # auto-tuner自动寻找最适合当前配置的高效算法增加训练效率。
    # 如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true 可以增加运行效率；
    # 如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
    cudnn.benchmark = True

    # Data loading code
    cifar100_training_loader = get_cifar100_training_dataloader(args)

    cifar100_test_loader = get_cifar100_test_dataloader(args)

    if args.evaluate:
        top1 = validate(cifar100_test_loader, model, criterion, args)
        with open('res.txt', 'w') as f:
            print(f"Acc@1: {top1}", file=f)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        acc1, losses = train(cifar100_training_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(cifar100_test_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if (epoch+1) % 100 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                # 'amp': amp.state_dict(),
            }, is_best, "./checkpoint/checkpoint_epoch%s_loss%s_acc%s.pth" % (epoch + 1, losses, acc1))


if __name__ == "__main__":
    main()
