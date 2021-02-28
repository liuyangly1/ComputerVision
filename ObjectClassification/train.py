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
import torch.nn as nn
import torch.optim


from dataset import get_CIFAR_100_dataloader
from utils import get_network, get_criterion, get_optimizer, accuracy
from utils import adjust_learning_rate
from utils import save_checkpoint
from utils import AverageMeter, ProgressMeter

best_acc1 = 0


def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')

    parser.add_argument('data', metavar='DIR', default='./data/cifar-100-python',
                        help='path to dataset')
    parser.add_argument('-m', '--comment', metavar='COMMENT', default='',
                        help='reason for training')
    parser.add_argument('--seed', metavar='N', default=0, type=int,
                        help='seed for training')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use,-1 for cud.')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='VGGNet',
                        help='model architecture (default: alexnet)')
    parser.add_argument('--num_classes', type=int, default=100,
                        help="number of dataset category.")
    parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N',
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
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    # parser.add_argument('--pretrained', dest='pretrained', action='store_true',
    #                     help='use pre-trained model')

    return parser.parse_args()


def train(train_loader, model, criterion, optimizer, epoch, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':4.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':4.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def main():
    # 参数
    args = parse_args()

    # 设置工作时间
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.job_name = f"{start_time}_{args.comment}" if args.comment is not None else start_time

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        warnings.warn('You have chosen to seed training.')

    # gpu
    device = torch.device("cpu")
    if torch.cuda.is_available():

        if args.gpu is not None and args.gpu != -1:
            warnings.warn('You have chosen a specific GPU.')
            print(f"Use GPU: {args.gpu} for training!")
            torch.cuda.set_device(args.gpu)
            device = torch.device("cuda:%s" % args.gpu)
        else:
            warnings.warn('You have a CUDA device, so you should probably run cuda')

    global best_acc1

    model = get_network(args.arch, args.num_classes)
    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = get_criterion()
    criterion = criterion.to(device)

    optimizer = get_optimizer(
                    model, args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay
                )

    # Data loading code
    cifar100_training_loader = get_CIFAR_100_dataloader(
                                    root=args.data,
                                    batch_size=args.batch_size,
                                    shuffle=args.shuffle,
                                    num_workers=args.num_workers,
                                    mean=(0.5, 0.5, 0.5),
                                    std=(0.5, 0.5, 0.5),
                                    train=True
                                )

    cifar100_test_loader = get_CIFAR_100_dataloader(
                                    root=args.data,
                                    batch_size=args.batch_size,
                                    shuffle=args.shuffle,
                                    num_workers=args.num_workers,
                                    mean=(0.5, 0.5, 0.5),
                                    std=(0.5, 0.5, 0.5),
                                    train=False
                                )
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
