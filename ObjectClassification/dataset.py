#!D:/Miniconda3/envs/cv/python
# -*- encoding: utf-8 -*-
"""
################################################################################
@File              :   dataset.py
@Time              :   2021/02/26 10:55:34
@Author            :   l-yangly
@Email             :   522927317@qq.com
@Version           :   1.0
@Desc              :   None
################################################################################
"""

import os
import pickle
import time

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
from utils import compute_mean_std


class CIFAR100Dataset(data.Dataset):
    """
        读取数据、初始化数据
    """
    def __init__(self, root, train=True, transform=None, target_transform=None):

        super(CIFAR100Dataset, self).__init__()

        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'test')

        self.images, self.targets, self.category, self.length = self._load_image(root)

    def __getitem__(self, index):

        image, target = self.images[index], int(self.targets[index])

        img = cv2.imread(image)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):

        return self.length

    def _load_image(self, root):
        images = []
        targets = []
        category = {}
        length = 0
        for dirs in sorted(os.listdir(root)):  # 名字按顺序排列，便于分类排序

            if dirs not in category:
                category[dirs] = len(category)
            for file in os.listdir(os.path.join(root, dirs)):
                if file.endswith('.png'):
                    filename = os.path.join(root, dirs, file)
                    images.append(filename)
                    targets.append(category[dirs])
                    length += 1

        return images, targets, category, length


class CIFAR100DataLoader(data.DataLoader):

    def __init__(self, dataset, batch_size=64, shuffle=True, num_workers=0):

        super(CIFAR100DataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        # 假设的dataset返回两个数据项: x和y
        # 那么传入collate_fn的参数定义为data, 则其shape为(batch_size, 2,…)
        # 自定义数据堆叠过程
        # 自定义batch数据的输出形式
        # 不同长度的数据，用来补齐数据长度
        images = []
        targets = []
        for image, target in batch:
            images.append(image)
            targets.append(target)
        images = torch.stack(images)
        targets = torch.LongTensor(targets)
        return images, targets


def get_CIFAR_100_dataloader(
    root,
    batch_size,
    shuffle,
    num_workers,
    train=True,
    normalize=True,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5)
):

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),

    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if normalize:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    if train:
        dataset = CIFAR100Dataset(root, train=True, transform=transform_train)
    else:
        dataset = CIFAR100Dataset(root, train=False, transform=transform_test)

    data_loader = CIFAR100DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return data_loader


if __name__ == '__main__':

    root = r'./data/cifar100'
    batch_size = 20

    # 训练数据和测试数据的装载
    train_loader = get_CIFAR_100_dataloader(
                        root,
                        batch_size,
                        shuffle=True,
                        num_workers=0,
                        train=True,
                        normalize=False,
                        mean=(0, 0, 0),
                        std=(1, 1, 1),
                    )
    # # --图像的均值和方差------------------------------------------------
    # train_mean, train_std = compute_mean_std(train_loader, normalize=True)
    # print(train_mean, train_std)

    # # --可视化,使用OpenCV----------------------------------------------
    (images, lables) = next(iter(train_loader))
    print(images.shape, lables)

    img = torchvision.utils.make_grid(images, nrow=10)
    img = img.numpy().transpose(1, 2, 0)  # OpenCV默认为BGR，这里img为RGB，因此需要对调img[:,:,::-1]
    cv2.imshow('img', img[:, :, ::-1])
    cv2.waitKey(0)
