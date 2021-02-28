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
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_CIFAR_100(root, train=True, fine_label=True):
    """
    root,文件名
    train  训练数据集时取True，测试集时取False
    fine_label  如果分类为100类时取True，分类为20类时取False
     """

    if train:
        filename = os.path.join(root, "train")
    else:
        filename = os.path.join(root, "test")
    # 读取图片
    with open(filename, 'rb')as f:

        datadict = pickle.load(f, encoding='bytes')

        X = datadict[b'data']

        if train:
            # [50000, 32, 32, 3]
            X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        else:
            # [10000, 32, 32, 3]
            X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        # fine_labels细分类，共100中类别
        # coarse_labels超级类，共20中类别，每个超级类中实际包含5种fine_labels
        # 如trees类中，又包含maple, oak, palm, pine, willow，5种具体的树
        # 这里只取fine_labels
        # Y = datadict[b'coarse_labels'] + datadict[b'fine_labels']
        if fine_label:
            Y = datadict[b'fine_labels']
        else:
            Y = datadict[b'coarse_labels']

        Y = np.array(Y)

        return X, Y


class CIFAR100Dataset(Dataset):
    """
        读取数据、初始化数据
    """
    def __init__(self, root, train=True, fine_label=True, transform=None):
        super(CIFAR100Dataset, self).__init__()

        self.x, self.y = load_CIFAR_100(root, train=train, fine_label=fine_label)

        self.transform = transform

        self.train = train

    def __getitem__(self, index):

        image, target = self.x[index], int(self.y[index])

        if self.transform is not None:
            image = self.transform(image)

        # image = torch.transpose(image, 0, 2)
        # [-1, 3, 32, 32] [-1, 32]
        return image, target

    def __len__(self):

        return len(self.x)


class CIFAR100DataLoader(DataLoader):

    def __init__(self, dataset, batch_size=16, shuffle=True, num_workers=4):
        super(CIFAR100DataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        self.collate_fn = self._collate_fn

    def _collate_fn(self, data):
        # 不同长度的数据，用来补齐数据长度
        images, targets = zip(*data)

        return images, targets


def get_CIFAR_100_dataloader(
    root,
    batch_size,
    shuffle,
    num_workers,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    train=True
):

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        # transforms.Resize(224),
        transforms.ToTensor(),  # 灰度范围从0-255变换到0-1之间
        transforms.Normalize(mean, std)  # 灰度范围从0-1变换到(-1,1)
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if train:
        dataset = CIFAR100Dataset(
                        root,
                        train=True,
                        fine_label=True,
                        transform=transform_train
                    )
    else:
        dataset = CIFAR100Dataset(
                        root,
                        train=False,
                        fine_label=True,
                        transform=transform_test
                    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,  # 一个批次可以认为是一个包，每个包中含有batch_size张图片
        shuffle=shuffle,
        num_workers=num_workers
    )

    return data_loader


if __name__ == '__main__':

    root = r'./data/cifar-100-python'
    batch_size = 20

    # 训练数据和测试数据的装载
    train_loader = get_CIFAR_100_dataloader(
                        root,
                        batch_size,
                        shuffle=True,
                        num_workers=0,
                        mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5),
                        train=True
                    )

    # 这里train_loader包含:batch_size、dataset等属性，数据类型分别为int，Dataset
    # dataset中又包含train_labels, train_set等属性;  数据类型均为ndarray
    print(f'train_loader.batch_size: {train_loader.batch_size}\n')
    print(f'train_loader.dataset.x.shape: {train_loader.dataset.x.shape}\n')
    print(f'train_loader.dataset.y.shape: {train_loader.dataset.y.shape}\n')

    # --可视化,使用OpenCV----------------------------------------------
    images, lables = next(iter(train_loader))

    img = torchvision.utils.make_grid(torch.transpose(images, 1, 3), nrow=10)
    img = img.numpy().transpose(1, 2, 0)
    # OpenCV默认为BGR，这里img为RGB，因此需要对调img[:,:,::-1]
    cv2.imshow('img', img[:, :, ::-1])
    cv2.waitKey(0)
