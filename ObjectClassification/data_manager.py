#!D:/Miniconda3/envs/cv/python
# -*- encoding: utf-8 -*-
"""
################################################################################
@File              :   data_manager.py
@Time              :   2021/03/07 21:57:23
@Author            :   l-yangly
@Email             :   522927317@qq.com
@Version           :   1.0
@Desc              :   None
################################################################################
"""

from PIL import Image
import numpy as np
import pickle
import os
from tqdm import trange
from os.path import join


def my_mkdirs(path):

    if not os.path.exists(path):
        os.makedirs(path)


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')

    return dict


# settings
src_dir = './data/cifar-100-python'  # the dir you uncompress the dataset
dst_dir = './data/new-cifar-100'  # the dir you want the img_dataset to be


if __name__ == '__main__':
    meta = unpickle(join(src_dir, 'meta'))  # KEYS: {'fine_label_names', 'coarse_label_names'}
    my_mkdirs(dst_dir)

    for data_set in ['train', 'test']:
        print('Unpickling {} dataset......'.format(data_set))
        data_dict = unpickle(join(src_dir, data_set))  # KEYS: {'filenames', 'batch_label', 'fine_labels', 'coarse_labels', 'data'}
        my_mkdirs(join(dst_dir, data_set))

        for fine_label_name in meta['fine_label_names']:
            my_mkdirs(join(dst_dir, data_set, fine_label_name))

        for i in trange(data_dict['data'].shape[0]):
            img = np.reshape(data_dict['data'][i], (3, 32, 32))
            i0 = Image.fromarray(img[0])
            i1 = Image.fromarray(img[1])
            i2 = Image.fromarray(img[2])
            img = Image.merge('RGB', (i0, i1, i2))
            img.save(join(dst_dir, data_set, meta['fine_label_names'][data_dict['fine_labels'][i]], data_dict['filenames'][i]))

    print('All done.')

