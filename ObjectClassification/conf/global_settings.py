""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

# CIFAR100 dataset path (python version)
ROOT = './data/cifar100'

# mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.44091782, 0.48654908, 0.50707483)
CIFAR100_TRAIN_STD = (0.2726346, 0.25357336, 0.2642534)

# net
NUM_CLASSES = 100

# total training epoches
EPOCH = 200
MILESTONES = [60, 120, 160]

# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 20

# initial learning rate
# INIT_LR = 0.1

# time of we run the script
TIME_NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

# tensorboard log dir
LOG_DIR = 'runs'









