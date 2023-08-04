'''
    setup model and datasets
'''
import copy

import numpy as np
import torch
from common_models.models import models

from dataset import *
# from advertorch.utils import NormalizeByChannelMeanStd

__all__ = ['setup_model_dataset']


def setup_model_dataset(args):

    if args.dataset == 'cifar10':
        classes = 10
        # normalization = NormalizeByChannelMeanStd(
        # mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_set_loader, val_loader, test_loader = cifar10_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers)

    elif args.dataset == 'cifar100':
        classes = 100
        # normalization = NormalizeByChannelMeanStd(
        # mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762])
        train_set_loader, val_loader, test_loader = cifar100_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers)

    else:
        raise ValueError('Dataset not supprot yet !')

    model = models[args.arch](num_classes=classes, seed=args.seed)

    # model.normalize = normalization
    # print(model)

    return model, train_set_loader, val_loader, test_loader
