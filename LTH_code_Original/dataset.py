'''
    function for loading datasets
    contains: 
        CIFAR-10
        CIFAR-100   
'''
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100

__all__ = ['cifar10_dataloaders', 'cifar100_dataloaders']


class DatasetSplitter(torch.utils.data.Dataset):
    """This splitter makes sure that we always use the same training/validation split"""

    def __init__(self, parent_dataset, split_start=-1, split_end=-1):
        split_start = split_start if split_start != -1 else 0
        split_end = split_end if split_end != -1 else len(parent_dataset)
        assert split_start <= len(parent_dataset) - 1 and split_end <= len(
            parent_dataset
        ) and split_start < split_end, "invalid dataset split"

        self.parent_dataset = parent_dataset
        self.split_start = split_start
        self.split_end = split_end

    def __len__(self):
        return self.split_end - self.split_start

    def __getitem__(self, index):
        assert index < len(self), "index out of bounds in split_datset"
        return self.parent_dataset[index + self.split_start]


def cifar10_dataloaders(
    batch_size=128,
    data_dir='datasets/cifar10',
    num_workers=2,
    validation_split=0.1,
):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    print(
        'Dataset information: CIFAR-10\t 45000 images for training \t 500 images for validation\t'
    )
    print('10000 images for testing\t no normalize applied in data_transform')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')
    full_dataset = CIFAR10(root='./data',
                           train=True,
                           download=True,
                           transform=train_transform)
    test_dataset = CIFAR10(root='./data',
                           train=False,
                           download=True,
                           transform=test_transform)

    max_threads = 2 if num_workers < 2 else num_workers
    if max_threads >= 6:
        val_threads = 2
        train_threads = max_threads - val_threads
    else:
        val_threads = 1
        train_threads = max_threads - 1
    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0 - validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset, split_end=split)
        val_dataset = DatasetSplitter(full_dataset, split_start=split)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size,
                                                   num_workers=train_threads,
                                                   pin_memory=True,
                                                   shuffle=True)
        valid_loader = torch.utils.data.DataLoader(val_dataset,
                                                   batch_size,
                                                   num_workers=val_threads,
                                                   pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(full_dataset,
                                                   batch_size,
                                                   num_workers=8,
                                                   pin_memory=True,
                                                   shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=True)

    return train_loader, valid_loader, test_loader


def cifar100_dataloaders(batch_size=128,
                         data_dir='datasets/cifar100',
                         num_workers=2):

    cifar_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    cifar_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transform.Normalize(cifar_mean, cifar_std)])

    print(
        'Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t'
    )
    print('10000 images for testing\t no normalize applied in data_transform')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    train_set = Subset(
        CIFAR100(data_dir,
                 train=True,
                 transform=train_transform,
                 download=True), list(range(45000)))
    val_set = Subset(
        CIFAR100(data_dir, train=True, transform=test_transform,
                 download=True), list(range(45000, 50000)))
    test_set = CIFAR100(data_dir,
                        train=False,
                        transform=test_transform,
                        download=True)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader
