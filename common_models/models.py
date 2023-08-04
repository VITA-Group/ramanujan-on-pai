import torch
import torchvision

from .utils import init
from .utils import set_seed
from .utils import stripping_bias
from .utils import stripping_skip_connections

__all__ = ['resnet18', 'resnet34', 'resnet50', 'vgg16', 'models']

import os

RANDOM_PATH = "/home/dnh754/nndst/common_models/random_weights/"
os.makedirs(RANDOM_PATH, exist_ok=True)


def modify_model(func):
    """modify the functionality with some argument 
    stripping away its skip connections/ bias etc

    :func: should be any model generator function
    :returns: TODO

    """

    def with_modification(*args, **kwargs):
        model = func(*args, **kwargs)

        strip_bias = kwargs.get('strip_bias', False)
        strip_skip = kwargs.get('strip_skip', False)
        if strip_bias:
            print(f"stripping_bias from {func.__name__}")
            stripping_bias(model)
        if strip_skip:
            if 'resnet' in func.__name__:
                print(f"stripping_bias from {func.__name__}")
                stripping_skip_connections(model)
            else:
                print(f"skipping strip skip function for {func.__name__}")
        return model

    return with_modification


def get_name(model_name, num_classes, seed):
    return f"{model_name}_c-{num_classes}_seed-{seed}.path.tar"


@modify_model
def resnet18(*args, **kwargs):
    n = resolve(args, kwargs)
    model = torchvision.models.resnet18(num_classes=n)
    seed = kwargs.get('seed', 69)
    try_load_random(model,
                    os.path.join(RANDOM_PATH, get_name('resnet18', n, seed)),
                    seed)
    return model


@modify_model
def resnet34(*args, **kwargs):
    n = resolve(args, kwargs)
    model = torchvision.models.resnet34(num_classes=n)  #*args, **kwargs)
    seed = kwargs.get('seed', 69)
    try_load_random(model,
                    os.path.join(RANDOM_PATH, get_name('resnet34', n, seed)),
                    seed)
    return model


@modify_model
def resnet50(*args, **kwargs):
    n = resolve(args, kwargs)
    model = torchvision.models.resnet50(num_classes=n)  #*args, **kwargs)
    seed = kwargs.get('seed', 69)
    try_load_random(model,
                    os.path.join(RANDOM_PATH, get_name('resnet50', n, seed)),
                    seed)
    return model


@modify_model
def vgg16(*args, **kwargs):
    n = resolve(args, kwargs)
    model = torchvision.models.vgg16_bn(num_classes=n)  #*args, **kwargs)
    seed = kwargs.get('seed', 69)
    try_load_random(model, os.path.join(RANDOM_PATH,
                                        get_name('vgg16', n, seed)), seed)
    return model


def resolve(args, kwargs):
    n = kwargs.get("num_classes", None)
    if n is None:
        dataset = kwargs.get("dataset", None)
        if dataset is None:
            if len(args):
                n = args[-1]
            else:
                n = 1000
        else:
            if dataset.lower() == 'cifar10':
                n = 10
            elif dataset.lower() == 'cifar100':
                n = 100
            elif dataset.lower() == 'imagenet':
                n = 1000
    return n


def try_load_random(model, path, seed, verbose=False):
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))
    else:
        print(f"{path} does not exists, generating one now")
        set_seed(seed)
        init(model)
        torch.save(model.state_dict(), path)


models = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "vgg16": vgg16
}
