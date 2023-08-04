import os
import shutil
import sys

import common_models as CM
import torch
from common_models.utils import init


def link_latest(dst):
    sub_folders = os.listdir(dst)
    for sub_folder in sub_folders:
        sympath = os.path.join(dst, sub_folder, "latest")
        if os.path.islink(sympath):
            os.remove(sympath)

        path = [
            os.path.join(dst, sub_folder, i)
            for i in os.listdir(os.path.join(dst, sub_folder))
        ]
        path2sort = list(filter(lambda x: len(os.listdir(x)) > 0, path))
        path2sort.sort(key=lambda x: os.path.getmtime(x))
        os.symlink(os.path.basename(path2sort[-1]), sympath)


def generate_init_weights(seed, dst, num_classes):

    torch.manual_seed(int(seed))
    os.makedirs(dst, exist_ok=True)
    num_classes = [int(i) for i in num_classes.split(",")]

    for model_name, fn in CM.models.items():
        for n in num_classes:
            name = f"{model_name}_c-{n}_seed-{seed}.pth.tar"
            path = os.path.join(dst, name)
            model = init(fn(pretrained=False, num_classes=n))
            torch.save(model.state_dict(), path)
