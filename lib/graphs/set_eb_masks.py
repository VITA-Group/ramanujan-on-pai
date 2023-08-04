"""
convert Early bird masks into unstructured masks
"""
import json
import os
import os.path as osp
import sys

import torch
from torch import nn
from tqdm import tqdm as tqdm

from common_models.models import models
# import pdb


def pruning(model: torch.nn.Module, percent: float) -> torch.Tensor:
    """
    I stole this function of EarlyBird
    """
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    y, _ = torch.sort(bn)
    thre_index = int(total * percent)
    thre = y[thre_index]

    mask = torch.zeros(total)
    index = 0
    for _, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.numel()
            weight_copy = m.weight.data.abs().clone()
            _mask = weight_copy.gt(thre).float()
            mask[index:(index + size)] = _mask.view(-1)
            index += size

    return mask


def struct_unstruct_conversion(model: torch.nn.Module,
                               sparsity: float) -> torch.nn.Module:
    """convert unstruct mask to unstruct mask

    :model: TODO
    :sparsity: float ratio of for remaining params
    :returns: TODO

    """
    bn_mask = pruning(model, sparsity)
    cur_idx = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            outc = m.weight.shape[0]
            sub_mask = bn_mask[cur_idx:cur_idx + outc]
            m.weight.data = m.weight.data * sub_mask
            cur_idx += outc
    assert cur_idx == bn_mask.size(0)
    return model


def main(path: str, model_type: str, num_classes: int, sparsity: float,
         dst: str):
    """
    main function for conversion
    """
    model = models[model_type](num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    struct_unstruct_conversion(model, sparsity)
    torch.save(model.state_dict(), dst)


if __name__ == "__main__":
    _, eb_mask_directory, dst_folder = sys.argv
    masks = os.listdir(eb_mask_directory)
    masks = list(filter(lambda x: "mask" in x, masks))

    with open(osp.join(eb_mask_directory, 'config.txt'), 'rt') as handle:
        config = json.load(handle)

    model_name = config['arch']
    n_class = 10 if config['dataset'] == "cifar10" else 100
    sparsity = 1 - config['sparsity']
    os.makedirs(dst_folder, exist_ok=True)
    for m in tqdm(masks, total=len(masks), desc=f'{eb_mask_directory}'):
        path = osp.join(eb_mask_directory, m)
        dst = osp.join(dst_folder, m)
        main(path, model_name, n_class, sparsity, dst)
