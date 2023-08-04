import collections
import json
import os
import os.path as osp
import sys
from multiprocessing import Process
from typing import Any
from typing import Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from generate_unstructured_bipartie_graphs import from_networkx
from generate_unstructured_bipartie_graphs import generate_nx_graphs
from generate_unstructured_bipartie_graphs import mlp_to_network as unstruct_mlp_to_network
from generate_unstructured_bipartie_graphs import update_collections
from set_eb_masks import pruning

from common_models.models import models


def mlp_to_network(neuron_network, draw=False):
    """mlp_to_network

    Args:
        neuron_network (dict): key is layer name, and value is a 2D
            matrices
        draw (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    graphs = collections.defaultdict(dict)

    idx_out_start = 0
    for l, (name, data) in enumerate(neuron_network.items()):
        weight, sub_mask = data
        nx_new, dim_in, dim_out, idx_in_start, idx_out_start = generate_nx_graphs(
            weight, l, idx_out_start, masks=sub_mask)
        graphs[name].update(
            update_collections(nx_new, idx_in_start, idx_out_start, dim_in,
                               dim_out))
    return graphs


def generate_bipartie_graphs(m: nn.Module,
                             masks: torch.Tensor) -> collections.defaultdict:
    neuron_network = {}
    c_idx = 0
    for name, module in m.named_modules():
        if isinstance(module, (nn.Conv2d)):
            weight = module.weight_mask.detach()
            c_out, c_in, h, w = weight.shape
            sub_mask = masks[c_idx:c_idx + c_out]
            sub_mask = sub_mask.tile(c_in, 1).T.tile(h * w).reshape(
                c_out, c_in, h, w)
            neuron_network[name] = (weight * sub_mask).numpy()
            c_idx += c_out
    return unstruct_mlp_to_network(neuron_network)


def process(path, model, num_classes, dst, ratio):
    print(f'working on {path}')
    m = models[model](num_classes)

    m.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
    model_mask = pruning(m, ratio)
    print(f"model density: {model_mask.sum() / model_mask.size(0)}")
    graphs = generate_bipartie_graphs(m, model_mask)
    torch.save(graphs, osp.join(dst, osp.basename(path)))
