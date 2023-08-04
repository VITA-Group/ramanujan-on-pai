"""
File: scores.py
Description: get various score criterias
"""
import collections
import math
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch_geometric as pyg
from scipy.sparse import coo_array


def get_eig_values(matrix: np.array, k: int = 3) -> List[float]:
    """
    get the real eig of a square matrix
    for bi-graph, the third largest eig denotes connectivity
    """
    adj_eigh_val, _ = sp.sparse.linalg.eigsh(matrix, k=k, which='LM')
    abs_eig = [abs(i) for i in adj_eigh_val]
    abs_eig.sort(reverse=True)
    return abs_eig


def delta_s(eig_first: float, eig_second: float) -> float:
    """
    calc the change in eig
    """
    if eig_first < 1:
        return -1
    ub = 2 * math.sqrt(eig_first - 1)
    return ub - eig_second, ub


def delta_r(avg_deg_left: float, avg_deg_right: float,
            eig_second: float) -> float:
    """
    calc the change in degree
    """
    ub = math.sqrt(avg_deg_left - 1) + math.sqrt(avg_deg_right - 1)
    return ub - eig_second, ub


def random_bound(eig_second: float, total_edges: int, dim_in: int,
                 dim_out: int, avg_left_degree: float, num_left: int,
                 num_right: int):
    """determine if graph can be considered a random graph.
    function is given as:
        (\hat{\mu)^2/4 + 1)*sqrt(num_left*num_right) - abs(total_edges - davg/n * num_left *
        num_right
        if differences <= 0 graph is considered random
        if larger > 0 then not

    :eig_second: TODO
    :total_edges: TODO
    :num_left: TODO
    :num_right: TODO
    :returns: TODO

    """
    ls = (eig_second**2 / 4 + 1) * math.sqrt(num_left * num_right)
    rs = abs(total_edges - (avg_left_degree / dim_out) *
             (num_left * num_right))
    ret = (ls - rs) / (dim_in * dim_out)
    return ret


def filter_zero_degree(graph: pyg.data.Data,
                       n: int) -> Tuple[pyg.data.Data, int]:
    """
    filter zero degree so not to affect d_avg_l/d_avg_r
    """
    degree = pyg.utils.degree(graph.edge_index[0], graph.num_nodes)

    non_zero_mask = degree != 0.0
    if non_zero_mask.sum() == degree.size(0):
        return graph, n

    dim_in = non_zero_mask[0:n].sum()
    node_to_keep = torch.nonzero(non_zero_mask).squeeze()
    subgraph = pyg.utils.subgraph(node_to_keep,
                                  graph.edge_index,
                                  graph.edge_attr,
                                  num_nodes=graph.num_nodes,
                                  relabel_nodes=True)

    subgraph = pyg.data.Data(edge_index=subgraph[0],
                             edge_attr=subgraph[1],
                             num_nodes=node_to_keep.size(0))

    return subgraph, dim_in


def ramanujan_score(layer: dict) -> Tuple[float]:
    """
    compute the spectral gap base on ramanujan principle.
    return two different estimation and the two eig values:
        The first estimate the differences of bound between the first and second eig values
        The second estimate the differences of bound between avg degree and second eig values  
    """
    graph = layer['graph']

    dim_in = layer['dim_in']
    graph, dim_in = filter_zero_degree(graph, dim_in)

    degree = pyg.utils.degree(graph.edge_index[0], graph.num_nodes)

    d_avg_l = degree[0:dim_in].mean()
    d_avg_r = degree[dim_in::].mean()
    if d_avg_l >= 3 and d_avg_r >= 3:
        edge_index = graph.edge_index.numpy()
        num_nodes = graph.num_nodes
        adj_matrix = coo_array(
            (np.ones_like(edge_index[0]), (edge_index[0], edge_index[1])),
            shape=(num_nodes, num_nodes),
            dtype=np.float32)
        m_eig_vals = get_eig_values(adj_matrix)
        t2_m, t1_m = m_eig_vals[-1], m_eig_vals[0]
        sm, sm_ub = delta_s(t1_m, t2_m)
        rm, rm_ub = delta_r(d_avg_l, d_avg_r, t2_m)
        expansion_ratio = (degree.size(0) - dim_in) / dim_in
        if 'dim_out' in layer:
            randomness_factor = random_bound(t2_m, graph.edge_index.size(1),
                                             layer['dim_in'], layer['dim_out'],
                                             d_avg_l, dim_in,
                                             degree.size(0) - dim_in)
        else:
            randomness_factor = None
    else:
        sm = sm_ub = rm = rm_ub = None
        t1_m = t2_m = None
        expansion_ratio = None
        randomness_factor = None

    return (sm, sm_ub, rm, rm_ub, t1_m, t2_m, expansion_ratio,
            randomness_factor)
    # layer collapsed we skip calculation


def pair_layers(layernames: Union[List[str], List[Tuple]]) -> List[str]:
    """
    get sequential pairing layer for resnet type model
    params:
        layernames: list of names in already seq order in resnset.
    return pair of names
    """
    pairs = []
    for i in range(1, len(layernames)):
        cur = layernames[i]
        prev = layernames[i - 1]
        cur_name = cur if isinstance(cur, str) else cur[0]
        prev_name = prev if isinstance(prev, str) else prev[0]

        if cur == 'fc' or prev == 'fc':
            continue
        if 'classifier' in cur and 'features' in prev:
            continue

        if 'downsample' in cur_name:
            components = cur_name.split('.')
            sublayer = int(components[1])
            if sublayer == 0:
                for j in range(i - 1, -1, -1):
                    past_layer = layernames[j]
                    past_layer_name = past_layer if isinstance(
                        past_layer, str) else past_layer[0]
                    past_layer_comp = past_layer_name.split('.')
                    if past_layer_comp[0] != components[0]:
                        pairs.append([past_layer, cur])
                        break
            else:
                for j in range(i - 1, -1, -1):
                    past_layer = layernames[j]
                    past_layer_name = past_layer if isinstance(
                        past_layer, str) else past_layer[0]
                    past_layer_comp = past_layer_name.split('.')
                    if int(past_layer_comp[1]) < sublayer:
                        pairs.append([past_layer, cur])
                        break
        else:
            pairs.append([prev, cur])
            if 'downsample' in prev_name:
                pairs.append([pairs[-3][-1], cur])
    return pairs


def get_degrees(graph: pyg.data.Data) -> Union[torch.Tensor, None]:
    """get degrees of a graphs if there are edges in this graph

    :graphs: TODO
    :returns: TODO

    """
    edges = graph.edge_index
    if edges[0].size(0) > 0:
        return pyg.utils.degree(graph.edge_index[0])
    return None


def copeland_score(layer1: dict, layer2: dict) -> float:
    """
    get a copeland score. input and output node degree are normalized
    across input and output masks. this modified copeland scores in between [0, R] and
    is the quotient between normallzed out and in degree.

    1.0 = good
    >1.0=compression
    <1.0=bottlneck
    params:
        layer1: dict type produced by generate_bipartie_graph
        layer2: ~same type as layer1. And should be the sequential layer after layer1
    return:
        the modified copeland score
    """
    l1_out = layer1['dim_out']
    l2_in = layer2['dim_in']
    k_size = l2_in // l1_out
    l1deg = get_degrees(layer1['graph'])
    l2deg = get_degrees(layer2['graph'])
    if l1deg is None or l2deg is None:
        return 0.0
    in_deg = l1deg[-l1_out::]
    out_deg = l2deg[0:l2_in].reshape(l1_out, k_size)
    ###
    in_deg_norm = in_deg / (l1deg[0:layer1['dim_in']] != 0).sum()
    out_deg_norm = out_deg / (l2deg[layer2['dim_in']::] != 0).sum()
    ###
    mask = in_deg != 0
    ###
    out_deg_m = out_deg_norm[mask]
    in_deg_m = in_deg_norm[mask]
    throughput = out_deg_m / in_deg_m.tile(k_size).view(
        k_size, in_deg_m.size(0)).T
    return throughput.mean().item()


def channel_overlap_coefs(layer: dict, in_channels: int) -> float:
    """
    get the mean channel's kernel overlap coefs
    overlap coef is defined to be the ratio between intersection(A,B) / min(|A|, |B|)

    params:
        layer: the layer dictionary generated by generate_bipartie_graph.py
        in_channels: the input channels of l1's
    """
    k_size = layer['dim_in'] // in_channels
    channel_coefs = []
    for channel in range(in_channels):
        overlap_nodes = None
        min_graph = float("inf")
        for k in range(k_size):
            node = channel * k_size + k
            mask = layer['graph'].edge_index[0] == node
            tgt_nodes = set(layer['graph'].edge_index[1, mask].tolist())
            min_graph = min(min_graph, mask.float().sum().item())
            if overlap_nodes is None:
                overlap_nodes = tgt_nodes
            else:
                overlap_nodes = overlap_nodes.intersection(tgt_nodes)
        if min_graph == 0:
            continue
        coef = len(overlap_nodes) / min_graph
        channel_coefs.append(coef)
    if len(channel_coefs) > 0:
        return sum(channel_coefs) / len(channel_coefs)
    return 0


def compatibility_ratio(layer1: dict, layer2: dict) -> float:
    """Get iou of input / output degree between two layers.

    :layer1: TODO
    :layer2: TODO
    :returns: TODO

    """
    l1_out = layer1['dim_out']
    l2_in = layer2['dim_in']
    k_size = l2_in // l1_out
    l1deg = get_degrees(layer1['graph'])
    l2deg = get_degrees(layer2['graph'])

    if l1deg is None or l2deg is None:
        return 0
    ####
    in_deg = l1deg[-l1_out::]
    out_deg = l2deg[0:l2_in].reshape(l1_out, k_size).mean(dim=-1)
    ####
    in_mask = in_deg != 0.0
    out_mask = out_deg != 0.0
    compatibility = (in_mask
                     & out_mask).float().sum().item() / in_mask.sum().item()
    return compatibility


def connectivity_bound(layer: dict) -> float:
    """Get the connectivity 1144bound of an irregular graph.
    There are 2 bounds we are interested in.
    1. the first eig value of the incident matrix (this is returned by the ramanujan_score function)
    2. from Ramanujan property, d-regularity can be estimated at sqrt(1-d_avg_left) +
    sqrt(1-d_avg_right)
    :layer: TODO
    :returns: return the estimated bound 

    """
    graph = layer['graph']
    degree = pyg.utils.degree(graph.edge_index[0], graph.num_nodes)
    d_avg_l = degree[0:layer['dim_in']].mean()
    d_avg_r = degree[layer['dim_in']::].mean()
    return math.sqrt(d_avg_l - 1) + math.sqrt(d_avg_r - 1)


def iterative_mean_spectral_gap(layer: dict):
    """Calculate the total variance of bounds on every sub-graph of layers that has at-least
    d-in/out degree

    :layer: TODO
    :returns: TODO

    """
    sms = []
    sms_norm = []
    rms = []
    rms_norm = []
    expansion_ratios = []
    for left_regular_subgraph in find_d_left_regular(layer):
        ram_scores = ramanujan_score(left_regular_subgraph)

        if ram_scores[0] == None:  # == (-1, -1, -1, 0, -1):
            continue
        else:
            sms.append(ram_scores[0])
            sms_norm.append(ram_scores[0] / ram_scores[1])
            rms.append(ram_scores[2])
            rms_norm.append(ram_scores[2] / ram_scores[3])
            expansion_ratios.append(ram_scores[-2])
    if len(sms) == 0 and len(rms) == 0:
        return None, None, None, None, None
    mean_sm = sum(sms) / len(sms)
    mean_rm = sum(rms) / len(rms)
    mean_rm_norm = sum(rms_norm) / len(rms)
    mean_sm_norm = sum(sms_norm) / len(rms)
    max_ep = max(expansion_ratios)
    mean_ep = sum(expansion_ratios) / len(expansion_ratios)
    return mean_sm, mean_rm, mean_sm_norm, mean_rm_norm, len(
        sms), max_ep, mean_ep


def find_d_left_regular(layer: dict, mindegree: int = 3):
    """return a subgraph of graph which contains at minimum d in/out degree

    :returns: TODO

    """
    graph = layer['graph']
    degree = pyg.utils.degree(graph.edge_index[0], graph.num_nodes)
    d_l = degree[0:layer['dim_in']]
    d_r = degree[layer['dim_in']::]
    degrees = collections.Counter(d_l.tolist())
    for degree, count in degrees.items():
        if count < mindegree or degree == 0.0: continue
        node_mask = d_l == degree
        non_zero_idx = torch.nonzero(node_mask).squeeze()

        right_nodes = torch.arange(d_r.size(0)) + layer['dim_in']
        node_to_keep = torch.cat([non_zero_idx, right_nodes], dim=0)
        # d-left regular bipartie graph
        subgraph = pyg.utils.subgraph(node_to_keep,
                                      graph.edge_index,
                                      graph.edge_attr,
                                      num_nodes=graph.num_nodes,
                                      relabel_nodes=True)
        ret = {
            'dim_in':
            count,
            'graph':
            pyg.data.Data(
                edge_index=subgraph[0],
                edge_attr=subgraph[1],
                num_nodes=node_to_keep.size(0),
            ),
        }
        yield ret
