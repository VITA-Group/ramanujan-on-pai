import collections
import os
import os.path as osp
import random
import sys

import pandas as pd
import torch
from tqdm import tqdm


def get_layers_df(graphs_dir: str) -> pd.DataFrame:
    """
    generate dataframe structure from graphs in graphs_dir 
    """
    layers_df = collections.defaultdict(list)
    for mask in os.listdir(graphs_dir):
        graphs = torch.load(osp.join(graphs_dir, mask))

        for layer, info in graphs.items():
            (sm, rm, t1m), (sw, rw, t1w) = info['ram_scores']
            layers_df['mask_no'].append(int(mask.split('.')[0].split('_')[0]))
            layers_df['layer'].append(layer)
            layers_df['sparsity'].append(info['sparsity'])
            layers_df['sm'].append(sm)
            layers_df['rm'].append(rm)
            layers_df['sw'].append(sw)
            layers_df['rw'].append(rw)
            layers_df['t1m'].append(t1m)
            layers_df['t1w'].append(t1w)

    layers_df = pd.DataFrame.from_dict(layers_df)

    return layers_df


def sort_by_key(data: pd.DataFrame, key: str):
    """sort data by keys

    :data: TODO
    :key: TODO
    :returns: TODO

    """
    data = data.sort_values(by=key, ascending=False)
    ret = {}
    scores = {}
    for layer in data.layer.unique():
        subset = data.loc[data.layer == layer]
        ordered_masks = subset.mask_no.tolist()
        ordered_scores = subset[key].tolist()
        ret[layer] = ordered_masks
        scores[layer] = ordered_scores
    scores = pd.DataFrame.from_dict(scores).T.mean().tolist()
    ret['mask_scores'] = scores

    return pd.DataFrame.from_dict(ret)


def generate_ranked_masks(data: pd.DataFrame,
                          masks_dir: str,
                          dst: str,
                          num_samples: int = 10):
    """generate new model's masks ranked by some criteria.

    :data: TODO
    :masks_dir: TODO
    :dst: TODO
    :num_samples: TODO
    :returns: TODO

    """
    samples = random.sample(range(len(data)), num_samples)
    samples.sort()
    for index in tqdm(samples, total=len(samples), desc=dst):
        masks_dict = collections.OrderedDict()
        row = data.iloc[index]
        row, row_scores = row[0:-1], row[-1]
        mask_no = row.unique().tolist()

        for num in mask_no:
            layers = [
                i + '.weight' for i in row.loc[row == num].index.tolist()
            ]
            n_mask = torch.load(osp.join(masks_dir,
                                         str(int(num)) + "_mask.pth.tar"),
                                map_location='cpu')
            for layer in layers:
                masks_dict[layer] = (n_mask[layer] != 0.0).float()
        torch.save(masks_dict, osp.join(dst, f"{index}_mask.pth.tar"))


def main(graphs: str, masks: str, key: str, dst: str):
    """
    run main algorithm
    param:
        graphs: directory to bi-partited graph structure generated by generate_bipartie_graphs.py
        masks: directory to raw mask folder referenced by our graphs
        key: the criteria we would like to our mask for
        dst: where to save
    """
    dst = osp.join(dst, key)
    os.makedirs(dst, exist_ok=True)
    data_frame = get_layers_df(graphs)
    layers_sort = sort_by_key(data_frame, key)
    generate_ranked_masks(layers_sort, masks, dst, len(layers_sort))
    layers_sort.to_csv(osp.join(dst, "masks.csv"))


if __name__ == "__main__":
    _, graphs_dir, masks_dir, dst = sys.argv

    assert osp.isdir(graphs_dir)
    keys = ['sm', 'rm']
    for k in keys:
        main(graphs_dir, masks_dir, k, dst)
