import os
import os.path as osp
import sys
from typing import List

import pandas as pd
import torch
from scores import pair_layers
from scores import ramanujan_score
from tqdm import tqdm


def get_model_pairs(graphs_dir: str) -> List[str]:
    """
    generate sequential layer pairs
    :graphs_dir: TODO
    :returns: TODO

    """
    files = list(os.listdir(graphs_dir))
    files = list(filter(lambda x: "mask.pth.tar" in x, files))
    sample = torch.load(osp.join(graphs_dir, files[0]))
    pairs = pair_layers(list(sample.keys()))
    return pairs


def rank_pairs(pair: List[str], masks_dir: str) -> pd.DataFrame:
    """
    for each layer pair, generate scores between every timestep
    """
    pass


def main(graphs_dir, masks_dir):
    """
    we take graphs generated by  generate_bipartie_graphs and raw masks weights, generate a csv
    ranking data of ramscores for every layers pairs across time

    :graphs_dir: TODO
    :masks_dir: TODO
    :returns: TODO

    """
    layer_pairs = get_model_pairs(graphs_dir)


if __name__ == "__main__":
    _, graph_directory, mask_directory = sys.argv

    main(graph_directory, mask_directory)
